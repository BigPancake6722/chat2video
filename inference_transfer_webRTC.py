import os
import sys
import wave
import librosa
import asyncio
import uvicorn
import numpy as np
from io import BytesIO
import soundfile as sf
import warnings 
import json
import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, AudioStreamTrack
from pydantic import BaseModel
from typing import AsyncGenerator, Tuple, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.append("/root/chat2video/chat2audio/")
sys.path.append("/root/chat2video/sync-gaussian-talker/")
from app_stream import llm_to_tts_stream_with_text as llm_to_tts_stream
from render_transfer import main as audio2video

app = FastAPI()

# 添加CORS中间件解决跨域问题
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


# SRS服务器配置
SRS_SERVER = "47.111.96.21"
SRS_API_PORT = 1985
SRS_RTC_PORT = 8000

@app.options("/chat2video")
async def handle_options():
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        }
    )


# 全局变量存储PeerConnection
peer_connections = {}

class VideoStreamTrackImpl(VideoStreamTrack):
    def __init__(self, track_id):
        super().__init__()
        self.track_id = track_id

class AudioStreamTrackImpl(AudioStreamTrack):
    def __init__(self, track_id):
        super().__init__()
        self.track_id = track_id

async def process_tts_stream(
    tts_stream: AsyncGenerator[bytes, None],
    media_type: str = "wav",
    target_sr: int = 16000,
    min_chunk_size: int = 1024
) -> AsyncGenerator[Tuple[np.ndarray, int], None]:
    buffer = BytesIO()
    header_received = False
    pending_data = b""
    
    async for chunk in tts_stream:
        if not chunk:
            continue
            
        pending_data += chunk
        
        if len(pending_data) < min_chunk_size and not header_received:
            continue
            
        buffer.write(pending_data)
        pending_data = b""
        
        if media_type == "wav" and not header_received:
            if buffer.getbuffer().nbytes < 44:
                continue
            header_received = True
        
        buffer.seek(0)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio_data, sr = sf.read(buffer)
                
                if len(audio_data) > 0:
                    if audio_data.ndim > 1:
                        audio_data = np.mean(audio_data, axis=1)
                    
                    if sr != target_sr:
                        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
                    
                    print(f"✅ 产生音频块: {audio_data.shape} samples @ {target_sr}Hz")
                    yield audio_data, target_sr
                    
                    remaining = buffer.getvalue()[buffer.tell():]
                    buffer = BytesIO()
                    buffer.write(remaining)
                    
        except Exception as e:
            print(f"⚠️ 音频解析暂缓: {str(e)}")
        finally:
            buffer.seek(0, 2)
        
        await asyncio.sleep(0)

async def create_srs_stream(session_id: str) -> str:
    url = f"http://{SRS_SERVER}:{SRS_API_PORT}/rtc/v1/publish/"
    data = {
        "api": url,
        "streamurl": f"webrtc://{SRS_SERVER}:{SRS_RTC_PORT}/live/{session_id}",
        "sdp": ""
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=500, detail="Failed to create SRS stream")
            response = await resp.json()
            return response.get("sdp")

class TTSRequest(BaseModel):
    input: str
    speaker: str = "Firefly"
    model_id: str = "Chinese1"
    prompt: Optional[str] = None
    session_id: str  # 前端提供的唯一会话ID

@app.post("/chat2video")
async def api_llm_tts_stream(request: TTSRequest):
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Expose-Headers": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS"
    }
    # 为每个会话创建新的PeerConnection
    pc = RTCPeerConnection()
    peer_connections[request.session_id] = pc
    
    # 创建音视频轨道
    video_track = VideoStreamTrackImpl(request.session_id)
    audio_track = AudioStreamTrackImpl(request.session_id)
    pc.addTrack(video_track)
    pc.addTrack(audio_track)
    
    # 创建SRS流
    sdp = await create_srs_stream(request.session_id)
    offer = RTCSessionDescription(sdp=sdp, type="offer")
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    tts_stream = llm_to_tts_stream(request.input, request.speaker, request.prompt, media_type="wav")
    video_chunks = []
    
    async def generate_video():
        async for text, raw_chunk in tts_stream:
            async for audio_chunk, sr in process_tts_stream(mock_stream([raw_chunk]), min_chunk_size=0):
                from model_params import model_params
                video_bytes = audio2video(
                    ['--model_path', model_params[request.model_id]['model_path'],
                     '--source_path', model_params[request.model_id]['source_path'],
                     '--batch', '16'],
                    audio_chunk
                )
                
                if isinstance(video_bytes, BytesIO):
                    video_bytes = video_bytes.getvalue()
                video_chunks.append(video_bytes)
                yield video_bytes
    
    return StreamingResponse(
        generate_video(),
        media_type="video/mp4",
        headers=headers
    )

@app.on_event("shutdown")
async def shutdown_event():
    # 关闭所有PeerConnection
    for pc in peer_connections.values():
        await pc.close()
    peer_connections.clear()

async def mock_stream(chunks):
    for chunk in chunks:
        yield chunk

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)