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
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, RTCRtpSender
from aiortc.mediastreams import AudioStreamTrack, VideoStreamTrack
from pydantic import BaseModel
from typing import AsyncGenerator, Tuple, List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.append("/root/chat2video/chat2audio/")
sys.path.append("/root/chat2video/sync-gaussian-talker/")
from app_stream import llm_to_tts_stream_with_text as llm_to_tts_stream
from render_transfer import main as audio2video

app = FastAPI()

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SRS服务器配置
SRS_SERVER = "47.111.96.21"
SRS_API_PORT = 1985
SRS_RTC_PORT = 8000

class MediaStreamTrackWrapper:
    """音视频轨道包装器，实现音画同步"""
    def __init__(self, kind):
        self.kind = kind
        self._queue = asyncio.Queue()
        self._timestamp = 0
        self._start_time = asyncio.get_event_loop().time()
        
    async def recv(self):
        frame = await self._queue.get()
        if frame is None:
            raise Exception("Stream ended")
        
        # 计算精确时间戳
        now = asyncio.get_event_loop().time()
        elapsed = now - self._start_time
        if self.kind == "audio":
            frame.pts = int(elapsed * 16000)  # 16kHz采样率
        else:
            frame.pts = int(elapsed * 90000)  # 视频时钟频率
        return frame

    def put_frame(self, frame):
        self._queue.put_nowait(frame)

class RTCStreamPublisher:
    """处理WebRTC推流到SRS"""
    def __init__(self):
        self.pcs = set()
        self.audio_track = MediaStreamTrackWrapper("audio")
        self.video_track = MediaStreamTrackWrapper("video")
        self.text_queue = asyncio.Queue()
        
    async def push_to_srs(self, session_id: str):
        """建立到SRS的推流连接"""
        pc = RTCPeerConnection()
        self.pcs.add(pc)
        
        # 添加音视频轨道
        pc.addTrack(self.audio_track)
        pc.addTrack(self.video_track)
        
        # 配置编解码偏好
        capabilities = RTCRtpSender.getCapabilities("video")
        preferences = [c for c in capabilities.codecs if c.mimeType in ["video/H264", "video/VP8"]]
        transceiver = pc.getTransceivers()[1]
        transceiver.setCodecPreferences(preferences)
        
        # 创建offer并发送到SRS
        await pc.setLocalDescription(await pc.createOffer())
        offer = pc.localDescription
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{SRS_SERVER}:{SRS_API_PORT}/rtc/v1/whip/?app=live&stream={session_id}",
                data=offer.sdp,
                headers={"Content-Type": "application/sdp"}
            ) as resp:
                if resp.status != 201:
                    raise HTTPException(status_code=500, detail="SRS推流失败")
                answer = await resp.text()
                await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type="answer"))
        
        return pc

    async def push_media_frame(self, audio_data: np.ndarray, video_data: bytes, text: str):
        """推送同步的音视频帧和文本"""
        # 推送音频帧
        audio_frame = AudioFrame.from_ndarray(audio_data, layout="mono", format="s16")
        self.audio_track.put_frame(audio_frame)
        
        # 推送视频帧
        video_frame = VideoFrame.from_ndarray(video_data, format="bgr24")
        self.video_track.put_frame(video_frame)
        
        # 推送文本
        await self.text_queue.put(text)
        
        # 保证音画同步
        await asyncio.sleep(0.02)  # 20ms音频帧间隔

# 全局推流器实例
rtc_publisher = RTCStreamPublisher()

async def process_tts_stream(
    tts_stream: AsyncGenerator[bytes, None],
    media_type: str = "wav",
    target_sr: int = 16000,
    min_chunk_size: int = 1024
) -> AsyncGenerator[Tuple[np.ndarray, int, str], None]:
    """处理TTS流，返回音频数据和对应文本"""
    buffer = BytesIO()
    header_received = False
    pending_data = b""
    
    async for text, chunk in tts_stream:
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
                    
                    yield audio_data, target_sr, text
                    
                    remaining = buffer.getvalue()[buffer.tell():]
                    buffer = BytesIO()
                    buffer.write(remaining)
                    
        except Exception as e:
            print(f"⚠️ 音频解析暂缓: {str(e)}")
        finally:
            buffer.seek(0, 2)
        
        await asyncio.sleep(0)

class TTSRequest(BaseModel):
    input: str
    speaker: str = "Firefly"
    model_id: str = "Chinese1"
    prompt: str = None
    session_id: str = "default_stream"  # 推流会话ID

@app.post("/chat2video")
async def api_llm_tts_stream(request: TTSRequest):
    """处理视频生成并推流到SRS"""
    # 初始化SRS推流连接
    await rtc_publisher.push_to_srs(request.session_id)
    
    tts_stream = llm_to_tts_stream(request.input, request.speaker, request.prompt, media_type="wav")
    
    async for audio_chunk, sr, text in process_tts_stream(tts_stream, min_chunk_size=0):
        # 生成视频帧
        from model_params import model_params
        video_bytes = audio2video(
            ['--model_path', model_params[request.model_id]['model_path'],
             '--source_path', model_params[request.model_id]['source_path'],
             '--batch', '16'],
            audio_chunk
        )
        
        if isinstance(video_bytes, BytesIO):
            video_bytes = video_bytes.getvalue()
        
        # 推送同步的音视频帧和文本
        await rtc_publisher.push_media_frame(
            audio_data=audio_chunk,
            video_data=video_bytes,
            text=text
        )
    
    # 返回文本流
    async def text_stream():
        while True:
            text = await rtc_publisher.text_queue.get()
            if text is None:  # 结束信号
                break
            yield f"data: {json.dumps({'text': text})}\n\n"
    
    return StreamingResponse(
        text_stream(),
        media_type="text/event-stream"
    )

async def mock_stream(chunks):
    for chunk in chunks:
        yield chunk

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)