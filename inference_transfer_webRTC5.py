import sys
import time
import random
import asyncio
import uvicorn
import librosa
import numpy as np
import soundfile as sf
from io import BytesIO
from pydantic import BaseModel
from typing import AsyncGenerator
from av import AudioFrame, VideoFrame
from fastapi.staticfiles import StaticFiles
from fastapi.responses import  RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request
from aiortc.mediastreams import AudioStreamTrack, VideoStreamTrack
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, RTCRtpSender

sys.path.append("/root/chat2video/chat2audio/")
sys.path.append("/root/chat2video/sync-gaussian-talker/")
from app_stream3 import llm_to_tts_stream_dual
from render_transfer1 import main as audio2video
from humanPlayer2 import HumanPlayer
from test import ndarray_audio_to_mp4

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置参数
AUDIO_SAMPLE_RATE = 32000
VIDEO_FPS = 25
players = {}  # {session_id: HumanPlayer}

async def process_dual_tts_stream(
    tts_stream: AsyncGenerator[tuple[str, bytes, bytes], None],
    target_sr: int = 32000
) -> AsyncGenerator[tuple[np.ndarray, int, str, bytes], None]:
    """处理双音频TTS流，返回音频数据、采样率、对应文本以及AAC音频流"""
    buffer = BytesIO()
    
    async for text, wav_chunk, pcm_chunk in tts_stream:
        if not wav_chunk:
            continue

        # 处理WAV音频用于视频生成
        buffer.write(wav_chunk)
        buffer.seek(0)
        
        try:
            audio_data, sr = sf.read(buffer)

            if len(audio_data) > 0:
                if audio_data.ndim > 1:
                    audio_data = np.mean(audio_data, axis=1)

                if sr != target_sr:
                    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
                print("声之形", audio_data.shape)
                yield text, target_sr, audio_data, pcm_chunk

        except Exception as e:
            print(f"⚠️ 音频解析暂缓: {str(e)}")
        finally:
            buffer = BytesIO()  # 重置缓冲区
            await asyncio.sleep(0)


class TTSRequest(BaseModel):
    input: str
    speaker: str = "Firefly"
    model_id: str = "Chinese1"
    prompt: str = None
    session_id: str = None

class OfferRequest(BaseModel):
    sdp: str
    type: str

def randN(N: int) -> int:
    return random.randint(10**(N-1), 10**N -1)

@app.post("/offer")
async def handle_offer(request: OfferRequest):
    offer = RTCSessionDescription(sdp=request.sdp, type=request.type)
    session_id = str(randN(6))
    
    # 初始化 Player
    player = HumanPlayer(output_fps=VIDEO_FPS)
    players[session_id] = player
    
    # 配置编解码器
    codec_prefs = [
        c for c in RTCRtpSender.getCapabilities("video").codecs
        if c.name in ["H264", "VP8", "rtx"]
    ]
    for t in player.pc.getTransceivers():
        if t.kind == "video":
            t.setCodecPreferences(codec_prefs)
    
    # 处理信令
    await player.pc.setRemoteDescription(offer)
    answer = await player.pc.createAnswer()
    await player.pc.setLocalDescription(answer)
    
    # 连接状态监听
    @player.pc.on("connectionstatechange")
    async def _on_state_change():
        if player.pc.connectionState in ["closed", "failed"]:
            player.close()
            del players[session_id]
    player.start_silent()
    return {
        "sdp": player.pc.localDescription.sdp,
        "type": player.pc.localDescription.type,
        "sessionid": session_id
    }

@app.post("/chat2video")
async def api_llm_tts_stream(request: TTSRequest):
    player = players.get(request.session_id)
    if not player:
        raise HTTPException(404, "Session not found")
    try:
        # 获取TTS流
        tts_stream = llm_to_tts_stream_dual(
            request.input, request.speaker, request.prompt
        )
        
        async def process_stream():
            async for text, sr, wav_data, pcm_data in process_dual_tts_stream(tts_stream):
                # 生成视频帧 (假设audio2video返回np.ndarray列表)
                from model_params import model_params
                params = model_params[request.model_id]
                video_frames = audio2video(
                    ['--model_path', params['model_path'],
                     '--source_path', params['source_path'],
                     '--batch', '16'],
                    wav_data
                )
                
                # 推送数据到Player
                player.stop_silent()
                player.feed_external_data(
                    rgb_frames=np.array(video_frames, dtype=np.uint8),
                    pcm_audio=pcm_data
                )
                player.start_silent()
        asyncio.create_task(process_stream())
        return {"text": "推流中"}
    
    except Exception as e:
        print(f"推流失败: {str(e)}")
        raise HTTPException(500, str(e))


@app.on_event("shutdown")
async def shutdown():
    for player in players.values():
        player.close()
    players.clear()

app.mount("/", StaticFiles(directory="web", html=True), name="static")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)