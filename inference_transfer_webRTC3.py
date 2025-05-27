import sys
import time
import random
import asyncio
import uvicorn
import logging
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
from app_stream2 import llm_to_tts_stream_dual
from render_transfer1 import main as audio2video
from humanPlayer import HumanPlayer


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
MAX_BITRATE = 2000 * 1000  # 2Mbps
player = None

class TTSRequest(BaseModel):
    input: str
    speaker: str = "Firefly"
    model_id: str = "Chinese1"
    prompt: str = None
    session_id: str = "default_stream"

class OfferRequest(BaseModel):
    sdp: str
    type: str

pcs = list()
async def process_dual_tts_stream(
    tts_stream: AsyncGenerator[tuple[str, bytes, bytes], None],
    target_sr: int = 32000,
    min_chunk_size: int = 1024
) -> AsyncGenerator[tuple[np.ndarray, int, str, bytes], None]:
    """处理双音频TTS流，返回音频数据、采样率、对应文本以及AAC音频流"""
    buffer = BytesIO()
    
    async for text, wav_chunk, opus_chunk in tts_stream:
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

                yield text, target_sr, audio_data, opus_chunk  # 返回opus流用于推流

        except Exception as e:
            print(f"⚠️ 音频解析暂缓: {str(e)}")
        finally:
            buffer = BytesIO()  # 重置缓冲区
            await asyncio.sleep(0)

def randN(N)->int:
    '''生成长度为 N的随机数 '''
    min = pow(10, N - 1)
    max = pow(10, N)
    return random.randint(min, max - 1)

@app.post("/offer")
async def handle_offer(request: OfferRequest):
    offer = RTCSessionDescription(sdp=request.sdp, type=request.type)
    session_id = randN(6)
    pc = RTCPeerConnection()
    pcs.append(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.remove(pc)
        if pc.connectionState == "closed":
            pcs.remove(pc)

    player = HumanPlayer()
    audio_sender = pc.addTrack(player.create_audio_track())  # 添加音频轨道
    video_sender = pc.addTrack(player.create_video_track())  # 添加视频轨道
    capabilities = RTCRtpSender.getCapabilities("video")
    preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
    transceiver = pc.getTransceivers()[1]
    transceiver.setCodecPreferences(preferences)
    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, 
            "type": pc.localDescription.type, 
            "session_id": session_id
            }

@app.post("/chat2video")
async def api_llm_tts_stream(request: TTSRequest):
    try:
        pc = pcs[0]
        audio_sender = next((s for s in pc.getSenders() if s.track and s.track.kind == "audio"), None)
        video_sender = next((s for s in pc.getSenders() if s.track and s.track.kind == "video"), None)
        if not audio_sender or not video_sender:
            raise HTTPException(status_code=500, detail="未找到音视频轨道")
        
        audio_track = audio_sender.track
        video_track = video_sender.track
        # 获取TTS流
        tts_stream = llm_to_tts_stream_dual(request.input, request.speaker, request.prompt)

        async def generate_and_push():
            async for text, sample_rate, wav_chunk, opus_chunk in process_dual_tts_stream(tts_stream):
                # 生成视频帧
                from model_params import model_params
                video_images = audio2video(
                    ['--model_path', model_params[request.model_id]['model_path'],
                     '--source_path', model_params[request.model_id]['source_path'],
                     '--batch', '16'],
                    wav_chunk
                )
                
                
                # 推送音频数据
                await audio_track.put_frame(audio_frame)

                for image in video_images:
                    video_frame = VideoFrame.from_ndarray(image, format="rgb24")
                    await video_track.put_frame(video_frame)
                    await asyncio.sleep(1/VIDEO_FPS)

        asyncio.create_task(generate_and_push())
        return {"status": "推流已启动"}

    except Exception as e:
        logger.exception(f"推流异常: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    # 关闭所有PeerConnection
    for pc in pcs:
        await pc.close()
    pcs.clear()

app.mount("/", StaticFiles(directory="web", html=True), name="static")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)