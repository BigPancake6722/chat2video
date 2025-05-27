import sys
import time
import asyncio
import uvicorn
import numpy as np
from io import BytesIO
from pydantic import BaseModel
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, RTCRtpSender
from aiortc.mediastreams import AudioStreamTrack, VideoStreamTrack
import logging

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

class TTSRequest(BaseModel):
    input: str
    speaker: str = "Firefly"
    model_id: str = "Chinese1"
    prompt: str = None
    session_id: str = "default_stream"

class OfferRequest(BaseModel):
    sdp: str
    type: str

pcs = set()

# 自定义媒体轨道
class CustomAudioTrack(AudioStreamTrack):
    def __init__(self):
        super().__init__()
        self._queue = asyncio.Queue()

    async def put_audio(self, audio_data):
        await self._queue.put(audio_data)

    async def recv(self):
        return await self._queue.get()

class CustomVideoTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self._queue = asyncio.Queue()

    async def put_video(self, video_data):
        await self._queue.put(video_data)

    async def recv(self):
        return await self._queue.get()

@app.post("/offer")
async def handle_offer(request: OfferRequest):
    pc = RTCPeerConnection()
    pcs.add(pc)

    # 创建自定义轨道
    audio_track = CustomAudioTrack()
    video_track = CustomVideoTrack()

    # 添加轨道到连接
    pc.addTrack(audio_track)
    pc.addTrack(video_track)

    # 配置编码参数
    def configure_sender(sender: RTCRtpSender):
        parameters = sender.getParameters()
        if parameters:
            parameters.encodings[0].maxBitrate = MAX_BITRATE
            sender.setParameters(parameters)

    @pc.on("track")
    def on_track(track):
        logger.info(f"Track {track.kind} received")

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state is {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    # 处理offer
    await pc.setRemoteDescription(RTCSessionDescription(sdp=request.sdp, type=request.type))

    # 创建answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "session_id": request.session_id
    }

@app.post("/chat2video")
async def api_llm_tts_stream(request: TTSRequest):
    try:
        # 初始化WebRTC连接
        pc = RTCPeerConnection()
        pcs.add(pc)

        # 创建自定义轨道
        audio_track = CustomAudioTrack()
        video_track = CustomVideoTrack()

        # 添加轨道到连接
        pc.addTrack(audio_track)
        pc.addTrack(video_track)

        # 配置编码参数
        def configure_sender(sender: RTCRtpSender):
            parameters = sender.getParameters()
            if parameters:
                parameters.encodings[0].maxBitrate = MAX_BITRATE
                sender.setParameters(parameters)

        # 设置ICE候选处理
        @pc.on("icecandidate")
        def on_icecandidate(candidate):
            logger.info(f"ICE candidate: {candidate}")

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state is {pc.connectionState}")
            if pc.connectionState == "failed":
                await pc.close()
                pcs.discard(pc)

        # 获取TTS流
        tts_stream = llm_to_tts_stream_dual(request.input, request.speaker, request.prompt)

        async def generate_and_push():
            async for text, audio_chunk in tts_stream:
                # 生成视频帧
                video_bytes = audio2video(
                    ['--model_path', model_params[request.model_id]['model_path'],
                     '--source_path', model_params[request.model_id]['source_path'],
                     '--batch', '16'],
                    audio_chunk
                )

                # 推送音频数据
                await audio_track.put_audio(audio_chunk)

                # 推送视频数据
                await video_track.put_video(video_bytes)

                # 控制推流速率
                await asyncio.sleep(1 / VIDEO_FPS)

        # 启动异步任务来生成和推送流
        asyncio.create_task(generate_and_push())

        # 创建offer
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "session_id": request.session_id
        }

    except Exception as e:
        logger.exception(f"推流异常: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    # 关闭所有PeerConnection
    for pc in pcs:
        await pc.close()
    pcs.clear()

@app.get("/web.html")
def redirect_to_web():
    print("redirection")
    return RedirectResponse(url="/web/web.html")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)