import cv2
import sys
import asyncio
import uvicorn
import traceback
import numpy as np
from io import BytesIO
from pydantic import BaseModel
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from aiortc.contrib.media import MediaRelay
from aiortc.rtcrtpsender import RTCRtpSender
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from aiortc import VideoStreamTrack, AudioStreamTrack, MediaStreamTrack
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration

sys.path.append("/root/chat2video/chat2audio/")
sys.path.append("/root/chat2video/sync-gaussian-talker/")
from app_stream2 import llm_to_tts_stream_dual
from render_transfer1 import main as audio2video

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 自定义视频轨道
class LiveVideoStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self._queue = asyncio.Queue(maxsize=30)  # 缓冲队列

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = await self._queue.get()
        frame.pts = pts
        frame.time_base = time_base
        return frame

    def add_frame(self, img):
        # 将RGB24转换为YUV420P
        yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV_I420)
        # 创建视频帧并放入队列
        self._queue.put_nowait(
            VideoFrame.from_ndarray(yuv, format="yuv420p")
        )

# 自定义音频轨道 
class LiveAudioStreamTrack(AudioStreamTrack):
    def __init__(self):
        super().__init__()
        self._buffer = bytearray()

    async def recv(self):
        if len(self._buffer) < 960 * 2:  # 20ms的48kHz音频
            return None
        data = bytes(self._buffer[:960 * 2])
        self._buffer = self._buffer[960 * 2:]
        return AudioFrame(samples=data, layout="mono", sample_rate=48000)

# 添加WebRTC信令接口
class OfferRequest(BaseModel):
    sdp: str
    type: str

# 初始化 PeerConnection 时强制添加 transceiver
pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))

# 显式添加音视频传输通道
video_transceiver = pc.addTransceiver("video", direction="recvonly")
audio_transceiver = pc.addTransceiver("audio", direction="recvonly")

@app.post("/webrtc/offer")
async def handle_offer(request: OfferRequest):
    try:
        # 校验 Offer 类型
        if request.type.lower() != "offer":
            raise HTTPException(status_code=400, detail="Invalid Offer Type")

        # 设置远端描述
        offer = RTCSessionDescription(sdp=request.sdp, type="offer")
        await pc.setRemoteDescription(offer)

        # 强制刷新 transceiver 状态
        for transceiver in pc.getTransceivers():
            if transceiver.direction == "inactive":
                transceiver._setDirection("recvonly")  # 关键修复点

        # 创建 Answer 前检查媒体状态
        if not any(t for t in pc.getTransceivers() if t.direction != "stopped"):
            raise RuntimeError("No active media channels")

        # 生成 Answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # 验证生成的 SDP 包含媒体行
        if "m=audio" not in answer.sdp or "m=video" not in answer.sdp:
            raise ValueError("Invalid SDP generated")

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type.lower()  # 强制输出小写
        }

    except Exception as e:
        print(f"SDP生成失败: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# 修改推流接口
class TTSRequest(BaseModel):
    input: str
    speaker: str = "Firefly"
    model_id: str = "Chinese1"
    prompt: str = None

async def process_dual_tts_stream(
    tts_stream: AsyncGenerator[tuple[str, bytes, bytes], None],
    target_sr: int = 32000,
    min_chunk_size: int = 1024
) -> AsyncGenerator[tuple[np.ndarray, int, str, bytes], None]:
    """处理双音频TTS流，返回音频数据、采样率、对应文本以及AAC音频流"""
    buffer = BytesIO()
    async for text, wav_chunk, aac_chunk in tts_stream:
        if not wav_chunk:
            continue
        # 处理WAV音频用于视频生成
        buffer.write(wav_chunk)
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

                    yield audio_data, target_sr, text, aac_chunk  # 返回AAC流用于推流

        except Exception as e:
            print(f"⚠️ 音频解析暂缓: {str(e)}")
        finally:
            buffer = BytesIO()  # 重置缓冲区
            await asyncio.sleep(0)


@app.post("/chat2video")
async def api_llm_tts_stream(request: TTSRequest):
    try:
        print("推流任务已启动，开始推理")
        tts_stream = llm_to_tts_stream_dual(request.input, request.speaker, request.prompt)
        print("tts推理完成")
        async for audio_chunk, sr, text, opus_stream in process_dual_tts_stream(tts_stream):
            print("text", text)
            from model_params import model_params
            # 生成视频帧
            video_imgs = audio2video(
                ['--model_path', model_params[request.model_id]['model_path'],
                '--source_path', model_params[request.model_id]['source_path'],
                '--batch', '16'],
                audio_chunk
            )  # 保持原有参数
            
            # 视频推流处理
            for img in video_imgs:
                video_track.add_frame(img)
            
            # 音频推流处理
            audio_track._buffer.extend(opus_stream)
            
            # 维持音视频同步
            await asyncio.sleep(len(opus_stream)/(2 * 48000))  # 根据采样率计算时长
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)