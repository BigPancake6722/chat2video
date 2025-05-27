import asyncio
import struct
import sys
import warnings
from io import BytesIO
from typing import AsyncGenerator
import librosa
import numpy as np
import soundfile as sf
import uvicorn
from aiortc import (
    RTCIceServer,
    RTCPeerConnection,
    RTCConfiguration,
    RTCSessionDescription,
)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rtmp_ffmpeg2 import RTMPPusher
from validation import verify_h264_stream

sys.path.append("/root/chat2video/chat2audio/")
sys.path.append("/root/chat2video/sync-gaussian-talker/")
from app_stream1 import llm_to_tts_stream_dual
from render_transfer3 import main as audio2video

app = FastAPI()
# rtmp_publisher = RTMPStreamPublisher()
rtmp_publisher = RTMPPusher()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

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
    
    async for text, wav_chunk, opus_chunk in tts_stream:
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

                    yield audio_data, target_sr, text, opus_chunk

        except Exception as e:
            print(f"⚠️ 音频解析暂缓: {str(e)}")
        finally:
            buffer = BytesIO()  # 重置缓冲区
            await asyncio.sleep(0)

@app.post("/chat2video")
async def api_llm_tts_stream(request: TTSRequest):
    try:
        print("推流任务已启动，开始推理")
        # 使用新的三元组流接口
        tts_stream = llm_to_tts_stream_dual(request.input, request.speaker, request.prompt)
        
        async for audio_chunk, sr, text, opus_stream in process_dual_tts_stream(tts_stream, min_chunk_size=0):
            from model_params import model_params
            # 使用WAV音频生成视频
            video_imgs = audio2video(
                ['--model_path', model_params[request.model_id]['model_path'],
                '--source_path', model_params[request.model_id]['source_path'],
                '--batch', '16'],
                audio_chunk
            )
            
            
            # 使用AAC音频和视频进行推流
            # rtmp_publisher.push_stream(aac_stream, video_bytes)
                    
    except Exception as e:
        print(f"推流过程中出现异常: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)