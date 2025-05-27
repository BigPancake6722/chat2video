import sys
import time
import asyncio
import uvicorn
import numpy as np
from io import BytesIO
from pydantic import BaseModel
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# 导入自定义模块
sys.path.append("/root/chat2video/chat2audio/")
sys.path.append("/root/chat2video/sync-gaussian-talker/")
from app_stream2 import llm_to_tts_stream_dual
from render_transfer4 import main as audio2video
from rtc_utils1 import WebRTCStreamer, AudioProcessor, VideoProcessor

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置参数
SIGNALING_SERVER = "http://47.111.96.21:1985/rtc/v1/whip/?app=live&stream=livestream"
AUDIO_SAMPLE_RATE = 32000
VIDEO_FPS = 25

class TTSRequest(BaseModel):
    input: str
    speaker: str = "Firefly"
    model_id: str = "Chinese1"
    prompt: str = None
    session_id: str = "default_stream"

@app.post("/chat2video")
async def api_llm_tts_stream(request: TTSRequest):
    """处理音视频流生成和WebRTC推流"""
    # 初始化处理器
    audio_processor = AudioProcessor(sample_rate=AUDIO_SAMPLE_RATE)
    video_processor = VideoProcessor()
    rtc_streamer = WebRTCStreamer(SIGNALING_SERVER)
    
    try:
        # 初始化WebRTC连接
        await rtc_streamer.initialize()
        await rtc_streamer.start_stream()
        
        # 获取TTS流
        tts_stream = llm_to_tts_stream_dual(request.input, request.speaker, request.prompt)
        
        # 处理并推流
        async for text, audio_chunk in tts_stream:
            # 生成视频帧
            video_bytes = audio2video(
                ['--model_path', model_params[request.model_id]['model_path'],
                '--source_path', model_params[request.model_id]['source_path'],
                '--batch', '16'],
                audio_chunk
            )
            
            # 处理音频格式
            opus_audio = audio_processor.repackage_opus(audio_chunk)
            if not opus_audio:
                raise ValueError("音频格式转换失败")
            
            # 处理视频格式
            vp8_video = video_processor.repackage_vp8(video_bytes)
            if not vp8_video:
                raise ValueError("视频格式转换失败")
            
            # 推送媒体数据
            await rtc_streamer.push_media(opus_audio, vp8_video)
            
            # 控制推流速率 (根据视频FPS)
            await asyncio.sleep(1/VIDEO_FPS)
            
        return {"status": "completed"}
        
    except Exception as e:
        print(f"推流异常: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await rtc_streamer.close()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)