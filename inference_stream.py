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
from pydantic import BaseModel
from typing import AsyncGenerator, Tuple
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

sys.path.append("/root/chat2video/chat2audio/")
sys.path.append("/root/chat2video/sync-gaussian-talker/")
from app_stream import llm_to_tts_stream_all_head as llm_to_tts_stream
from render_transfer import main as audio2video



app = FastAPI()

async def process_tts_stream(
    tts_stream: AsyncGenerator[bytes, None],
    media_type: str = "wav",
    target_sr: int = 16000,
    min_chunk_size: int = 1024  # 新增最小块大小参数
) -> AsyncGenerator[Tuple[np.ndarray, int], None]:
    buffer = BytesIO()
    header_received = False
    pending_data = b""
    
    async for chunk in tts_stream:
        if not chunk:
            continue
            
        # 累积数据直到达到最小处理大小
        pending_data += chunk
        
        # 只有当数据足够时才处理
        if len(pending_data) < min_chunk_size and not header_received:
            continue
            
        buffer.write(pending_data)
        pending_data = b""
        
        # WAV头特殊处理
        if media_type == "wav" and not header_received:
            if buffer.getbuffer().nbytes < 44:
                continue
            header_received = True
        
        # 尝试解析音频
        buffer.seek(0)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio_data, sr = sf.read(buffer)
                
                if len(audio_data) > 0:
                    # 单声道处理
                    if audio_data.ndim > 1:
                        audio_data = np.mean(audio_data, axis=1)
                    
                    # 重采样
                    if sr != target_sr:
                        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
                    
                    print(f"✅ 产生音频块: {audio_data.shape} samples @ {target_sr}Hz")
                    yield audio_data, target_sr
                    
                    # 重置缓冲区但保留未消费数据
                    remaining = buffer.getvalue()[buffer.tell():]
                    buffer = BytesIO()
                    buffer.write(remaining)
                    
        except Exception as e:
            print(f"⚠️ 音频解析暂缓: {str(e)}")
        finally:
            buffer.seek(0, 2)
        
        await asyncio.sleep(0)


def save_bytes_stream_as_wav(byte_stream, output_path, sample_rate=32000):
    num_channels = 1
    sample_width = 2
    audio_array = np.frombuffer(byte_stream, dtype=np.int16)

    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_array.tobytes())


class TTSRequest(BaseModel):
    input: str
    speaker: str = "Firefly"
    model_id : str = "Chinese1"
    prompt : str = None

@app.post("/chat2video")
async def api_llm_tts_stream(request: TTSRequest):
    tts_stream = llm_to_tts_stream(request.input, request.speaker, request.prompt, media_type="wav")
    index = 0
    video_stream = None
    async for raw_chunk in tts_stream:
        print(f"🔥 收到原始块: {len(raw_chunk)} bytes")
        save_bytes_stream_as_wav(raw_chunk, f"/root/autodl-tmp/GaussianTalker/{index}.wav")
        # 处理为音频帧
        async for audio_chunk, sr in process_tts_stream(
            mock_stream([raw_chunk]),  # 将单块包装为生成器
            min_chunk_size=0  # 禁用最小块检查
        ):
            print(f"🎵 处理后的音频: {audio_chunk.shape}")
            from model_params import model_params
            video_stream = audio2video(['--model_path', model_params[request.model_id]['model_path'],
                                        '--source_path', model_params[request.model_id]['source_path'],
                                        '--batch', '16'], audio_chunk)
    return StreamingResponse(video_stream)

async def mock_stream(chunks):
    """将单个块包装为异步生成器"""
    for chunk in chunks:
        yield chunk
  

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)