import re
import wave
import sys
from io import BytesIO
from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, Response
from transformers import Qwen2Tokenizer, Qwen2ForCausalLM
import time

from tts.tts_non_stream import TTSNonStreamer
from LLM.llm_non_stream import LLMNonStreamer

app = FastAPI()
llm_streamer = LLMNonStreamer()
tts_streamer = TTSNonStreamer()
start = 0

def generate_wav_header(sample_rate=32000):
    """生成WAV文件头"""
    with BytesIO() as wav_buf:
        with wave.open(wav_buf, 'wb') as vfout:
            vfout.setnchannels(1)
            vfout.setsampwidth(2)
            vfout.setframerate(sample_rate)
            vfout.writeframes(b'')
        wav_buf.seek(0)
        return wav_buf.read()

async def handle_non_streaming_response(text, speaker, media_type):
    """处理非流式音频响应"""
    tts_params = {
        "text": text,
        "speaker_id": speaker,
        "media_type": media_type
    }

    audio = await tts_streamer.tts_get_endpoint(**tts_params)
    if audio is None:
        raise HTTPException(status_code=500, detail="TTS generation failed")

    content_type = {
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "raw": "audio/x-raw",
        "aac": "audio/aac"
    }.get(media_type, "application/octet-stream")
    print("used time", time.time()-start)
    return Response(
        content=audio,
        media_type=content_type,
        headers={
            "Content-Disposition": f"attachment; filename=audio.{media_type}",
            "X-Audio-Format": media_type
        }
    )

class TTSRequest(BaseModel):
    input: str
    speaker: str = "Firefly"

@app.post("/chat2audio")
async def chat2audio(request: TTSRequest):
    global start
    start = time.time()
    user_input = request.input
    speaker = request.speaker
    media_type = 'wav'
    if not user_input:
        raise HTTPException(status_code=400, detail="Input text is required")

    # 获取文本回复
    response = llm_streamer.chat(user_input, speaker)
    print(f"Generated text: {response[:200]}...")  # 日志截断

    return await handle_non_streaming_response(response, speaker, media_type)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)