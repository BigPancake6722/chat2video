import sys
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from io import BytesIO
from typing import AsyncGenerator, Tuple
import time

from LLM.prompts import speaker_prompts
from LLM.llm_stream1 import LLMStreamer
from tts.tts_stream import TTSStreamer

app = FastAPI()
tts_streamer = TTSStreamer()
llm = LLMStreamer()


async def llm_to_tts_stream(
    user_input : str, 
    speaker_id : str = "Firefly", 
    prompt : str = None, 
    media_type: str = "wav"
    ) -> AsyncGenerator[bytes, None]:
    """从LLM到TTS的完整流式管道"""
    # 1. 获取LLM文本流
    llm_stream = llm.stream_output(user_input, speaker_id, prompt)
    tts_streamer.init_speaker_and_weight(speaker_id)

    # 2. 生成WAV头
    if media_type == "wav":
        yield tts_streamer.wave_header_chunk()
        media_type = "raw"
    buffer = ""
    punctuations = "。，！？；：“”‘’《》〈〉「」『』【】（）（）、…"  # 标点符号集合
    try:
        async for text_chunk in llm_stream:
            # 删除括号内的文字
            in_bracket = False
            new_text_chunk = ""
            for char in text_chunk:
                if char in "（(":
                    in_bracket = True
                elif char in "）)":
                    in_bracket = False
                elif not in_bracket:
                    new_text_chunk += char
            buffer += new_text_chunk

            while True:
                # 尝试按标点断句
                punctuation_index = next((i for i, char in enumerate(buffer) if char in punctuations), None)
                if punctuation_index is not None and punctuation_index + 1 <= 20:
                    segment = buffer[:punctuation_index + 1]
                    buffer = buffer[punctuation_index + 1:]
                # 如果没有合适的标点，按20字分块
                elif len(buffer) >= 20:
                    segment = buffer[:20]
                    buffer = buffer[20:]
                else:
                    break

                if segment:
                    tts_gen = tts_streamer.generate_audio(segment, media_type)
                    for sr, audio_chunk in tts_gen:
                        yield tts_streamer.pack_audio(BytesIO(), audio_chunk, sr, media_type).getvalue()
            await asyncio.sleep(0)  # 释放控制权，允许其他协程运行

    except asyncio.CancelledError:
        # 处理任务取消的情况
        pass

async def llm_to_tts_stream_all_head(
    user_input : str, 
    speaker_id : str = "Firefly",
    prompt : str = None, 
    media_type: str = "wav"
    ) -> AsyncGenerator[bytes, None]:
    """从LLM到TTS的完整流式管道"""
    # 1. 获取LLM文本流
    llm_stream = llm.stream_output(user_input, speaker_id, prompt)
    tts_streamer.init_speaker_and_weight(speaker_id)

    # 2. 生成WAV头
    buffer = ""
    punctuations = "。，！？；：“”‘’《》〈〉「」『』【】（）（）、…"  # 标点符号集合
    try:
        async for text_chunk in llm_stream:
            # 删除括号内的文字
            in_bracket = False
            new_text_chunk = ""
            for char in text_chunk:
                if char in "（(":
                    in_bracket = True
                elif char in "）)":
                    in_bracket = False
                elif not in_bracket:
                    new_text_chunk += char
            buffer += new_text_chunk

            while True:
                # 尝试按标点断句
                punctuation_index = next(
                    (i for i, char in enumerate(buffer) if char in punctuations), 
                    None
                )
                if punctuation_index is not None and punctuation_index + 1 <= 20:
                    segment = buffer[:punctuation_index + 1]
                    buffer = buffer[punctuation_index + 1:]
                # 如果没有合适的标点，按20字分块
                elif len(buffer) >= 20:
                    segment = buffer[:20]
                    buffer = buffer[20:]
                else:
                    break

                if segment:
                    tts_gen = tts_streamer(segment, "wav")
                    for sr, audio_chunk in tts_gen:
                        yield tts_streamer.wave_header_chunk()
                        yield tts_streamer.pack_audio(BytesIO(), audio_chunk, sr, media_type).getvalue()
            await asyncio.sleep(0)  # 释放控制权，允许其他协程运行

    except asyncio.CancelledError:
        # 处理任务取消的情况
        pass


async def llm_to_tts_stream_with_text(
    user_input: str,
    speaker_id: str = "Firefly",
    prompt: str = None,
    media_type: str = "wav"
) -> AsyncGenerator[Tuple[str, bytes], None]:  # 返回 (文本, 音频) 的元组
    """同时流式返回LLM文本和TTS音频"""
    # 1. 初始化LLM和TTS
    llm_stream = llm.stream_output(user_input, speaker_id, prompt)
    tts_streamer.init_speaker_and_weight(speaker_id)
    
    buffer = ""
    punctuations = "。，！？；：“”‘’《》〈〉「」『』【】（）（）、…"

    async for text_chunk in llm_stream:
        # 清理文本（如删除括号内容）
        cleaned_chunk = "".join(
            char for char in text_chunk 
            if not (char in "（(" or char in "）)")
        )
        buffer += cleaned_chunk

        # 分割成句子或固定长度块
        while True:
            # 按标点分割（优先）
            punctuation_index = next(
                (i for i, char in enumerate(buffer) if char in punctuations),
                None
            )
            if punctuation_index is not None and punctuation_index + 1 <= 20:
                segment = buffer[:punctuation_index + 1]
                buffer = buffer[punctuation_index + 1:]
            # 按固定长度分割（后备）
            elif len(buffer) >= 20:
                segment = buffer[:20]
                buffer = buffer[20:]
            else:
                break

            if segment:
                # 并行生成音频（假设 generate_audio 是同步函数，用线程池运行）
                loop = asyncio.get_event_loop()
                tts_gen = await loop.run_in_executor(
                    None,  # 使用默认线程池
                    lambda: list(tts_streamer.generate_audio(segment, media_type))) # 转换为列表以获取所有块
                
                # 返回文本和音频块
                for sr, audio_chunk in tts_gen:
                    audio_data = tts_streamer.pack_audio(BytesIO(), audio_chunk, sr, media_type).getvalue()
                    yield segment, audio_data  # 同时返回文本和音频

        await asyncio.sleep(0)  # 释放控制权

import asyncio
from io import BytesIO


async def llm_to_tts_stream_with_text1(
    user_input: str,
    speaker_id: str = "Firefly",
    prompt: str = None,
    media_type: str = "wav"
) -> AsyncGenerator[tuple[str, bytes], None]:
    """同时流式返回LLM文本和TTS音频"""
    # 1. 初始化LLM和TTS
    llm_stream = llm.stream_output(user_input, speaker_id, prompt)
    tts_streamer.init_speaker_and_weight(speaker_id)

    buffer = ""
    punctuations = "。，！？；：“”‘’《》〈〉「」『』【】（）（）、…"

    def clean_text(text):
        result = ""
        in_brackets = False
        for char in text:
            if char in "（(":
                in_brackets = True
            elif char in "）)":
                in_brackets = False
            elif not in_brackets:
                result += char
        return result

    def split_text(buffer):
        while True:
            # 按标点分割（优先）
            punctuation_index = next((i for i, char in enumerate(buffer) if char in punctuations), None)
            if punctuation_index is not None and punctuation_index + 1 <= 20:
                segment = buffer[:punctuation_index + 1]
                buffer = buffer[punctuation_index + 1:]
            # 按固定长度分割（后备）
            elif len(buffer) >= 20:
                segment = buffer[:20]
                buffer = buffer[20:]
            else:
                break

            if segment:
                yield segment

    async for text_chunk in llm_stream:
        buffer += clean_text(text_chunk)

        for segment in split_text(buffer):
            # 并行生成音频（假设 generate_audio 是同步函数，用线程池运行）
            loop = asyncio.get_event_loop()
            tts_gen = await loop.run_in_executor(
                None,  # 使用默认线程池
                lambda: list(tts_streamer.generate_audio(segment, media_type)))

            # 返回文本和音频块
            for sr, audio_chunk in tts_gen:
                audio_data = tts_streamer.pack_audio(BytesIO(), audio_chunk, sr, media_type).getvalue()
                yield segment, audio_data

        await asyncio.sleep(0)  # 释放控制权

async def llm_to_tts_stream_dual1(
    user_input: str,
    speaker_id: str = "Firefly",
    prompt: str = None,
    media_type: str = "wav"
) -> AsyncGenerator[tuple[str, bytes], None]:
    """同时流式返回LLM文本和TTS音频"""
    # 1. 初始化LLM和TTS
    llm_stream = llm.stream_output(user_input, speaker_id, prompt)
    tts_streamer.init_speaker_and_weight(speaker_id)

    buffer = ""
    punctuations = "。，！？；：“”‘’《》〈〉「」『』【】（）（）、…"

    def clean_text(text):
        result = ""
        in_brackets = False
        for char in text:
            if char in "（(":
                in_brackets = True
            elif char in "）)":
                in_brackets = False
            elif not in_brackets:
                result += char
        return result

    def split_text(buffer):
        while True:
            # 按标点分割（优先）
            punctuation_index = next((i for i, char in enumerate(buffer) if char in punctuations), None)
            if punctuation_index is not None and punctuation_index + 1 <= 20:
                segment = buffer[:punctuation_index + 1]
                buffer = buffer[punctuation_index + 1:]
            # 按固定长度分割（后备）
            elif len(buffer) >= 20:
                segment = buffer[:20]
                buffer = buffer[20:]
            else:
                break

            if segment:
                yield segment

    async for text_chunk in llm_stream:
        buffer += clean_text(text_chunk)

        for segment in split_text(buffer):
            # 并行生成音频（假设 generate_audio 是同步函数，用线程池运行）
            loop = asyncio.get_event_loop()
            tts_gen = await loop.run_in_executor(
                None,  # 使用默认线程池
                lambda: list(tts_streamer.generate_audio(segment, media_type)))

            # 返回文本和音频块
            for sr, audio_chunk in tts_gen:
                audio_data = tts_streamer.pack_audio(BytesIO(), audio_chunk, sr, media_type).getvalue()
                yield segment, audio_data

        await asyncio.sleep(0)  # 释放控制权

async def llm_to_tts_stream_dual(
    user_input: str,
    speaker_id: str = "Firefly",
    prompt: str = None,
    media_type: str = "wav"  # 保留参数但不使用
) -> AsyncGenerator[tuple[str, bytes, bytes], None]:
    # 1. 初始化LLM和TTS
    llm_stream = llm.stream_output(user_input, speaker_id, prompt)
    tts_streamer.init_speaker_and_weight(speaker_id)

    buffer = ""
    punctuations = "。，！？；："  # 标点符号集合

    async for text_chunk in llm_stream:
        buffer += text_chunk  # 直接使用LLM返回的文本

        while True:  # 持续处理buffer中的内容
            if len(buffer) > 25:
                split_pos = 25
                for i in range(min(25, len(buffer)-1), 0, -1):
                    if buffer[i] in punctuations + " ":
                        split_pos = i + 1
                        break
                
                segment = buffer[:split_pos]
                buffer = buffer[split_pos:]
            
            else:
                punctuation_index = next(
                    (i for i, char in enumerate(buffer) if char in punctuations),
                    None
                )
                
                if punctuation_index is not None and (punctuation_index + 1) >= 3:
                    segment = buffer[:punctuation_index + 1]
                    buffer = buffer[punctuation_index + 1:]
                else:
                    break

            if segment:
                # 生成双音频流
                loop = asyncio.get_event_loop()
                wav_future = loop.run_in_executor(
                    None,
                    lambda: list(tts_streamer.generate_audio(segment, "wav")))
                
                aac_future = loop.run_in_executor(
                    None,
                    lambda: list(tts_streamer.generate_audio(segment, "aac")))
                
                wav_gen, aac_gen = await asyncio.gather(wav_future, aac_future)
                
                for (wav_sr, wav_chunk), (aac_sr, aac_chunk) in zip(wav_gen, aac_gen):
                    wav_data = tts_streamer.pack_audio(BytesIO(), wav_chunk, wav_sr, "wav").getvalue()
                    aac_data = tts_streamer.pack_audio(BytesIO(), aac_chunk, aac_sr, "aac").getvalue()
                    
                    yield segment, wav_data, aac_data

        await asyncio.sleep(0)  # 释放控制权

class TTSRequest(BaseModel):
    input: str
    speaker: str = "Firefly"
    prompt : str = None

@app.post("/chat2audio")
async def api_llm_tts_stream(request: TTSRequest):
    return StreamingResponse(
        llm_to_tts_stream(
            user_input = request.input,
            speaker_id = request.speaker,
            media_type = "wav",
            prompt = request.prompt
        ),
        media_type="audio/x-wav"
    )

@app.post("/chat2audio_text")
async def api_llm_tts_stream_multistream(request: TTSRequest):
    """返回多部分流式响应，包含文本和音频"""
    async def generate_multipart_stream() -> AsyncGenerator[bytes, None]:
        # 1. 获取LLM+TTS的流式生成器
        stream = llm_to_tts_stream_with_text(
            user_input=request.input,
            speaker_id=request.speaker,
            media_type="wav",
            prompt=request.prompt
        )

        # 2. 遍历流，生成多部分数据
        try:
            async for text_chunk, audio_chunk in stream:
                # 2.1 发送文本部分（JSON格式）
                yield f"data: {json.dumps({'text': text_chunk})}\n\n".encode('utf-8')
                
                # 2.2 发送音频部分（二进制WAV数据）
                yield audio_chunk

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Stream generation failed: {str(e)}")

    # 3. 返回多部分流式响应
    return StreamingResponse(
        generate_multipart_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",  # 或 "text/event-stream"
        headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)