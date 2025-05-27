import asyncio
from io import BytesIO
from typing import AsyncGenerator, Tuple
import subprocess
import wave  # 添加wave模块导入

from LLM.prompts import speaker_prompts
from LLM.llm_stream2 import LLMStreamer
from tts.tts_stream import TTSStreamer

llm_streamer = LLMStreamer()
tts_streamer = TTSStreamer()

async def llm_to_tts_stream_dual(
    user_input: str,
    speaker_id: str = "Firefly",
    prompt: str = None
) -> AsyncGenerator[Tuple[str, bytes, bytes], None]:
    """实时流式返回文本、WAV音频和Opus音频（并行生成）"""
    print("speaker_id", speaker_id)
    llm_stream = llm_streamer.stream_output(user_input, speaker_id, prompt)
    tts_streamer.init_speaker_and_weight(speaker_id=speaker_id)
    buffer = ""
    punctuations = "。！？，、；：\"\''（）《》【】…"
    min_len = 3
    max_len = 20
    print("开始生成")
    async for text_chunk in llm_stream:
        buffer += text_chunk

        while len(buffer) >= min_len:
            # 切分逻辑保持不变
            split_pos = next((i for i, c in enumerate(buffer) if c in punctuations and i >= min_len - 1), -1)
            
            if 0 <= split_pos <= max_len - 1:
                segment = buffer[:split_pos + 1]
                buffer = buffer[split_pos + 1:]
            elif len(buffer) >= max_len:
                segment = buffer[:max_len]
                buffer = buffer[max_len:]
            else:
                break

            if segment:
                # 创建独立的PCM队列和输出队列
                wav_pcm_queue = asyncio.Queue()
                opus_pcm_queue = asyncio.Queue()
                wav_queue = asyncio.Queue()
                opus_queue = asyncio.Queue()

                # 启动并行任务
                tasks = [
                    asyncio.create_task(_generate_pcm(segment, wav_pcm_queue, opus_pcm_queue)),
                    asyncio.create_task(_convert_to_wav(wav_pcm_queue, wav_queue)),
                    asyncio.create_task(_convert_to_opus(opus_pcm_queue, opus_queue))
                ]

                # 并行收集结果
                while tasks:
                    # 同时等待两个输出队列
                    wav_task = asyncio.create_task(wav_queue.get())
                    opus_task = asyncio.create_task(opus_queue.get())
                    
                    done, pending = await asyncio.wait(
                        [wav_task, opus_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    for task in done:
                        data = task.result()
                        queue_type = 'wav' if task is wav_task else 'opus'
                        
                        if data is None:  # 结束标记
                            if queue_type == 'wav':
                                wav_task.cancel()
                            else:
                                opus_task.cancel()
                            continue
                            
                        other_data = b""
                        if queue_type == 'wav':
                            other_data = opus_queue.get_nowait() if not opus_queue.empty() else b""
                        else:
                            other_data = wav_queue.get_nowait() if not wav_queue.empty() else b""
                            
                        yield segment, data if queue_type == 'wav' else other_data, data if queue_type == 'opus' else other_data

                    # 检查任务完成状态
                    tasks = [t for t in tasks if not t.done()]

        await asyncio.sleep(0)

async def _generate_pcm(segment: str, wav_pcm_queue: asyncio.Queue, opus_pcm_queue: asyncio.Queue):
    """生成原始PCM数据流（并行分发）"""
    try:
        for sr, chunk in tts_streamer.generate_audio(segment, "raw"):
            # 同时写入两个队列
            await wav_pcm_queue.put((sr, chunk))
            await opus_pcm_queue.put((sr, chunk))
    finally:
        # 发送结束标记
        await wav_pcm_queue.put(None)
        await opus_pcm_queue.put(None)

async def _convert_to_wav(pcm_queue: asyncio.Queue, output_queue: asyncio.Queue):
    """PCM转WAV（并行版本）"""
    try:
        wav_buffer = BytesIO()
        header_written = False
        params_set = False
        sr = None
        
        while True:
            item = await pcm_queue.get()
            if item is None:
                break
            
            current_sr, chunk = item
            if not params_set:
                sr = current_sr
                params_set = True
            
            if not header_written:
                # 动态生成WAV头
                with wave.open(wav_buffer, 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(sr)
                    wav.writeframes(chunk.tobytes())
                header_written = True
            else:
                # 追加数据帧
                wav_buffer.write(chunk.tobytes())
            
            # 实时输出完整WAV数据
            await output_queue.put(wav_buffer.getvalue())
            wav_buffer.seek(0)
            wav_buffer.truncate()
            
        await output_queue.put(None)
    except Exception as e:
        await output_queue.put(None)
        raise

async def _convert_to_opus(pcm_queue: asyncio.Queue, output_queue: asyncio.Queue):
    """PCM转Opus（并行优化版）"""
    try:
        ffmpeg = await asyncio.create_subprocess_exec(
            'ffmpeg',
            '-f', 's16le',
            '-ar', '24000',
            '-ac', '1',
            '-i', 'pipe:0',
            '-c:a', 'libopus',
            '-b:a', '64k',
            '-vbr', 'on',
            '-frame_duration', '20',
            '-f', 'opus',
            'pipe:1',
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL  # 忽略错误输出
        )
        
        async def feed_input():
            try:
                while True:
                    item = await pcm_queue.get()
                    if item is None:
                        break
                    sr, chunk = item
                    ffmpeg.stdin.write(chunk.tobytes())
                    await ffmpeg.stdin.drain()
            finally:
                ffmpeg.stdin.close()
        
        async def read_output():
            try:
                while True:
                    chunk = await ffmpeg.stdout.read(4096)
                    if not chunk:
                        break
                    await output_queue.put(chunk)
            finally:
                await output_queue.put(None)
        
        await asyncio.gather(feed_input(), read_output())
    except Exception as e:
        await output_queue.put(None)
        raise