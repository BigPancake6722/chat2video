import asyncio
from io import BytesIO
from typing import AsyncGenerator, Tuple
import subprocess
import os
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
    """实时流式返回文本、WAV音频和原始PCM音频"""
    llm_stream = llm_streamer.stream_output(user_input, speaker_id, prompt)
    tts_streamer.init_speaker_and_weight(speaker_id=speaker_id)
    buffer = ""
    punctuations = "。！？，、；：\"\''（）《》【】…"
    min_len = 14
    max_len = 30

    async for text_chunk in llm_stream:
        buffer += text_chunk

        while len(buffer) >= min_len:
            split_pos = next((i for i, c in enumerate(buffer) if c in punctuations and i >= min_len - 1), -1)
            
            if split_pos != -1:
                segment = buffer[:split_pos + 1]
                buffer = buffer[split_pos + 1:]
            elif len(buffer) >= max_len:
                segment = buffer[:max_len]
                buffer = buffer[max_len:]
            else:
                break

            if segment:
                pcm_queue1 = asyncio.Queue()
                pcm_queue2 = asyncio.Queue()
                wav_queue = asyncio.Queue()
                opus_queue = asyncio.Queue()
                ogg_queue = asyncio.Queue()

                tasks = [
                    asyncio.create_task(_generate_pcm(segment, pcm_queue1, pcm_queue2)),
                    # asyncio.create_task(_generate_ogg(segment, ogg_queue)),
                    asyncio.create_task(_convert_to_wav(pcm_queue1, wav_queue))
                    # asyncio.create_task(_convert_to_opus(pcm_queue2, opus_queue))
                ]

                while tasks:
                    done, pending = await asyncio.wait(
                        tasks,
                        return_when=asyncio.ALL_COMPLETED
                    )

                    # for task in done:
                        # if task._coro.__name__ == '_convert_to_wav':
                    wav_data = await wav_queue.get()
                    if wav_data is not None:
                        # save_opus_file(opus_queue, "/root/autodl-tmp/test.opus")
                        pcm_data = await pcm_queue2.get()
                        # pcm_data = await pcm_queue2.get()
                        # print(pcm_data)
                        # wav_duration = get_wav_duration(wav_data)
                        # print("获取持续时间")
                        # opus_duration = get_opus_duration(ogg_data, is_ogg=True)
                        # pcm_duration = get_pcm_duration(pcm_data, 32000)
                        # print("wav持续时间：", wav_duration, "opus持续时间：", pcm_duration)
                        yield segment, wav_data, pcm_data

                    # 更新任务状态
                    tasks = [t for t in tasks if not t.done()]

        await asyncio.sleep(0.1)

async def _generate_pcm(segment: str, pcm_queue1: asyncio.Queue, pcm_queue2: asyncio.Queue):
    """生成原始PCM数据流"""
    try:
        for sr, chunk in tts_streamer.generate_audio(segment, "raw"):
            await pcm_queue1.put(chunk.tobytes())
            await pcm_queue2.put(chunk.tobytes())
    finally:
        await pcm_queue1.put(None)
        await pcm_queue2.put(None)

async def _generate_ogg(segment: str, ogg_queue: asyncio.Queue):
    """生成原始OGG数据流"""
    try:
        for sr, chunk in tts_streamer.generate_audio(segment, "ogg"):
            await ogg_queue.put(tts_streamer.pack_audio(BytesIO(), chunk, sr, "ogg").getvalue())
    finally:
        await ogg_queue.put(None)

async def _convert_to_wav(pcm_queue: asyncio.Queue, output_queue: asyncio.Queue):
    """动态生成流式WAV"""
    try:
        buffer = BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(32000)  # 假设固定采样率
            
            while True:
                pcm_chunk = await pcm_queue.get()
                if pcm_chunk is None:
                    break
                
                # 写入音频数据
                wav.writeframes(pcm_chunk)
                buffer.seek(0)
                await output_queue.put(buffer.read())
                buffer.truncate(0)
                buffer.seek(0)
                
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
            '-ar', '32000',
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

def get_wav_duration(wav_bytes):
    """计算WAV音频的播放时长（秒）"""
    with wave.open(BytesIO(wav_bytes)) as wav:
        frames = wav.getnframes()
        framerate = wav.getframerate()
        print("wav编码格式", frames, framerate)
        return frames / framerate

def get_pcm_duration(pcm_bytes, sample_rate, sample_width=2, num_channels=1):
    """计算PCM音频的播放时长（秒）"""
    bytes_per_frame = sample_width * num_channels
    num_frames = len(pcm_bytes) // bytes_per_frame
    return num_frames / sample_rate

def is_opus_in_ogg(data: bytes) -> bool:
    if len(data) < 32 or data[:4] != b'OggS':
        return False
    
    # 解析第一个 OGG Page
    header_type = data[5]  # 第 6 字节表示 Page 类型
    if header_type != 0x02:  # 0x02 表示 BOS (Beginning of Stream)
        return False
    
    # 在 Page 数据段查找 'OpusHead'
    page_segment_count = data[26]
    segment_table_start = 27
    segment_table_end = segment_table_start + page_segment_count
    if segment_table_end > len(data):
        return False
    
    # 检查数据段是否包含 'OpusHead'
    data_start = segment_table_end
    return data[data_start:data_start+8] == b'OpusHead'

def get_opus_duration(data: bytes, is_ogg: bool = True) -> float:
    """
    计算Opus音频流的持续时间（单位：秒）
    
    参数：
        data: Opus音频字节流
        is_ogg: 是否为Ogg封装格式（默认True）
        
    返回：
        float: 音频时长（秒）
        
    异常：
        ValueError: 数据格式不支持时抛出
    """
    SAMPLES_PER_SECOND = 48000  # Opus标准采样率
    print("不是，我怎么访问不了data？", type(data))
    if is_ogg:
        print("哎嘿，我就是ogg")
        # ================== Ogg封装格式处理 ==================
        if not data.startswith(b'OggS'):
            print("我明明都封装了")
            raise ValueError("Invalid Ogg header")

        total_samples = 0
        pos = 0
        print("页头解析好像没进去")
        while pos + 27 < len(data):  # Ogg页头最小长度28字节
            # 解析页头
            print('进来了，好耶')
            _, flags, granule_pos, _, page_seq = struct.unpack_from('<4sBBqI', data, pos)
            print("unpack也没出问题")
            pos += 27  # 跳转到segment数量位置
            
            num_segments = data[pos]
            pos += 1 + num_segments  # 跳过segment表
            
            # 最后一个页面的granule_pos包含总采样数
            if (flags & 0x04) or (pos >= len(data)):
                total_samples = granule_pos
                break
        print("顺利退出，好耶")
        return total_samples / SAMPLES_PER_SECOND
        
    else:
        # ================== 原始Opus数据包处理 ==================
        # 需要解析TOC头计算每包采样数
        total_samples = 0
        pos = 0
        print("data就这么不可言说")
        print("未进入循环", len(data))
        while pos < len(data):
            if pos + 1 > len(data):
                break        
            # 解析TOC头
            toc = data[pos]
            config = (toc >> 3) & 0x1F
            stereo = (toc >> 2) & 0x01
            # 根据配置获取帧时长（单位：采样数）
            frame_sizes = {
                0: 480,   # 10 ms
                1: 960,   # 20 ms
                2: 1920,  # 40 ms
                3: 2880,  # 60 ms
            }
            samples = frame_sizes.get((config >> 3) & 0x03, 0)
            print("采样成功")
            # 计算包长度
            if config < 12:
                frame_count = 1
            elif config < 16:
                frame_count = 2
            else:
                frame_count = data[pos + 1] & 0x3F
                pos += 1
                
            total_samples += samples * frame_count
            pos += 1  # 移动到下一包
            
        return total_samples / SAMPLES_PER_SECOND