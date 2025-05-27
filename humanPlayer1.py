import numpy as np
import av
import threading
import time
import asyncio
import fractions  # 新增导入
from av import AudioFrame, VideoFrame
from aiortc import RTCPeerConnection, MediaStreamTrack
from aiortc.mediastreams import AudioStreamTrack, VideoStreamTrack

class CustomVideoTrack(VideoStreamTrack):
    def __init__(self, buffer):
        super().__init__()
        self.buffer = buffer  # asyncio.Queue
        self._last_pts = 0
        
    async def recv(self):
        frame = await self.buffer.get()
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        return frame

class CustomAudioTrack(AudioStreamTrack):
    def __init__(self, buffer, sample_rate):
        super().__init__()
        self.buffer = buffer  # asyncio.Queue
        self._sample_rate = sample_rate
        self._pts = 0

    async def recv(self):
        data = await self.buffer.get()
        audio_frame = AudioFrame.from_ndarray(
            np.frombuffer(data, dtype='<i2').reshape(-1, 1),
            format='s16',
            layout='mono'
        )
        audio_frame.pts = self._pts
        audio_frame.sample_rate = self._sample_rate
        audio_frame.time_base = fractions.Fraction(1, self._sample_rate)
        self._pts += audio_frame.samples
        return audio_frame

class HumanPlayer:
    def __init__(self, output_fps=25, audio_sample_rate=32000):
        self.pc = RTCPeerConnection()
        self.output_fps = output_fps
        self.audio_sample_rate = audio_sample_rate
        
        # 异步队列（线程安全通信）
        self.video_buffer = asyncio.Queue()
        self.audio_buffer = asyncio.Queue()
        
        # 保存事件循环引用
        self._video_loop_event_loop = None
        self._audio_loop_event_loop = None

        # 音视频轨道
        self.video_track = CustomVideoTrack(self.video_buffer)
        self.audio_track = CustomAudioTrack(self.audio_buffer, self.audio_sample_rate)
        self.pc.addTrack(self.video_track)
        self.pc.addTrack(self.audio_track)
        
        # 控制状态
        self.running = False
        
        # 启动处理线程
        self._start_threads()

    def _start_threads(self):
        self.running = True
        threading.Thread(target=self._video_loop, daemon=True).start()
        threading.Thread(target=self._audio_loop, daemon=True).start()

    def feed_external_data(self, rgb_frames: np.ndarray, pcm_audio: bytes):
        """线程安全的音视频数据输入"""
        # 视频处理（直接推送到异步队列）
        for frame in rgb_frames:
            asyncio.run_coroutine_threadsafe(
                self.video_buffer.put(VideoFrame.from_ndarray(frame, format="rgb24")),
                loop=self._video_loop_event_loop
            )
        
        # 音频处理（分块推送到异步队列）
        chunk_size = int(self.audio_sample_rate * 0.02)  # 20ms的样本数
        pcm_samples = np.frombuffer(pcm_audio, dtype='<i2')
        for i in range(0, len(pcm_samples), chunk_size):
            chunk = pcm_samples[i:i+chunk_size].tobytes()
            if chunk:
                asyncio.run_coroutine_threadsafe(
                    self.audio_buffer.put(chunk),
                    loop=self._audio_loop_event_loop
                )

    def _video_loop(self):
        """视频处理线程（仅维护事件循环）"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._video_loop_event_loop = loop  # 保存事件循环引用
        loop.run_forever()  # 保持事件循环运行

    def _audio_loop(self):
        """音频处理线程（仅维护事件循环）"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._audio_loop_event_loop = loop  # 保存事件循环引用
        loop.run_forever()  # 保持事件循环运行

    def close(self):
        self.running = False
        # 停止事件循环
        self._video_loop_event_loop.call_soon_threadsafe(self._video_loop_event_loop.stop)
        self._audio_loop_event_loop.call_soon_threadsafe(self._audio_loop_event_loop.stop)
        self.pc.close()

    import binascii

    def save_bytes_as_hex_txt(self, bytes_data: bytes, output_txt_path: str) -> None:
        """
        将 bytes 数据直接转为十六进制字符串并保存为 TXT 文件
        
        参数:
            bytes_data: 输入的 bytes 对象（如 b"\x01\x00\xFF\x7F"）
            output_txt_path: 输出的 TXT 文件路径（如 "hex_output.txt"）
        """
        # 将 bytes 转为十六进制字符串（每字节用空格分隔）
        print("到底能不能报错啊")
        hex_str = binascii.hexlify(bytes_data, ''.decoder()).decode('utf-8')
        print("能不能保存啊")
        # 写入 TXT 文件
        with open(output_txt_path, 'w') as txt_file:
            txt_file.write(hex_str)
        
        print(f"十六进制内容已保存到: {output_txt_path}")