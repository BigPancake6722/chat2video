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
    def __init__(self, buffer, input_rate=32000):
        super().__init__()
        self.buffer = buffer  # asyncio.Queue[bytes]
        self._input_rate = input_rate
        self._pts = 0
        self._resampler = self._init_resampler()

    def _init_resampler(self):
        # 使用PyAV创建重采样器
        return av.AudioResampler(
            format='s16',
            layout='mono',
            rate=48000  # 目标采样率
        )

    async def recv(self):
        # 获取原始音频数据（32kHz）
        data = await self._safe_get_data()
        
        # 转换为48kHz
        pcm_32k = np.frombuffer(data, dtype=np.int16)
        pcm_48k = self._resample_frame(pcm_32k)
        # 构造AudioFrame
        frame = AudioFrame.from_ndarray(
            array=pcm_48k[np.newaxis, :],
            format='s16',
            layout='mono'
        )
        frame.pts = self._pts
        frame.sample_rate = 48000
        frame.time_base = fractions.Fraction(1, 48000)
        self._pts += len(pcm_48k)
        return frame

    async def _safe_get_data(self):
        try:
            return await asyncio.wait_for(self.buffer.get(), timeout=0.01)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return self._generate_silence()

    def _resample_frame(self, pcm_32k: np.ndarray) -> np.ndarray:
        if np.all(pcm_32k == 0):
            return np.zeros(1600, dtype=np.int16)
        # 使用PyAV重采样
        input_frame = AudioFrame(
            format='s16',
            layout='mono',
            samples=len(pcm_32k)
        )
        input_frame.rate = self._input_rate
        input_frame.planes[0].update(pcm_32k.tobytes())
        output_frame = self._resampler.resample(input_frame)
        return np.frombuffer(bytes(output_frame[0].planes[0]), dtype=np.int16)

    def _generate_silence(self):
        # 生成48kHz静音（10ms = 480 samples）
        return np.zeros(480, dtype=np.int16).tobytes()

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
