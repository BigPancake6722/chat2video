import numpy as np
import av
import threading
import time
import asyncio
import fractions
from av import AudioFrame, VideoFrame
from aiortc import RTCPeerConnection, MediaStreamTrack
from aiortc.mediastreams import AudioStreamTrack, VideoStreamTrack

class CustomVideoTrack(VideoStreamTrack):
    def __init__(self, buffer):
        super().__init__()
        self.buffer = buffer
        self._start_time = time.monotonic()
        self._frame_interval = 1.0 / 25  # 25fps
        self._last_pts = 0

    async def recv(self):
        frame = await self.buffer.get()
        # 基于真实时间的PTS计算
        elapsed = time.monotonic() - self._start_time
        frame.pts = int(elapsed / self._frame_interval)
        frame.time_base = fractions.Fraction(1, 1000)  # 毫秒级时间基准
        return frame

class CustomAudioTrack(AudioStreamTrack):
    def __init__(self, buffer, sample_rate):
        super().__init__()
        self.buffer = buffer
        self._sample_rate = sample_rate
        self._start_time = time.monotonic()
        self._samples_sent = 0

    async def recv(self):
        data = await self.buffer.get()
        # 计算基于真实时间的PTS
        elapsed = time.monotonic() - self._start_time
        expected_samples = int(elapsed * self._sample_rate)
        
        # 维持采样率同步
        if expected_samples > self._samples_sent:
            await asyncio.sleep((expected_samples - self._samples_sent) / self._sample_rate)
        
        audio_frame = AudioFrame.from_ndarray(
            np.frombuffer(data, dtype=np.int16).reshape(-1, 1),
            format='s16',
            layout='mono'
        )
        audio_frame.pts = self._samples_sent
        audio_frame.sample_rate = self._sample_rate
        audio_frame.time_base = fractions.Fraction(1, self._sample_rate)
        self._samples_sent += len(data) // 2  # 16位样本每个占2字节
        return audio_frame

class HumanPlayer:
    def __init__(self, output_fps=25, audio_sample_rate=32000):
        self.pc = RTCPeerConnection()
        self.output_fps = output_fps
        self.audio_sample_rate = audio_sample_rate
        
        # 共享事件循环
        self._event_loop = asyncio.new_event_loop()
        self.video_buffer = asyncio.Queue(loop=self._event_loop)
        self.audio_buffer = asyncio.Queue(loop=self._event_loop)

        # 音视频轨道
        self.video_track = CustomVideoTrack(self.video_buffer)
        self.audio_track = CustomAudioTrack(self.audio_buffer, self.audio_sample_rate)
        self.pc.addTrack(self.video_track)
        self.pc.addTrack(self.audio_track)
        
        # 启动处理线程
        self._running = True
        threading.Thread(target=self._run_loop, daemon=True).start()

    def _run_loop(self):
        asyncio.set_event_loop(self._event_loop)
        self._event_loop.run_forever()

    def feed_external_data(self, rgb_frames: np.ndarray, pcm_audio: bytes):
        """实时节奏推送"""
        # 视频按帧率推送
        frame_interval = 1.0 / self.output_fps
        for idx, frame in enumerate(rgb_frames):
            asyncio.run_coroutine_threadsafe(
                self.video_buffer.put(VideoFrame.from_ndarray(frame, format="rgb24")),
                loop=self._event_loop
            )
            time.sleep(frame_interval)  # 维持帧率
        
        # 音频按20ms分块推送
        chunk_duration = 0.02  # 20ms
        chunk_size = int(self.audio_sample_rate * chunk_duration)
        pcm_samples = np.frombuffer(pcm_audio, dtype=np.int16)
        
        for i in range(0, len(pcm_samples), chunk_size):
            print("音频发送中........")
            chunk = pcm_samples[i:i+chunk_size].tobytes()
            if chunk:
                asyncio.run_coroutine_threadsafe(
                    self.audio_buffer.put(chunk),
                    loop=self._event_loop
                )
                time.sleep(chunk_duration)  # 维持音频节奏

    def close(self):
        self._running = False
        self._event_loop.call_soon_threadsafe(self._event_loop.stop)
        self.pc.close()