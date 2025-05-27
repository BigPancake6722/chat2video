import numpy as np
import av
import threading
import time
import asyncio
from io import BytesIO
from collections import deque
from pydub import AudioSegment
from av import AudioFrame, VideoFrame
from aiortc import MediaStreamTrack, RTCPeerConnection
from aiortc.mediastreams import AudioStreamTrack, VideoStreamTrack
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, RTCRtpSender

class CustomVideoTrack(VideoStreamTrack):
    def __init__(self, buffer):
        super().__init__()
        self.buffer = buffer
        self._last_pts = 0
        
    async def recv(self):
        frame = await self.buffer.get()
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        return frame

class CustomAudioTrack(AudioStreamTrack):
    def __init__(self, buffer):
        super().__init__()
        self.buffer = buffer
        self._sample_rate = 32000
        self._pts = 0

    async def recv(self):
        data = await self.buffer.get()
        audio_frame = AudioFrame.from_ndarray(
            np.frombuffer(data, dtype=np.int16).reshape(-1, 1),
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
        # WebRTC 初始化
        self.pc = RTCPeerConnection()
        self.output_fps = output_fps
        self.audio_sample_rate = audio_sample_rate
        
        # 缓存
        self.video_buffer = asyncio.Queue()
        self.audio_buffer = asyncio.Queue()

        # 初始化音视频轨道
        self.audio_track = AudioStreamTrack()
        self.video_track = CustomVideoTrack(self.video_buffer)
        self.pc.addTrack(self.audio_track)
        self.pc.addTrack(self.video_track)
        
        # 控制状态
        self.running = False
        self._lock = threading.Lock()
        
        # 启动处理线程
        self._start_threads()

    def _start_threads(self):
        """启动音视频处理线程"""
        self.running = True
        threading.Thread(target=self._video_loop, daemon=True).start()
        threading.Thread(target=self._audio_loop, daemon=True).start()

    def feed_external_data(self, rgb_frames: np.ndarray, pcm_audio: bytes):
        """外部数据输入接口"""
        print("写入音视频流")
        # 视频帧处理
        with self._lock:
            for frame in rgb_frames:
                self.raw_video_frames.append(frame)
        
        # 音频解码
        pcm = np.frombuffer(pcm_audio, dtype=np.int16)
        with self._lock:
            self.audio_buffer.extend(pcm)

    def _video_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def async_audio_loop():
            chunk_size = int(self.audio_sample_rate * 0.02)
            while self.running:
                with self._lock:
                    if len(self.audio_buffer) < chunk_size:
                        await asyncio.sleep(0.001)
                        continue
                    chunk = np.array([self.audio_buffer.popleft() for _ in range(chunk_size)])
                
                # 推送至异步队列
                await self.audio_buffer.put(chunk.tobytes())
        
        loop.run_until_complete(async_audio_loop())

    def _audio_loop(self):
        print("正在推送音频流")
        """音频推流循环"""
        chunk_size = int(self.audio_sample_rate * 0.02)  # 20ms
        while self.running:
            with self._lock:
                if len(self.audio_buffer) < chunk_size:
                    time.sleep(0.001)
                    continue
                chunk = np.array([self.audio_buffer.popleft() for _ in range(chunk_size)])
            
            # 推送音频数据
            self.audio_track.push_data(chunk.tobytes())

    def close(self):
        """释放资源"""
        self.running = False
        self.pc.close()