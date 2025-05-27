import asyncio
import numpy as np
import av
import aiohttp
import logging
import time
from fractions import Fraction
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from fastapi import HTTPException
from aiortc import (
    RTCPeerConnection,
    RTCConfiguration,
    RTCIceServer,
    MediaStreamTrack,
    RTCSessionDescription
)
from aiortc.mediastreams import AudioFrame, VideoFrame

# 配置参数（保持不变）
SRS_SERVER = "47.111.96.21"
SRS_API_PORT = 1985
SRS_RTC_PORT = 8000
RTMP_URL = "rtmp://47.111.96.21/chat2video/test_session"

AUDIO_SAMPLE_RATE = 48000
AUDIO_FORMAT = "s16le"
AUDIO_LAYOUT = "mono"
AUDIO_SAMPLES = 1024
VIDEO_FRAME_RATE = 30
VIDEO_WIDTH = 512
VIDEO_HEIGHT = 512
VIDEO_FORMAT = "rgb24"

logger = logging.getLogger(__name__)

class SyncedAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate=AUDIO_SAMPLE_RATE):
        super().__init__()
        self._queue = asyncio.Queue(maxsize=100)
        self._sample_rate = sample_rate
        self._time_base = Fraction(1, sample_rate)
        self._lock = asyncio.Lock() 
        self._last_pts = 0

    async def recv(self):
        frame = await self._queue.get()
        # self._last_pts = frame.pts
        return frame

    async def add_frame(self, pcm_data: bytes):
        samples = np.frombuffer(pcm_data, dtype=np.int16)
        frame = AudioFrame.from_ndarray(
            samples[np.newaxis, :],
            format='s16',
            layout='mono'
        )
        frame.rate = self._sample_rate
        async with self._lock:
            frame.pts = self._last_pts + AUDIO_SAMPLES
            frame.time_base = self._time_base
            self._last_pts = frame.pts
            await self._queue.put(frame)


class SyncedVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, width=VIDEO_WIDTH, height=VIDEO_HEIGHT, fps=VIDEO_FRAME_RATE):
        super().__init__()
        self._queue = asyncio.Queue(maxsize=100)
        self._width = width
        self._height = height
        self._time_base = Fraction(1, 90000)
        self._pts_increment = int(90000 / fps)
        self._last_pts = 0
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._lock = asyncio.Lock() 

    async def recv(self):
        return await self._queue.get()

    async def add_frame(self, video_bytes: bytes):
        loop = asyncio.get_event_loop()
        frame_arr = await loop.run_in_executor(
            self._executor,
            self._decode_single_frame,
            video_bytes
        )
        if frame_arr is not None:
            async with self._lock:  # 添加锁
                frame = VideoFrame.from_ndarray(frame_arr, format=VIDEO_FORMAT)
                frame.pts = self._last_pts + self._pts_increment
                frame.time_base = self._time_base
                self._last_pts = frame.pts
                await self._queue.put(frame)

    def _decode_single_frame(self, video_bytes: bytes) -> np.ndarray:
        try:
            with av.open(BytesIO(video_bytes)) as container:
                for frame in container.decode(video=0):
                    return frame.to_ndarray(format=VIDEO_FORMAT)
        except Exception as e:
            logger.error(f"视频解码失败: {str(e)}")
            return None

class RTCStreamPublisher:
    def __init__(self):
        self.pcs = set()
        self.audio_track = SyncedAudioTrack()
        self.video_track = SyncedVideoTrack()
        self.output_container = None
        self.audio_stream = None
        self.video_stream = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._packet_queue = asyncio.Queue(maxsize=100)
        self._audio_base_dts = None  # 新增音频基准DTS
        self._video_base_dts = None  # 新增视频基准DTS

    async def process_audio(self, loop):
        """处理音频流（关键DTS修正）"""
        while True:
            audio_frame = await self.audio_track.recv()
            
            # 计算相对时间
            current_time = time.time()
            expected_time = self._audio_base_time + (audio_frame.pts * self.audio_stream.time_base)
            delay = max(0, expected_time - current_time)
            await asyncio.sleep(delay)
            
            # 编码音频帧
            packets = await loop.run_in_executor(
                self._executor,
                lambda: list(self.audio_stream.encode(audio_frame))
            )
            
            # 记录基准时间（只在第一帧设置）
            if self._audio_base_dts is None and packets:
                self._audio_base_dts = packets[0].dts
                self._audio_base_time = current_time
            
            # 修正DTS并放入队列
            for packet in packets:
                if self._audio_base_dts is not None:
                    # 确保DTS从0开始递增
                    packet.dts -= self._audio_base_dts
                    packet.pts -= self._audio_base_dts
                packet.stream = self.audio_stream
                await self._packet_queue.put(packet)
                logger.debug(f"音频包 dts={packet.dts} pts={packet.pts}")
    
    async def process_video(self, loop):
        """处理视频流（新增完整方法）"""
        while True:
            video_frame = await self.video_track.recv()
            
            # 计算相对时间（基于时间戳）
            current_time = time.time()
            expected_time = self._video_base_time + (video_frame.pts * self.video_stream.time_base)
            delay = max(0, expected_time - current_time)
            await asyncio.sleep(delay)
            
            # 编码视频帧
            packets = await loop.run_in_executor(
                self._executor,
                lambda: list(self.video_stream.encode(video_frame)))
            
            # 记录基准时间（第一帧初始化）
            if self._video_base_dts is None and packets:
                self._video_base_dts = packets[0].dts
                self._video_base_time = current_time
            
            # 修正DTS并放入队列
            for packet in packets:
                if self._video_base_dts is not None:
                    packet.dts -= self._video_base_dts
                    packet.pts -= self._video_base_dts
                packet.stream = self.video_stream
                await self._packet_queue.put(packet)
                logger.debug(f"视频包 dts={packet.dts} pts={packet.pts}")

    async def global_muxer(self, loop):
        """全局封装器（严格DTS校验）"""
        last_dts = {
            self.audio_stream.index: -1,
            self.video_stream.index: -1
        }
        
        while True:
            packet = await self._packet_queue.get()
            stream = packet.stream
            
            # 转换到容器时间基准
            container_stream = self.output_container.streams[stream.index]
            packet_dts = int(packet.dts * stream.time_base / container_stream.time_base)
            
            # DTS单调性检查
            if packet_dts <= last_dts[stream.index]:
                logger.error(f"流{stream.index} DTS错误 {last_dts[stream.index]} >= {packet_dts}，丢弃包")
                continue
            last_dts[stream.index] = packet_dts
            
            # 执行mux
            await loop.run_in_executor(
                self._executor,
                self.output_container.mux,
                packet
            )
            logger.debug(f"Mux成功 流{stream.index} dts={packet_dts}")
    
    async def close(self):
        # 清理资源
        for pc in self.pcs:
            await pc.close()
        self.pcs.clear()
        if self.output_container:
            self.output_container.close()
        self._executor.shutdown()

    async def push_to_srs(self):
        try:
            loop = asyncio.get_event_loop()
            
            # 初始化FFmpeg容器
            self.output_container = await loop.run_in_executor(
                self._executor,
                lambda: av.open(RTMP_URL, 'w', format='flv')
            )
            
            # 配置音频流
            self.audio_stream = self.output_container.add_stream('aac', rate=AUDIO_SAMPLE_RATE)
            self.audio_stream.time_base = Fraction(1, AUDIO_SAMPLE_RATE)
            self.audio_stream.codec_context.options = {
                'strict': 'experimental',
                'fflags': '+genpts'  # 强制生成PTS
            }
            
            # 配置视频流
            self.video_stream = self.output_container.add_stream('libx264', rate=VIDEO_FRAME_RATE)
            self.video_stream.width = VIDEO_WIDTH
            self.video_stream.height = VIDEO_HEIGHT
            self.video_stream.pix_fmt = 'yuv420p'
            self.video_stream.time_base = Fraction(1, 90000)
            self.video_stream.codec_context.options = {
                'bframes': '0',
                'preset': 'ultrafast',
                'tune': 'zerolatency',
                'fflags': '+genpts'  # 确保生成时间戳
            }

            # 添加DTS监控逻辑
            last_audio_dts = {}
            last_video_dts = {}

            # 独立处理音视频流
            async def process_audio():
                last_pts_time = time.time()
                while True:
                    audio_frame = await self.audio_track.recv()
                    
                    # 动态等待
                    current_time = time.time()
                    expected_time = last_pts_time + (audio_frame.pts - self.audio_track._last_pts) * self.audio_stream.time_base
                    delay = max(0, expected_time - current_time)
                    await asyncio.sleep(delay)
                    
                    # 编码
                    packets = await loop.run_in_executor(
                        self._executor,
                        lambda: list(self.audio_stream.encode(audio_frame)))
                    
                    # Mux
                    for packet in packets:
                        current_dts = packet.dts
                        if last_audio_dts.get(packet.stream.index, -1) >= current_dts:
                            logger.error(f"音频DTS错误: {last_audio_dts.get(packet.stream.index)} >= {current_dts}")
                        last_audio_dts[packet.stream.index] = current_dts
                        await self._packet_queue.put(packet)
                    
                    last_pts_time = time.time()

            async def process_video():
                last_pts = None
                while True:
                    video_frame = await self.video_track.recv()
                    
                    # 动态等待
                    if last_pts is not None:
                        interval = (video_frame.pts - last_pts) * self.video_stream.time_base
                        await asyncio.sleep(interval)
                    last_pts = video_frame.pts
                    
                    # 编码
                    packets = await loop.run_in_executor(
                        self._executor,
                        lambda: list(self.video_stream.encode(video_frame))
                    )
                    
                    # Mux
                    for packet in packets:
                        await loop.run_in_executor(
                            self._executor,
                            self.output_container.mux,
                            packet
                        )

            await asyncio.gather(
                self.process_audio(loop),
                self.process_video(loop),
                self.global_muxer(loop)
            )
        except Exception as e:
            await self.close()
            raise HTTPException(status_code=500, detail=str(e))

    async def push_media(self, audio_data: bytes, video_data: bytes):
        try:
            await asyncio.gather(
                self.audio_track.add_frame(audio_data),
                self.video_track.add_frame(video_data)
            )
        except Exception as e:
            logger.error(f"媒体数据推送失败: {str(e)}")
            raise