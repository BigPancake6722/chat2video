import asyncio
import numpy as np
import av
import logging
from fractions import Fraction
from io import BytesIO
from fastapi import HTTPException
from aiortc import MediaStreamTrack
from aiortc.mediastreams import AudioFrame, VideoFrame

# 服务器配置
SRS_SERVER = "47.111.96.21"
SRS_API_PORT = 1985
SRS_RTC_PORT = 8000
RTMP_URL = "rtmp://47.111.96.21/live/livestream"

# 音视频参数
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
        self._queue = asyncio.Queue()
        self._sample_rate = sample_rate
        self._time_base = Fraction(1, sample_rate)
        self._last_pts = 0
        self._lock = asyncio.Lock()

    async def recv(self):
        async with self._lock:
            frame = await self._queue.get()
            logger.debug("audio frame pts changes in recv: %s %s", self._last_pts, frame.pts)
            self._last_pts = frame.pts
            return frame

    async def add_frame(self, pcm_data: bytes):
        async with self._lock:
            samples = np.frombuffer(pcm_data, dtype=np.int16)
            frame = AudioFrame.from_ndarray(
                samples[np.newaxis, :],  # mono
                format='s16',
                layout='mono'
            )
            frame.rate = self._sample_rate
            new_pts = self._last_pts + AUDIO_SAMPLES
            if new_pts <= self._last_pts:
                new_pts = self._last_pts + 1
            frame.pts = new_pts
            frame.time_base = self._time_base
            await self._queue.put(frame)
            self._last_pts = new_pts


class SyncedVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, width=VIDEO_WIDTH, height=VIDEO_HEIGHT, fps=VIDEO_FRAME_RATE):
        super().__init__()
        self._queue = asyncio.Queue()
        self._width = width
        self._height = height
        self._time_base = Fraction(1, 90000)
        self._pts_increment = int(90000 / fps)
        self._last_pts = 0
        self._lock = asyncio.Lock()

    async def recv(self):
        async with self._lock:
            frame = await self._queue.get()
            logger.debug("video frame pts changes in recv: %s %s", self._last_pts, frame.pts)
            self._last_pts = frame.pts
            return frame

    async def video_stream_to_numpy(self, video_stream: BytesIO) -> np.ndarray:
        frames = []
        video_stream.seek(0)
        container = await asyncio.to_thread(av.open, video_stream)
        try:
            for frame in container.decode(video=0):
                frame_array = await asyncio.to_thread(frame.to_ndarray, format=VIDEO_FORMAT)
                frames.append(frame_array)
        finally:
            container.close()
        result = np.stack(frames, axis=0)
        if result.shape[1:] != (512, 512, 3):
            raise ValueError("Video frame shape does not match expected shape.")
        return result

    async def add_frame(self, video_bytes: bytes):
        async with self._lock:
            video_stream = BytesIO(video_bytes)
            frame_arrays = await self.video_stream_to_numpy(video_stream)
            for frame_arr in frame_arrays:
                frame = VideoFrame.from_ndarray(frame_arr, format=VIDEO_FORMAT)
                new_pts = self._last_pts + self._pts_increment
                if new_pts <= self._last_pts:
                    new_pts = self._last_pts + 1
                frame.pts = new_pts
                frame.time_base = self._time_base
                await self._queue.put(frame)
                self._last_pts = new_pts
                logger.debug("video frame pts changes in add_frame: %s %s", self._last_pts, frame.pts)


class RTCStreamPublisher:
    def __init__(self):
        self.pcs = set()
        self.audio_track = SyncedAudioTrack()
        self.video_track = SyncedVideoTrack()
        self.output_container = None
        self.audio_stream = None
        self.video_stream = None

    async def push_to_srs(self):
        try:
            self.output_container = await asyncio.to_thread(av.open, RTMP_URL, 'w', format='flv')
            try:
                self.audio_stream = self.output_container.add_stream(
                    codec_name='aac', rate=AUDIO_SAMPLE_RATE)
                self.audio_stream.time_base = Fraction(1, AUDIO_SAMPLE_RATE)

                self.video_stream = self.output_container.add_stream(
                    codec_name='libx264', rate=VIDEO_FRAME_RATE)
                self.video_stream.width = VIDEO_WIDTH
                self.video_stream.height = VIDEO_HEIGHT
                self.video_stream.pix_fmt = 'yuv420p'
                self.video_stream.time_base = Fraction(1, 90000)

                while True:
                    logger.debug("Pushing frames...")
                    audio_frame = await self.audio_track.recv()
                    audio_packet = self.audio_stream.encode(audio_frame)
                    for packet in audio_packet:
                        packet.stream = self.audio_stream
                        self.output_container.mux(packet)
                    logger.debug("音频帧推送完成 %s %s %s", audio_frame.pts, self.audio_stream.time_base.denominator,
                                 self.audio_stream.time_base.numerator)
                    # 获取视频帧
                    video_frame = await self.video_track.recv()
                    logger.debug("视频帧pts计算 %s %s %s", video_frame.pts, self.video_stream.time_base.denominator,
                                 self.video_stream.time_base.numerator)
                    video_frame.pts = video_frame.pts * self.video_stream.time_base.denominator // self.video_stream.time_base.numerator
                    logger.debug("current video frame pts: %s", video_frame.pts)
                    video_packet = self.video_stream.encode(video_frame)
                    for packet in video_packet:
                        packet.stream = self.video_stream
                        self.output_container.mux(packet)
                    await asyncio.sleep(1 / VIDEO_FRAME_RATE)
                    logger.debug("视频帧推送完成 %s %s %s", video_frame.pts, self.video_stream.time_base.denominator,
                                 self.video_stream.time_base.numerator)
            except Exception as e:
                logger.error("推流失败: %s", str(e), exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
            finally:
                if self.output_container:
                    self.output_container.close()
        except Exception as e:
            logger.error("打开输出容器失败: %s", str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def push_media(self, audio_data: bytes, video_data: bytes):
        try:
            await self.audio_track.add_frame(audio_data)
            await self.video_track.add_frame(video_data)
            await asyncio.sleep(1 / VIDEO_FRAME_RATE)
        except Exception as e:
            logger.error("推送媒体帧失败: %s", str(e), exc_info=True)
            raise