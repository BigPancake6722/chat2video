import json
import time
import queue
import asyncio
import threading
import fractions
import numpy as np
from av import AudioFrame
from av.frame import Frame
from av.packet import Packet
from collections import deque
from av import AudioFrame, VideoFrame
from typing import Tuple, Dict, Optional, Set, Union

AUDIO_PTIME = 0.020  # 20ms audio packetization
VIDEO_CLOCK_RATE = 90000
VIDEO_PTIME = 1 / 25  # 30fps
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
SAMPLE_RATE = 16000
AUDIO_TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)
from aiortc import MediaStreamTrack


class PlayerStreamTrack(MediaStreamTrack):
    """
    A video track that returns an animated flag.
    """

    def __init__(self, player, kind):
        super().__init__()  # don't forget this!
        self.kind = kind
        self._player = player
        self._queue = asyncio.Queue()
        self.timelist = [] #记录最近包的时间戳
        if self.kind == 'video':
            self.framecount = 0
            self.lasttime = time.perf_counter()
            self.totaltime = 0
    
    _start: float
    _timestamp: int

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            raise Exception

        if self.kind == 'video':
            if hasattr(self, "_timestamp"):
                self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
                wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
                if wait>0:
                    await asyncio.sleep(wait)
            else:
                self._start = time.time()
                self._timestamp = 0
                self.timelist.append(self._start)
                print('video start:%f',self._start)
            return self._timestamp, VIDEO_TIME_BASE
        else: #audio
            if hasattr(self, "_timestamp"):
                self._timestamp += int(AUDIO_PTIME * SAMPLE_RATE)
                wait = self._start + (self._timestamp / SAMPLE_RATE) - time.time()
                if wait>0:
                    await asyncio.sleep(wait)
            else:
                self._start = time.time()
                self._timestamp = 0
                self.timelist.append(self._start)
                print('audio start:%f',self._start)
            return self._timestamp, AUDIO_TIME_BASE

    async def recv(self) -> Union[Frame, Packet]:       
        self._player._start(self)
        frame,eventpoint = await self._queue.get()
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        if frame is None:
            self.stop()
            raise Exception
        if self.kind == 'video':
            self.totaltime += (time.perf_counter() - self.lasttime)
            self.framecount += 1
            self.lasttime = time.perf_counter()
            if self.framecount==100:
                print(f"------actual avg final fps:{self.framecount/self.totaltime:.4f}")
                self.framecount = 0
                self.totaltime=0
        return frame
    
    def stop(self):
        super().stop()
        if self._player is not None:
            self._player._stop(self)
            self._player = None

def player_worker_thread(
    quit_event,
    loop,
    container,
    audio_track,
    video_track
):
    container.render(quit_event,loop,audio_track,video_track)



class HumanPlayer:
    def __init__(self):
        # 线程控制
        self.__thread: Optional[threading.Thread] = None
        self.__thread_quit = threading.Event()
        
        # 数据队列（线程安全）
        self.__media_queue = queue.Queue(maxsize=30)  # 存储元组 (audio_data, video_frame)
        self.__audio_track: Optional[PlayerStreamTrack] = None
        self.__video_track: Optional[PlayerStreamTrack] = None
        
        # 同步控制
        self.__sync_buffer = deque(maxlen=10)  # 用于音视频同步的缓冲队列
        self.__last_timestamp = 0  # 维持时间戳连续性

    def create_audio_track(self) -> PlayerStreamTrack:
        """创建音频轨道"""
        if self.__audio_track is None:
            self.__audio_track = PlayerStreamTrack(self, kind="audio")
        return self.__audio_track

    def create_video_track(self) -> PlayerStreamTrack:
        """创建视频轨道"""
        if self.__video_track is None:
            self.__video_track = PlayerStreamTrack(self, kind="video")
        return self.__video_track

    def push_data(self, audio_data: bytes, video_frame: np.ndarray, timestamp: Optional[float] = None):
        """外部数据入口：推送音视频帧"""
        if timestamp is None:
            timestamp = time.monotonic()
            
        # 转换为AV格式
        audio_frame = self._create_audio_frame(audio_data)
        video_frame = self._create_video_frame(video_frame)
        
        # 添加同步时间戳
        sync_pkt = (timestamp, audio_frame, video_frame)
        
        # 非阻塞式入队
        try:
            self.__media_queue.put_nowait(sync_pkt)
        except queue.Full:
            print("Media queue full, dropping frame")
            # 可选：当队列满时丢弃最旧帧
            # self.__media_queue.get_nowait()
            # self.__media_queue.put_nowait(sync_pkt)

    def _create_audio_frame(self, data: bytes) -> AudioFrame:
        """构造音频帧"""
        frame = AudioFrame(format='s16', layout='mono', samples=len(data)//2)
        frame.planes[0].update(data)
        frame.sample_rate = 16000
        return frame

    def _create_video_frame(self, image: np.ndarray) -> VideoFrame:
        """构造视频帧"""
        # 自动转换色彩空间
        if image.shape[-1] == 3:  # RGB转BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return VideoFrame.from_ndarray(image, format='rgb24')

    def _start(self, track: PlayerStreamTrack):
        """轨道激活入口（由PlayerStreamTrack自动调用）"""
        # 自动绑定轨道类型
        if track.kind == "audio":
            self.__audio_track = track
        elif track.kind == "video":
            self.__video_track = track
            
        # 按需启动工作线程
        if self.__thread is None:
            self.__thread = threading.Thread(
                target=self._worker_thread,
                args=(self.__thread_quit, asyncio.get_event_loop())
            )
            self.__thread.start()

    def _worker_thread(self, quit_event: threading.Event, loop: asyncio.AbstractEventLoop):
        """工作线程主循环"""
        print("Media worker thread started")
        
        while not quit_event.is_set():
            try:
                # 非阻塞获取数据
                timestamp, audio_frame, video_frame = self.__media_queue.get(timeout=0.5)
                
                # 音视频同步处理
                self._handle_sync(timestamp, audio_frame, video_frame)
                
                # 推送至轨道
                if self.__audio_track:
                    self._push_audio(audio_frame, loop)
                if self.__video_track:
                    self._push_video(video_frame, loop)
                    
            except queue.Empty:
                self._handle_idle_state()
            except Exception as e:
                print(f"Worker thread error: {str(e)}")
                break
                
        print("Media worker thread exited")

    def _handle_sync(self, timestamp: float, audio: AudioFrame, video: VideoFrame):
        """音视频同步策略"""
        # 维持时间戳连续性
        if timestamp <= self.__last_timestamp:
            timestamp = self.__last_timestamp + 1/25  # 默认30fps补偿
        self.__last_timestamp = timestamp
        
        # 时间戳对齐（单位：微秒）
        base_ts = int(timestamp * 1e6)
        audio.pts = base_ts
        video.pts = base_ts
        
        # 可选：动态调整帧率
        # current_delay = time.monotonic() - timestamp
        # if current_delay > 0.2:  # 超过200ms延迟则丢帧
        #     raise SkipFrameException

    def _push_audio(self, frame: AudioFrame, loop: asyncio.AbstractEventLoop):
        """异步推送音频帧（添加空值检查）"""
        if self.__audio_track and not self.__audio_track._queue.full():
            fut = asyncio.run_coroutine_threadsafe(
                self.__audio_track._queue.put((frame, None)),  # 保持元组结构
                loop
            )
            try:
                fut.result(timeout=1)
            except asyncio.TimeoutError:
                print("Audio track queue full")

    def _push_video(self, frame: VideoFrame, loop: asyncio.AbstractEventLoop):
        """异步推送视频帧（添加空值检查）"""
        if self.__video_track and not self.__video_track._queue.full():
            fut = asyncio.run_coroutine_threadsafe(
                self.__video_track._queue.put((frame, None)),  # 保持元组结构 
                loop
            )
            try:
                fut.result(timeout=1)
            except asyncio.TimeoutError:
                print("Video track queue full")

    def _handle_idle_state(self):
        """空闲状态处理（发送静音帧/黑帧）"""
        if time.monotonic() - self.__last_timestamp > 1.0:  # 超时1秒
            self._push_silence()
            self._push_black_frame()

    def _push_silence(self):
        """生成静音音频帧"""
        silence = bytes([0] * 640 * 2)  # 48kHz/20ms
        frame = self._create_audio_frame(silence)
        asyncio.run_coroutine_threadsafe(
            self.__audio_track._queue.put(frame),
            asyncio.get_event_loop()
        )

    def _push_black_frame(self):
        """生成黑场视频帧"""
        black = np.zeros((512, 512, 3), dtype=np.uint8)
        frame = self._create_video_frame(black)
        asyncio.run_coroutine_threadsafe(
            self.__video_track._queue.put(frame),
            asyncio.get_event_loop()
        )

    def stop(self):
        """停止所有资源"""
        self.__thread_quit.set()
        if self.__thread:
            self.__thread.join(timeout=5)
        self.__media_queue.queue.clear()