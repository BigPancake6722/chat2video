import asyncio
import numpy as np
import av
import aiohttp
import time
from fractions import Fraction
from io import BytesIO
from fastapi import HTTPException
from aiortc import (
    RTCPeerConnection, 
    RTCConfiguration, 
    RTCIceServer,
    MediaStreamTrack,
    RTCSessionDescription
)
from aiortc.mediastreams import AudioFrame, VideoFrame
# 服务器配置
SRS_SERVER = "47.111.96.21"
SRS_API_PORT = 1985
SRS_RTC_PORT = 8000

# 音视频参数
AUDIO_SAMPLE_RATE = 48000
AUDIO_FORMAT = "s16le"
AUDIO_LAYOUT = "mono"
AUDIO_SAMPLES = 1024
VIDEO_FRAME_RATE = 30
VIDEO_WIDTH = 512
VIDEO_HEIGHT = 512
VIDEO_FORMAT = "rgb24"

# logger = logging.getLogger(__name__)
class SyncedAudioTrack(MediaStreamTrack):
    kind = "audio"
    
    def __init__(self, sample_rate=AUDIO_SAMPLE_RATE):
        super().__init__()
        self._queue = asyncio.Queue()
        self._sample_rate = sample_rate
        self._time_base = Fraction(1, sample_rate)
        self._last_pts = 0
    
    async def recv(self):
        frame = await self._queue.get()
        self._last_pts = frame.pts
        return frame
    
    async def add_frame(self, pcm_data: bytes):
        samples = np.frombuffer(pcm_data, dtype=np.int16)
        frame = AudioFrame.from_ndarray(
            samples[np.newaxis, :],  # mono
            format='s16',
            layout='mono'
        )
        frame.rate = self._sample_rate
        frame.pts = self._last_pts + AUDIO_SAMPLES
        frame.time_base = self._time_base
        await self._queue.put(frame)

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
    
    async def recv(self):
        frame = await self._queue.get()
        self._last_pts = frame.pts
        return frame
    
    def video_stream_to_numpy(self, video_stream: BytesIO) -> np.ndarray:
        frames = []
        video_stream.seek(0)
        container = av.open(video_stream)
        for frame in container.decode(video=0):
            frame_array = frame.to_ndarray(format=VIDEO_FORMAT)
            frames.append(frame_array)
        container.close()
        result = np.stack(frames, axis=0)
        assert result.shape[1:] == (512, 512, 3)
        return result

    
    async def add_frame(self, video_bytes: BytesIO):
        for frame_arr in self.video_stream_to_numpy(video_bytes):
            frame = VideoFrame.from_ndarray(frame_arr, format=VIDEO_FORMAT)
            frame.pts = self._last_pts + self._pts_increment
            frame.time_base = self._time_base
            await self._queue.put(frame)

class RTCStreamPublisher:
    def __init__(self):
        self.pcs = set()
        self.audio_track = SyncedAudioTrack()
        self.video_track = SyncedVideoTrack()
    
    async def push_to_srs(self, session_id: str):
        pc_config = RTCConfiguration(iceServers=[
            RTCIceServer(urls="stun:stun.l.google.com:19302?transport=tcp"),
            RTCIceServer(
                urls=f"turn:{SRS_SERVER}:3478?transport=tcp", 
                username="root",
                credential="root",
                credentialType="password"
            )]
        )
        pc = RTCPeerConnection(configuration=pc_config)
        self.pcs.add(pc)
        
        try:
            pc.addTrack(self.audio_track)
            pc.addTrack(self.video_track)
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://{SRS_SERVER}:{SRS_API_PORT}/rtc/v1/whip/?app=live&stream={session_id}",
                    data=offer.sdp,
                    headers={"Content-Type": "sdp"}
                ) as resp:
                    if resp.status != 201:
                        error = await resp.text()
                        raise Exception(f"SRS API错误: {resp.status} - {error}")
                    answer = await resp.text()
                    await pc.setRemoteDescription(
                        RTCSessionDescription(sdp=answer, type="answer")
                    )
            print(f"推流会话 {session_id} 建立成功")
            return pc
            
        except Exception as e:
            print(f"推流失败: {str(e)}")
            await pc.close()
            raise HTTPException(status_code=500, detail=str(e))
    
    async def push_media(self, audio_data: bytes, video_data: np.ndarray):
        try:
            await self.audio_track.add_frame(audio_data)
            await self.video_track.add_frame(video_data)
            await asyncio.sleep(1/VIDEO_FRAME_RATE)
            
        except Exception as e:
            print(f"推送媒体帧失败: {str(e)}")
            raise