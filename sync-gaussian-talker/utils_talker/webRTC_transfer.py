from aiortc import VideoStreamTrack
from av import VideoFrame
import cv2


class GaussianVideoStream(VideoStreamTrack):
    """WebRTC视频流生成器"""
    def __init__(self, render_generator):
        super().__init__()
        self.render_generator = render_generator
        self.iter_obj = None

    def __iter__(self):
        self.iter_obj = iter(self.render_generator)
        return self

    def __next__(self):
        if self.iter_obj is None:
            raise StopIteration
        try:
            frames = next(self.iter_obj)
            for frame in frames:
                # 转换为WebRTC需要的格式
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
                video_frame.pts, video_frame.time_base = self.next_timestamp()
                return video_frame
        except StopIteration:
            raise

    async def recv(self):
        # 这里可以根据需要调整，如果迭代器逻辑已经在__next__中处理好，这里可以简单调用__next__
        try:
            return next(self)
        except StopIteration:
            return None
