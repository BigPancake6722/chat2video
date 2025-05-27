import numpy as np
import av
import threading
import time
import asyncio
import fractions  # 新增导入
import sys
from io import BytesIO
import soundfile as sf
from av import AudioFrame, VideoFrame
from aiortc import RTCPeerConnection, MediaStreamTrack
from aiortc.mediastreams import AudioStreamTrack, VideoStreamTrack

sys.path.append("/root/chat2video/sync-gaussian-talker/")
from render_transfer1 import main as audio2video

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

        # 静音任务控制
        self._silent_task = None
        self._silent_running = False
        self._silent_frames = np.array([])
        self.model_id = ""
        
        # 启动处理线程
        self._start_threads()

    def _start_threads(self):
        self.running = True
        threading.Thread(target=self._video_loop, daemon=True).start()
        threading.Thread(target=self._audio_loop, daemon=True).start()
    
    
    def start_silent(self):
        """启动静音推送任务"""
        if not self._silent_running:
            self._silent_running = True
            self._silent_task = asyncio.create_task(self._push_silent_frames())

    def stop_silent(self):
        """停止静音推送任务"""
        self._silent_running = False
        if self._silent_task and not self._silent_task.done():
            self._silent_task.cancel()
    
    async def _push_silent_frames(self):
        """持续推送静音帧的任务"""
        while self._silent_running:
            try:
                # 生成并推送静音数据
                self.push_silent_frames()
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"静音推送异常: {e}")
                break

    def feed_external_data(self, rgb_frames: np.ndarray, opus_audio: bytes):
        """线程安全的音视频数据输入"""
        # 视频处理（直接推送到异步队列）
        for frame in rgb_frames:
            asyncio.run_coroutine_threadsafe(
                self.video_buffer.put(VideoFrame.from_ndarray(frame, format="rgb24")),
                loop=self._video_loop_event_loop
            )
        
        # 音频处理（分块推送到异步队列）
        chunk_size = int(self.audio_sample_rate * 0.02)  # 20ms的样本数
        pcm_samples = np.frombuffer(pcm_audio, dtype=np.int16)
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


    # 生成静音音频流
    def generate_silent_wav_array(self, duration) -> np.ndarray:
        # 参数配置
        sample_rate = 32000    # 采样率32kHz
        num_samples = int(sample_rate * duration)
        
        # 创建静音数据 (int16全零数组)
        silent_data = np.zeros(num_samples, dtype=np.int16)
        
        # 通过内存文件操作生成WAV格式数据
        with BytesIO() as wav_buffer:
            # 写入内存中的WAV文件
            sf.write(
                wav_buffer,
                silent_data,
                sample_rate,
                format='WAV',
                subtype='PCM_16',  # 16位样本（2字节宽度）
                endian='LITTLE'    # WAV标准使用小端序
            )
            
            # 重置指针并读取验证
            wav_buffer.seek(0)
            audio_data, sr = sf.read(
                wav_buffer,
                dtype='int16',
                always_2d=False     # 保持单声道形状为(n,)
            )
        
        # 验证参数
        assert sr == sample_rate, f"采样率错误: {sr} ≠ {sample_rate}"
        assert audio_data.dtype == np.int16, "数据类型错误"
        assert audio_data.shape == (num_samples,), f"形状错误: {audio_data.shape}"
        assert len(audio_data) == duration * sample_rate, "样本数错误"
        
        return audio_data

    def push_silent_frames(self, model_id="Chinese1"):
        from model_params import model_params
        params = model_params[model_id]
        if not self._silent_frames.size > 0:
            self._silent_frames = audio2video(
                ['--model_path', params['model_path'],
                    '--source_path', params['source_path'],
                    '--batch', '16'],
                self.generate_silent_wav_array(2.0)
            )
        
        # 推送数据到Player
        self.feed_external_data(
            rgb_frames=np.array(self._silent_frames, dtype=np.uint8),
            pcm_audio=np.zeros(2 * 32000, dtype=np.int16).astype('<i2').tobytes()
        )

    def close(self):
        self.running = False
        # 停止事件循环
        self._video_loop_event_loop.call_soon_threadsafe(self._video_loop_event_loop.stop)
        self._audio_loop_event_loop.call_soon_threadsafe(self._audio_loop_event_loop.stop)
        self.pc.close()