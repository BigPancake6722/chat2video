import av
import subprocess
from io import BytesIO

class AudioProcessor:
    def __init__(self, sample_rate=32000):  # OPUS标准采样率
        self.sample_rate = sample_rate
        self.opus_header = bytes([0x4F, 0x70, 0x75, 0x73, 0x48, 0x65, 0x61, 0x64])  # "OpusHead"
    
    def validate_opus(self, audio_bytes: bytes) -> bool:
        """验证OPUS流有效性"""
        # 检查魔数头
        if len(audio_bytes) < 8:
            return False
        return audio_bytes[:8] == self.opus_header
    
    def repackage_opus(self, audio_bytes: bytes) -> bytes:
        """重新封装为WebRTC兼容格式"""
        if not self.validate_opus(audio_bytes):
            # 尝试转换原始数据
            try:
                return self.raw_to_opus(audio_bytes)
            except Exception as e:
                print(f"OPUS转换失败: {str(e)}")
                return None
        
        # WebRTC需要添加TOC头
        toc_byte = 0x80 | ((1 << 3) & 0x78)  # 单帧，48kHz
        return bytes([toc_byte]) + audio_bytes
    
    def raw_to_opus(self, raw_data: bytes) -> bytes:
        """原始PCM转OPUS"""
        with av.open(BytesIO(raw_data), format='s16le', mode='r') as input_container:
            output_buffer = BytesIO()
            with av.open(output_buffer, mode='w', format='ogg') as output_container:
                output_stream = output_container.add_stream('libopus', rate=self.sample_rate)
                for frame in input_container.decode(audio=0):
                    for packet in output_stream.encode(frame):
                        output_container.mux(packet)
            return output_buffer.getvalue()

class VideoProcessor:
    VP8_HEADER = b'\x9d\x01\x2a'  # VP8帧起始标记
    
    def validate_vp8(self, video_bytes: bytes) -> bool:
        """验证VP8流有效性"""
        return self.VP8_HEADER in video_bytes[:100]  # 检查前100字节
    
    def repackage_vp8(self, video_bytes: bytes) -> bytes:
        """重新封装为WebRTC兼容格式"""
        if not self.validate_vp8(video_bytes):
            try:
                return self.transcode_to_vp8(video_bytes)
            except Exception as e:
                print(f"VP8转码失败: {str(e)}")
                return None
        
        # 添加RTP负载头
        payload_header = bytes([0x80, 0x60, 0x00, 0x00])  # RTP基础头
        return payload_header + video_bytes
    
    def transcode_to_vp8(self, video_bytes: bytes) -> bytes:
        """其他格式转VP8"""
        with av.open(BytesIO(video_bytes), mode='r') as input_container:
            output_buffer = BytesIO()
            with av.open(output_buffer, mode='w', format='webm') as output_container:
                output_stream = output_container.add_stream('libvpx', rate=25)
                output_stream.options = {
                    'quality': 'realtime',
                    'cpu-used': '8',
                    'lag-in-frames': '0'
                }
                for frame in input_container.decode(video=0):
                    for packet in output_stream.encode(frame):
                        output_container.mux(packet)
            return output_buffer.getvalue()

class WebRTCStreamer:
    def __init__(self, signaling_server_url):
        self.peer_connection = RTCPeerConnection()
        self.audio_sender = None
        self.video_sender = None
        self.signaling_server = signaling_server_url
        
        # 配置ICE服务器
        config = RTCConfiguration(iceServers=[
            RTCIceServer(urls="stun:stun.l.google.com:19302")
        ])
        self.peer_connection.configuration = config
        
        # 设置媒体轨道
        self._setup_media_tracks()
    
    def _setup_media_tracks(self):
        """创建媒体轨道"""
        # 音频轨道（OPUS）
        audio_media = MediaStreamTrack(kind='audio')
        self.audio_sender = self.peer_connection.addTrack(audio_media)
        
        # 视频轨道（VP8）
        video_media = MediaStreamTrack(kind='video')
        self.video_sender = self.peer_connection.addTrack(video_media)
        
        # 编解码器优先级设置
        for transceiver in self.peer_connection.getTransceivers():
            if transceiver.sender == self.audio_sender:
                transceiver.setCodecPreferences([RTCRtpCodecParameters(
                    mimeType='audio/opus',
                    clockRate=32000,
                    channels=2
                )])
            elif transceiver.sender == self.video_sender:
                transceiver.setCodecPreferences([RTCRtpCodecParameters(
                    mimeType='video/VP8',
                    clockRate=90000
                )])
    
    async def push_media(self, audio_data: bytes = None, video_data: bytes = None):
        """推送媒体数据"""
        if audio_data:
            await self._push_audio(audio_data)
        if video_data:
            await self._push_video(video_data)
    
    async def _push_audio(self, data: bytes):
        """推送OPUS音频数据"""
        frame = AudioFrame(
            data=data,
            sample_rate=32000,
            channels=1,
            samples_per_channel=960  # 20ms帧
        )
        await self.audio_sender.send(frame)
    
    async def _push_video(self, data: bytes):
        """推送VP8视频数据"""
        frame = VideoFrame(
            data=data,
            width=512,
            height=512,
            format='VP8',
            timestamp=time.time_ns() // 1000
        )
        await self.video_sender.send(frame)
    
    async def start_stream(self):
        """启动信令交互"""
        offer = await self.peer_connection.createOffer()
        await self.peer_connection.setLocalDescription(offer)
        
        # 发送offer到信令服务器
        async with aiohttp.ClientSession() as session:
            await session.post(
                self.signaling_server,
                json={'sdp': self.peer_connection.localDescription.sdp, 'type': 'offer'}
            )