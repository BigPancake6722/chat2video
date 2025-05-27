import av
import time
import librtmp
import numpy as np
from io import BytesIO
from typing import Optional
from librtmp import RTMP

RTMP_URL = "rtmp://47.111.96.21:1935/live/test_session"

class AudioProcessor:
    def __init__(self, channels=1, sample_rate=32000):
        self.channels = channels
        self.sample_rate = sample_rate
        self._validate_sample_rate()

    def _validate_sample_rate(self):
        valid_rates = [96000, 88200, 64000, 48000, 44100, 32000,
                      24000, 22050, 16000, 12000, 11025, 8000, 7350]
        if self.sample_rate not in valid_rates:
            raise ValueError(f"不支持的采样率: {self.sample_rate}")

    def process_audio(self, audio_bytes: bytes) -> Optional[bytes]:
        if self._is_valid_aac(audio_bytes):
            return audio_bytes
        return self._convert_to_aac(audio_bytes)

    def _is_valid_aac(self, data: bytes) -> bool:
        return len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xF0) == 0xF0

    def _convert_to_aac(self, raw_audio: bytes) -> Optional[bytes]:
        try:
            aligned_audio = self._align_data(raw_audio, 16)
            with av.open(BytesIO(aligned_audio), format='s16le', mode='r') as input_container:
                output_buffer = BytesIO()
                with av.open(output_buffer, mode='w', format='adts') as output_container:
                    return self._encode_audio_stream(input_container, output_container)
        except Exception as e:
            print(f"音频转换失败: {str(e)}")
            return None

    def _encode_audio_stream(self, input_container, output_container):
        output_stream = output_container.add_stream('aac', rate=self.sample_rate)
        output_stream.bit_rate = 128000
        output_stream.layout = f"{self.channels}c"

        for frame in input_container.decode(audio=0):
            aligned_frame = self._align_audio_frame(frame)
            for packet in output_stream.encode(aligned_frame):
                output_container.mux(packet)

        for packet in output_stream.encode(None):
            output_container.mux(packet)

        return self._get_aligned_output(output_container)

    def _align_audio_frame(self, frame: av.AudioFrame) -> av.AudioFrame:
        buffer = frame.planes[0]
        aligned_buffer = self._align_data(buffer, 16)
        return av.AudioFrame.from_ndarray(
            np.frombuffer(aligned_buffer, dtype=np.int16),
            layout=frame.layout.name,
            format=frame.format.name
        )

    @staticmethod
    def _align_data(data: bytes, alignment: int) -> bytes:
        padding = (-len(data)) % alignment
        return data + b'\x00' * padding

    @staticmethod
    def _get_aligned_output(container) -> bytes:
        output = container.getbuffer()
        return output[:len(output) - (len(output) % 16)]

class VideoProcessor:
    def __init__(self, width=512, height=512, fps=25):
        self.width = width
        self.height = height
        self.fps = fps
        self._codec = None

    def __del__(self):
        if self._codec and self._codec.is_open:
            self._codec.close()

    def process_video(self, video_bytes: bytes) -> Optional[bytes]:
        try:
            self._init_codec()
            headers = self._generate_headers()
            return self._insert_headers(video_bytes, headers)
        except Exception as e:
            print(f"视频处理失败: {str(e)}")
            return None

    def _init_codec(self):
        if not self._codec:
            self._codec = av.CodecContext.create('libx264', 'w')
            self._codec.width = self.width
            self._codec.height = self.height
            self._codec.pix_fmt = 'yuv420p'
            self._codec.framerate = self.fps
            self._codec.options = {
                'profile': 'baseline',
                'x264-params': 'keyint=60:min-keyint=60:annexb=1'
            }
            self._codec.open()

    def _generate_headers(self) -> bytes:
        if self._codec.extradata:
            return self._parse_avcc_headers()
        return self._default_headers()

    def _parse_avcc_headers(self) -> bytes:
        try:
            sps, pps = self._parse_avcc_data(self._codec.extradata)
            return b''.join([b'\x00\x00\x00\x01' + s for s in sps + pps])
        except:
            return self._default_headers()

    @staticmethod
    def _parse_avcc_data(data: bytes):
        # 保持原有解析逻辑
        pass

    def _default_headers(self) -> bytes:
        sps = bytes.fromhex('6764001EAC2C')
        pps = bytes.fromhex('68EBE3CB')
        return b'\x00\x00\x00\x01' + sps + b'\x00\x00\x00\x01' + pps

    def _insert_headers(self, data: bytes, headers: bytes) -> bytes:
        aligned_data = self._align_data(data + headers, 16)
        return aligned_data

    @staticmethod
    def _align_data(data: bytes, alignment: int) -> bytes:
        padding = (-len(data)) % alignment
        return data + b'\x00' * padding

class RTMPStreamPublisher:
    def __init__(self, rtmp_url=RTMP_URL):
        self.rtmp_url = rtmp_url
        self._connection = None
        self._stream = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def connect(self):
        try:
            self._connection = RTMP(self.rtmp_url, live=True)
            self._connection.connect()
            self._stream = self._connection.create_stream()
            self._write_flv_header()
        except Exception as e:
            self.close()
            raise RuntimeError(f"RTMP连接失败: {str(e)}")

    def push(self, audio: bytes, video: bytes):
        if not self._stream:
            raise ConnectionError("未建立RTMP连接")

        try:
            self._send_metadata()
            self._send_frames(audio, video)
        finally:
            self.close()

    def _send_frames(self, audio: bytes, video: bytes):
        timestamp = 0
        while True:
            self._send_video(video, timestamp)
            self._send_audio(audio, timestamp)
            timestamp += 40  # 25fps
            time.sleep(0.04)

    def _write_flv_header(self):
        self._stream.write(b'FLV\x01\x05\x00\x00\x00\x09\x00\x00\x00\x00')

    def _send_metadata(self):
        metadata = self._build_metadata()
        self._stream.write(metadata)

    def _build_metadata(self) -> bytes:
        # 保持原有元数据构建逻辑
        pass

    def _send_video(self, data: bytes, timestamp: int):
        aligned_data = self._align_data(data, 16)
        tag = self._build_tag(0x09, aligned_data, timestamp)
        self._stream.write(tag)

    def _send_audio(self, data: bytes, timestamp: int):
        aligned_data = self._align_data(data, 16)
        tag = self._build_tag(0x08, aligned_data, timestamp)
        self._stream.write(tag)

    @staticmethod
    def _align_data(data: bytes, alignment: int) -> bytes:
        padding = (-len(data)) % alignment
        return data + b'\x00' * padding

    def _build_tag(self, tag_type: int, data: bytes, timestamp: int) -> bytes:
        header = bytes([
            tag_type,
            (len(data) >> 16) & 0xFF,
            (len(data) >> 8) & 0xFF,
            len(data) & 0xFF,
            (timestamp >> 16) & 0xFF,
            (timestamp >> 8) & 0xFF,
            timestamp & 0xFF,
            (timestamp >> 24) & 0x7F,
            0x00, 0x00, 0x00
        ])
        return header + data

    def close(self):
        if self._stream:
            try:
                self._stream.close()
            except:
                pass
            self._stream = None
        if self._connection:
            try:
                self._connection.close()
            except:
                pass
            self._connection = None

def test_connection():
    try:
        with RTMPStreamPublisher() as publisher:
            print("RTMP连接成功")
            return True
    except Exception as e:
        print(f"连接失败: {str(e)}")
        return False

def rtmp_push(audio_bytes : bytes, video_bytes : bytes):
    if test_connection():
        audio_processor = AudioProcessor()
        video_processor = VideoProcessor()

        # 示例数据
        test_audio = b'\x00' * 16000  # 1秒16kHz采样率
        test_video = b'\x00' * (512*512*3)  # 测试帧

        processed_audio = audio_processor.process_audio(test_audio)
        processed_video = video_processor.process_video(test_video)

        if processed_audio and processed_video:
            with RTMPStreamPublisher() as publisher:
                publisher.push(processed_audio, processed_video)
    