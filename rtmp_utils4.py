import av
import time
import librtmp
from librtmp import RTMP
from pydub import AudioSegment
from io import BytesIO


RTMP_URL = "rtmp://47.111.96.21/chat2video/test_session"
class AudioProcessor:
    def __init__(self, channels=1, sample_rate=32000):
        self.channels_config = channels
        self.sample_rate_index = sample_rate
        self.config_mapping(channels, sample_rate)

    def config_mapping(self, channels, sample_rate):
        # 声道数映射表
        if channels in range(1, 7):
            self.channel_config = channels
        else:
            print("不支持的声道数。")
            return None

        # 采样率索引映射表
        sample_rate_index_list = [96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350]
        self.sample_rate_index = sample_rate_index_list.index(sample_rate) if sample_rate in sample_rate_index_list else -1
        if self.sample_rate_index == -1: 
            print("不支持的采样率。")
            return None

    # 验证并处理音频流
    def validate_and_add_header_audio(self, audio_bytes: bytes):
        # 验证音频流是否为 AAC 编码
        # AAC 裸流以 ADTS 头开头，ADTS 头第一个字节以 0xFF 开头
        if len(audio_bytes) < 2 or audio_bytes[0] != 0xFF or (audio_bytes[1] & 0xF0) != 0xF0:
            print("音频流不符合 RTMP 传输要求，不是有效的 AAC 编码，尝试转为AAC编码。")
            audio_bytes = self.raw_to_aac_bytes(audio_bytes)
        if len(audio_bytes) < 2 or audio_bytes[0] != 0xFF or (audio_bytes[1] & 0xF0) != 0xF0:
            print("音频流不符合 RTMP 传输要求，不是有效的 AAC 编码。")
            return None
            

        # 检查是否有 ADTS 头，如果没有则添加
        if len(audio_bytes) < 7 or audio_bytes[0] != 0xFF or (audio_bytes[1] & 0xF0) != 0xF0:
            # 计算 ADTS 头
            adts_fixed_header = 0xFFF1  # syncword + ID + layer + protection_absent
            adts_fixed_header = (adts_fixed_header << 2) | 0  # profile
            adts_fixed_header = (adts_fixed_header << 4) | index  # sampling_frequency_index
            adts_fixed_header = (adts_fixed_header << 1) | 0  # private_bit
            adts_fixed_header = (adts_fixed_header << 3) | channel_config  # channel_configuration
            adts_fixed_header = (adts_fixed_header << 1) | 0  # original/copy
            adts_fixed_header = (adts_fixed_header << 1) | 0  # home

            adts_variable_header = 0
            adts_variable_header = (adts_variable_header << 1) | 0  # copyright_identification_bit
            adts_variable_header = (adts_variable_header << 1) | 0  # copyright_identification_start
            frame_length = len(audio_bytes) + 7
            adts_variable_header = (adts_variable_header << 13) | frame_length  # frame_length
            adts_variable_header = (adts_variable_header << 11) | 0x7FF  # adts_buffer_fullness
            adts_variable_header = (adts_variable_header << 2) | 0  # number_of_raw_data_blocks_in_frame

            adts_header = (adts_fixed_header << 28) | adts_variable_header
            adts_header_bytes = adts_header.to_bytes(7, byteorder='big')
            audio_bytes = adts_header_bytes + audio_bytes
        
        return audio_bytes

    
    def raw_to_aac_bytes(self, raw_audio: bytes, sample_rate: int = 32000, sample_format: str = 's16', channels: int = 1, bit_rate: int = 128000) -> bytes:
        output_buffer = BytesIO()
    
        # 创建虚拟输入容器（关键修改点）
        input_container = av.open(
            BytesIO(raw_audio),
            format='s16le',  # 原始 PCM 格式
            mode='r',
            options={
                'ar': str(sample_rate),    # 采样率
                'ac': str(channels),       # 声道数
                'sample_fmt': sample_format  # 采样格式
            }
        )
        
        # 创建输出容器
        with av.open(output_buffer, mode='w', format='adts') as output_container:
            # 添加 AAC 编码流
            output_stream = output_container.add_stream('aac', rate=sample_rate)
            output_stream.bit_rate = bit_rate
            output_stream.layout = f"{channels}c"
            
            # 直接解码输入数据（无需手动添加流）
            for frame in input_container.decode(audio=0):
                # 重新设置时间戳避免警告
                frame.pts = None
                
                # 编码并复用
                for packet in output_stream.encode(frame):
                    output_container.mux(packet)
            
            # 刷新编码器
            for packet in output_stream.encode(None):
                output_container.mux(packet)
        
        input_container.close()
        return output_buffer.getvalue()

class VideoProcessor:
    def __init__(self, width=512, height=512, fps=25, pix_fmt='yuv420p'):
        self.width = width
        self.height = height
        self.fps = fps
        self.pix_fmt = pix_fmt

    # 验证并处理视频流
    def validate_and_add_header_video(self, video_bytes):
        def generate_default_sps_pps(width=512, height=512, fps=25):
            """生成512x512分辨率默认参数集"""
            # 默认SPS (Baseline Profile Level 3.1)
            sps = bytes.fromhex('6764001EAC2C')
            # 默认PPS
            pps = bytes.fromhex('68EBE3CB')
            return sps, pps

        # 阶段2：创建编码器上下文（修正参数设置）
        codec = av.CodecContext.create('libx264', 'w')

        # 后置参数配置
        codec.width = self.width       # 单独设置宽度
        codec.height = self.height     # 单独设置高度
        codec.pix_fmt = 'yuv420p'      # 像素格式
        codec.framerate = self.fps     # 帧率
        
        # 通过选项字典配置编码参数
        codec.options = {
            'profile': 'baseline',
            'x264-params': 'keyint=60:min-keyint=60:annexb=1'
        }

        try:
            codec.open()
        except av.AVError as e:
            print(f"编码器初始化失败: {str(e)}")
            return None

        extradata = codec.extradata
        if not extradata:
            print("警告：编码器未生成extradata，使用默认参数集")
            sps, pps = generate_default_sps_pps()
            annexb_headers = b'\x00\x00\x00\x01' + sps + b'\x00\x00\x00\x01' + pps
            return annexb_headers + video_bytes

        # 阶段3：解析AVCDecoderConfigurationRecord
        def parse_avcc(data):
            """解析AVC配置记录获取SPS/PPS"""
            try:
                if len(data) < 7 or data[0] != 0x01:
                    return None, None

                # 解析SPS
                sps_count = data[5] & 0x1F
                offset = 6
                sps_list = []
                for _ in range(sps_count):
                    if offset+2 > len(data):
                        break
                    sps_len = int.from_bytes(data[offset:offset+2], 'big')
                    offset += 2
                    sps_list.append(data[offset:offset+sps_len])
                    offset += sps_len

                # 解析PPS
                pps_count = data[offset]
                offset += 1
                pps_list = []
                for _ in range(pps_count):
                    if offset+2 > len(data):
                        break
                    pps_len = int.from_bytes(data[offset:offset+2], 'big')
                    offset += 2
                    pps_list.append(data[offset:offset+pps_len])
                    offset += pps_len

                return sps_list, pps_list
            except Exception as e:
                print(f"参数集解析异常: {str(e)}")
                return None, None

        # 获取并转换参数集
        sps_list, pps_list = parse_avcc(codec.extradata)
        if not sps_list or not pps_list:
            sps, pps = generate_default_sps_pps()
            sps_list = [sps]
            pps_list = [pps]
        # 转换为Annex B格式
        annexb_headers = b''.join(
            [b'\x00\x00\x00\x01' + sps for sps in sps_list] +
            [b'\x00\x00\x00\x01' + pps for pps in pps_list]
        )

        # 阶段4：智能参数集注入
        first_idr_pos = video_bytes.find(b'\x00\x00\x00\x01\x65')  # 查找IDR帧
        if first_idr_pos == -1:
            first_idr_pos = video_bytes.find(b'\x00\x00\x01\x65')
            if first_idr_pos != -1:
                first_idr_pos -= 1  # 修正3字节起始码偏移
        
        if first_idr_pos != -1:
            return video_bytes[:first_idr_pos] + annexb_headers + video_bytes[first_idr_pos:]
        else:
            return annexb_headers + video_bytes


class RTMPStreamPublisher:
    def __init__(self, rtmp_url=RTMP_URL, width=512, height=512, fps=25, channels=1, sample_rate=32000):
        self.rtmp_url = rtmp_url
        self.audio_processor = AudioProcessor(channels, sample_rate)
        self.video_processor = VideoProcessor(width, height, fps)        
        self.metadata = {
            'width': 512,
            'height': 512,
            'videocodecid': 7,  # H.264
            'audiocodecid': 10,  # AAC
            'framerate': 25,
            'audiosamplerate': 32000,
            'audiosamplesize': 16,
            'stereo': True
        }
        try:
            self.rtmp = RTMP(rtmp_url, live=True)
            self.rtmp.connect()
            self.stream = self.rtmp.create_stream()
            # self.rtmp.publish(stream)            
        except Exception as e:
                raise Exception(f"RTMP 连接或创建流失败: {str(e)}")

    def rtmp_push(self, audio_bytes, video_bytes):
        flv_header = b'FLV\x01\x05\x00\x00\x00\x09\x00\x00\x00\x00'
        print(type(self.rtmp))
        self.stream.write(flv_header)
        
        metadata_tag = self.build_flv_metadata_tag()
        print(metadata_tag)
        self.stream.write(metadata_tag)
        print("What is the problem of write")
        timestamp = 0
        frame_interval = 40  # 假设帧率为 25fps，每帧间隔 40ms
        print("开始推流")
        while True:
            if video_bytes:
                video_tag = self.build_flv_video_tag(video_bytes, timestamp)
                self.stream.write(video_tag)
            print("视频帧发送成功")
            # 发送音频帧
            if audio_bytes:
                audio_tag = self.build_flv_audio_tag(audio_bytes, timestamp)
                self.stream.write(audio_tag)
            print("音频帧发送成功")

            timestamp += frame_interval
            time.sleep(frame_interval / 1000)
    
    def build_flv_video_tag(self, video_bytes, timestamp):
        tag_type = 0x09
        data_size = len(video_bytes)
        timestamp_bytes = timestamp.to_bytes(3, byteorder='big') + bytes([(timestamp >> 24) & 0xFF])
        stream_id = 0

        header = bytes([tag_type]) + data_size.to_bytes(3, byteorder='big') + timestamp_bytes + bytes([0] * 3) + \
                stream_id.to_bytes(3, byteorder='big')
        return header + video_bytes

    def build_flv_audio_tag(self, audio_bytes, timestamp):
        tag_type = 0x08
        data_size = len(audio_bytes)
        timestamp_bytes = timestamp.to_bytes(3, byteorder='big') + bytes([(timestamp >> 24) & 0xFF])
        stream_id = 0

        header = bytes([tag_type]) + data_size.to_bytes(3, byteorder='big') + timestamp_bytes + bytes([0] * 3) + \
                stream_id.to_bytes(3, byteorder='big')
        return header + audio_bytes

    def amf_encode(self, value):
        if isinstance(value, str):
            return b'\x02' + len(value).to_bytes(2, byteorder='big') + value.encode('utf-8')
        elif isinstance(value, int):
            return b'\x00' + value.to_bytes(8, byteorder='big')
        elif isinstance(value, float):
            import struct
            return b'\x00' + struct.pack('>d', value)
        elif isinstance(value, bool):
            return b'\x01' + (b'\x01' if value else b'\x00')
        elif isinstance(value, dict):
            result = b'\x08'
            for key, val in value.items():
                result += len(key).to_bytes(2, byteorder='big') + key.encode('utf-8') + self.amf_encode(val)
            result += b'\x00\x00\x09'
            return result
        else:
            raise ValueError(f"Unsupported AMF encoding type: {type(value)}")

    def build_flv_metadata_tag(self):
        tag_type = 0x12
        data = self.amf_encode("onMetaData") + self.amf_encode(self.metadata)
        data_size = len(data)

        timestamp = 0
        timestamp_bytes = timestamp.to_bytes(3, byteorder='big') + bytes([(timestamp >> 24) & 0xFF])
        stream_id = 0
        
        header = bytes([tag_type]) + data_size.to_bytes(3, byteorder='big') + timestamp_bytes + bytes([0] * 3) + \
                stream_id.to_bytes(3, byteorder='big')
        return header + data

    def push_to_srs(self, audio_bytes, video_bytes):
        processed_audio = self.audio_processor.validate_and_add_header_audio(audio_bytes)
        processed_video = self.video_processor.validate_and_add_header_video(video_bytes)

        if processed_audio and processed_video:
            self.rtmp_push(processed_audio, processed_video)

def test_srs_connection():
    rtmp_url = "rtmp://47.111.96.21/live/test_stream"

    try:
        pusher = RTCStreamPublisher(rtmp_url)
        metadata_tag = pusher.build_flv_metadata_tag()
        assert len(metadata_tag) > 0, "元数据标签构建失败"
        print("RTMP 连接成功，元数据标签构建正常")
        return True

    except Exception as e:
        print(f"连接测试失败: {str(e)}")
        return False
    finally:
        if hasattr(pusher, 'rtmp') and pusher.rtmp:
            pusher.rtmp.close()

if __name__ == "__main__":
    connection_status = test_srs_connection()
    if connection_status:
        print("与 SRS 的连接测试通过")
    else:
        print("与 SRS 的连接测试失败")