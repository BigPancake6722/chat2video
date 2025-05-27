import av
import time
import librtmp
import subprocess
from io import BytesIO
from pydub import AudioSegment


RTMP_URL = "rtmp://47.111.96.21/live/livestream"
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


class RTMP_ffmpegPusher:
    def __init__(self, rtmp_url: str=RTMP_URL):
        self.rtmp_url = rtmp_url
        self.audio_processor = AudioProcessor()
        self.video_processor = VideoProcessor()

        self.ffmpeg_command = [
            'ffmpeg',
            '-f','s16le',
            '-ac', '1',
            '-ar', '32000',
            '-thread_queue_size', '64',
            '-i', '-',
            '-f', 'rawvideo',
            '-pixel_format', 'yuv420p',
            '-video_size', '512x512',
            '-framerate', '25',
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-f', 'flv',
            '-t', '60',
            self.rtmp_url
        ]


    def push_streams_to_rtmp(self, audio_stream_bytes, video_stream_bytes):
        """
        使用FFmpeg将原始音频流和视频流推送到RTMP服务器。

        :param audio_stream_bytes: 原始音频流的字节数据，格式为s16le，2声道，44100Hz采样率
        :param video_stream_bytes: 原始视频流的字节数据，格式为yuv420p，1920x1080分辨率，30fps帧率
        :param rtmp_url: RTMP服务器的地址，例如 'rtmp://your_rtmp_server_ip:port/app/stream_name'
        """
        
        try:
            process = subprocess.Popen(
                self.ffmpeg_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            # 发送音频数据
            process.stdin.write(audio_stream_bytes)
            # 发送视频数据
            process.stdin.write(video_stream_bytes)
            process.stdin.close()
            for line in process.stdout:
                print(line.decode('utf-8').strip())
            process.wait()
        except Exception as e:
            print(f"推流过程中出现错误: {e}")