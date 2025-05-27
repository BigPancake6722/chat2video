import av
import subprocess
from io import BytesIO

RTMP_URL = "rtmp://47.111.96.21/live/livestream"

class AudioProcessor:
    def __init__(self, channels=1, sample_rate=32000):
        self.channels = channels
        self.sample_rate = sample_rate
        
    def process_audio(self, audio_bytes):
        """处理原始音频为RTMP兼容的AAC格式（兼容PyAV 14.0.0+）"""
        # 如果已经是ADTS格式则直接返回
        if len(audio_bytes) >= 2 and audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xF0) == 0xF0:
            return audio_bytes
            
        return self.raw_to_aac(audio_bytes)
    
    def raw_to_aac(self, pcm_data):
        """将PCM转为AAC格式（PyAV 14.0.0+专用）"""
        output = BytesIO()
        
        # 创建输入容器（使用新的API）
        input_container = av.open(
            BytesIO(pcm_data),
            format='s16le',
            mode='r',
            options={
                'ar': str(self.sample_rate),
                'ac': str(self.channels),
            }
        )
        
        # 创建输出容器
        output_container = av.open(output, mode='w', format='adts')
        
        # 添加输出流（新API方式）
        output_stream = output_container.add_stream('aac')
        
        # 设置音频参数（PyAV 14.0.0+方式）
        output_stream.sample_rate = self.sample_rate
        output_stream.channels = self.channels
        output_stream.bit_rate = 128000
        
        # 转码流程
        try:
            for frame in input_container.decode(audio=0):
                frame.pts = None
                for packet in output_stream.encode(frame):
                    output_container.mux(packet)
                    
            # 刷新编码器
            for packet in output_stream.encode(None):
                output_container.mux(packet)
                
        finally:
            input_container.close()
            
        return output.getvalue()

class RTMPPusher:
    def __init__(self, rtmp_url=RTMP_URL):
        self.rtmp_url = rtmp_url
        self.audio_processor = AudioProcessor()
        
    def push_stream(self, audio_data, video_data=None):
        """推流方法（兼容PyAV 14.0.0+）"""
        aac_data = self.audio_processor.process_audio(audio_data)
        if not aac_data:
            raise ValueError("无效的音频数据")

        # 构建FFmpeg命令
        cmd = [
            'ffmpeg',
            '-y',
            '-f', 'aac',
            '-ar', '32000',
            '-ac', '1',
            '-i', 'pipe:0',
            '-c:a', 'copy',
            '-f', 'flv',
            self.rtmp_url
        ]
        
        # 处理视频流
        if video_data:
            cmd[6:6] = [
                '-f', 'h264',
                '-r', '25',
                '-i', 'pipe:1',
                '-c:v', 'copy'
            ]
        
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        
        try:
            # 写入音频数据
            proc.stdin.write(aac_data)
            
            # 写入视频数据（如果有）
            if video_data:
                proc.stdin.write(video_data)
                
            proc.stdin.close()
            
            # 打印日志
            for line in proc.stderr:
                print(line.decode().strip())
                
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"FFmpeg返回码: {proc.returncode}")
                
        except Exception as e:
            if proc.poll() is None:
                proc.kill()
            raise RuntimeError(f"推流失败: {str(e)}")

if __name__ == "__main__":
    # 测试数据生成
    def generate_silence(duration=1, sample_rate=32000):
        return b'\x00\x00' * int(duration * sample_rate)
    
    try:
        pusher = RTMPPusher()
        print("开始推流测试...")
        
        # 生成1秒静音
        audio_data = generate_silence()
        
        # 可选：生成测试视频帧
        video_data = b'\x00\x00\x00\x01\x65' + b'\x00'*(512*512*3//2)
        
        # pusher.push_stream(audio_data)  # 纯音频推流
        pusher.push_stream(audio_data, video_data)  # 音视频推流
        
        print("推流测试完成")
    except Exception as e:
        print(f"推流测试失败: {e}")