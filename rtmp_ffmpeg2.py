import av
import subprocess
from io import BytesIO
import time

RTMP_URL = "rtmp://47.111.96.21/live/livestream"

class AudioProcessor:
    def __init__(self, channels=1, sample_rate=32000):
        self.channels = channels
        self.sample_rate = sample_rate
        
    def process_audio(self, audio_bytes):
        """处理AAC音频流（确保符合RTMP要求）"""
        # 检查是否是有效的ADTS AAC流
        if self.is_valid_adts(audio_bytes):
            return audio_bytes
        raise ValueError("无效的AAC音频数据")

    def is_valid_adts(self, data):
        """检查是否为有效的ADTS AAC格式"""
        return len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xF0) == 0xF0

class RTMPPusher:
    def __init__(self, rtmp_url=RTMP_URL):
        self.rtmp_url = rtmp_url
        self.audio_processor = AudioProcessor()
        
    def push_stream(self, aac_data, video_data=None):
        """
        推送AAC音频和H264视频流到RTMP服务器
        Args:
            aac_data: 已经编码好的AAC音频数据
            video_data: H264编码的视频数据
        """
        # 验证音频数据
        aac_data = self.audio_processor.process_audio(aac_data)
        
        # 构建FFmpeg命令
        cmd = [
            'ffmpeg',
            '-y',
            '-f', 'aac',        # 输入音频格式
            '-ar', '32000',     # 音频采样率
            '-ac', '1',         # 音频通道数
            '-i', 'pipe:0',     # 音频输入管道
            '-c:a', 'copy',     # 音频直接复制不转码
            '-f', 'flv',        # 输出格式
            self.rtmp_url
        ]
        
        # 如果有视频数据，添加视频参数
        if video_data:
            cmd[6:6] = [
                '-f', 'h264',   # 输入视频格式
                '-r', '25',     # 视频帧率
                '-i', 'pipe:1', # 视频输入管道
                '-c:v', 'copy'  # 视频直接复制不转码
            ]
        
        # 启动FFmpeg进程
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

def test_rtmp_connection(rtmp_url, timeout=5):
    """
    测试RTMP服务器连接
    Args:
        rtmp_url: RTMP服务器地址
        timeout: 超时时间(秒)
    Returns:
        bool: 是否连接成功
    """
    test_cmd = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'error',
        '-t', str(timeout),
        '-i', rtmp_url,
        '-f', 'null', '-'
    ]
    
    try:
        start_time = time.time()
        proc = subprocess.Popen(
            test_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        while time.time() - start_time < timeout:
            if proc.poll() is not None:
                break
            time.sleep(0.1)
        
        if proc.poll() is None:
            print("what the funk")
            proc.kill()
            return False
        return proc.returncode == 0
    except Exception as e:
        print(f"连接测试异常: {e}")
        return False

if __name__ == "__main__":
    # 测试RTMP连接
    print("测试RTMP服务器连接...")
    if test_rtmp_connection(RTMP_URL):
        print("RTMP服务器连接成功")
        
        # 测试推流
        pusher = RTMPPusher()
        print("开始推流测试...")
        
        # 生成测试AAC数据 (静音1秒)
        aac_header = bytes.fromhex('FFF15080027FFC')
        aac_data = aac_header + bytes(1024)
        
        # 生成测试H264数据 (空帧)
        h264_data = bytes.fromhex('0000000165000000010000000161000100000001')
        
        try:
            pusher.push_stream(aac_data, h264_data)
            print("推流测试成功")
        except Exception as e:
            print(f"推流测试失败: {e}")
    else:
        print("RTMP服务器连接失败")