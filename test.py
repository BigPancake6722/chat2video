import numpy as np
import ffmpeg
from io import BytesIO

def ndarray_audio_to_mp4(
    video_frames: np.ndarray,
    audio_pcm: bytes,
    output_path: str,
    fps: int = 25,
    audio_sample_rate: int = 32000,
    audio_channels: int = 1
):
    # 验证输入数据
    if not isinstance(video_frames, np.ndarray) or video_frames.dtype != np.uint8:
        raise ValueError("video_frames必须是uint8类型的NDArray")
    if video_frames.ndim != 4 or video_frames.shape[-1] != 3:
        raise ValueError("video_frames形状必须为(n, height, width, 3)（RGB格式）")
    if not isinstance(audio_pcm, bytes):
        raise ValueError("audio_pcm必须是bytes类型")

    # 计算原始时长
    n_frames, height, width, _ = video_frames.shape
    video_duration = n_frames / fps
    
    bytes_per_sample = 2 * audio_channels  # 16位PCM每个样本占2字节，乘以声道数
    audio_samples = len(audio_pcm) // bytes_per_sample
    audio_duration = audio_samples / audio_sample_rate

    # 时长差异计算
    duration_diff = abs(video_duration - audio_duration)
    
    if duration_diff <= 0.5:  # 允许0.1秒差异
        print(f"时长差{duration_diff:.2f}s在允许范围内，进行末尾截断对齐")
        
        # 确定对齐目标时长
        target_duration = min(video_duration, audio_duration)
        
        # 调整视频帧
        if video_duration > target_duration:
            keep_frames = int(target_duration * fps)
            video_frames = video_frames[:keep_frames]
            n_frames = keep_frames
            print(f"截断视频至{keep_frames}帧，时长{target_duration:.2f}s")
        
        # 调整音频数据
        if audio_duration > target_duration:
            required_samples = int(target_duration * audio_sample_rate)
            required_bytes = required_samples * bytes_per_sample
            audio_pcm = audio_pcm[:required_bytes]
            print(f"截断音频至{required_samples}样本，时长{target_duration:.2f}s")
    else:
        raise ValueError(f"音视频时长差({duration_diff:.2f}s)超过0.1秒！")

    # 重新计算最终时长
    final_video_duration = n_frames / fps
    final_audio_samples = len(audio_pcm) // bytes_per_sample
    final_audio_duration = final_audio_samples / audio_sample_rate
    
    print(f"最终对齐时长：视频 {final_video_duration:.2f}s，音频 {final_audio_duration:.2f}s")

    # 创建视频流
    video_stream = ffmpeg.input(
        'pipe:', 
        format='rawvideo', 
        pix_fmt='rgb24', 
        s=f"{width}x{height}", 
        r=fps
    )

    # 创建音频流
    audio_stream = ffmpeg.input(
        BytesIO(audio_pcm),
        format='s16le', 
        ac=audio_channels,
        ar=audio_sample_rate,
        sample_fmt='s16le'
    )

    # 合成输出
    (ffmpeg
     .concat(video_stream, audio_stream, v=1, a=1)
     .output(
         output_path,
         vcodec='libx264',
         crf=23,
         acodec='aac',
         strict='experimental'
     )
     .overwrite_output()
     .run(
         input=video_frames.tobytes(),
         capture_stdout=True,
         capture_stderr=True
     ))