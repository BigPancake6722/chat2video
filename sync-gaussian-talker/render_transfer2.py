import os
import cv2
import torch
import numpy as np
import ffmpeg
from io import BytesIO
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from argparse import ArgumentParser
from scene import Scene
from gaussian_renderer import GaussianModel
from gaussian_renderer import render_from_batch_infer as render_from_batch
from utils_talker.render_utils import interpolate_viewpoint
from utils_talker.general_utils import safe_state
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams

def auto_start(func):
    def wrapper(*args, **kwargs):
        gen = func(*args, **kwargs)
        next(gen)
        return gen
    return wrapper

@auto_start
def camera_delay_wrapper(coef=0.0):
    assert 0 <= coef <= 1
    cam_pre = None
    while 1:
        cam_post = yield cam_pre
        if cam_pre is None:
            cam_pre = cam_post
        cam_pre = interpolate_viewpoint(cam_post, cam_pre, coef) if coef < 1 else cam_post

def time_to_frame(time, fps=25):
    try:
        hour, minute, second = [int(x) for x in time.split(":")]
    except:
        raise ValueError(f"Invalid time format of {time}. Expected 'HH:MM:SS'")
    seconds = hour * 3600 + minute * 60 + second
    return int(seconds * fps)

def tensor_to_frames(tensor):
    """Convert tensor to video frames in RGB24 format"""
    if torch.is_tensor(tensor):
        frames = tensor.detach().cpu().numpy()
    else:
        frames = np.array(tensor)
    
    # Normalize to 0-255 and convert to uint8
    frames = (255 * np.clip(frames, 0, 1)).astype(np.uint8)
    
    # Handle batch dimension
    if len(frames.shape) == 4:  # Batch of frames
        for i in range(frames.shape[0]):
            frame = frames[i].transpose(1, 2, 0)  # CHW to HWC
            yield frame
    else:  # Single frame
        frame = frames.transpose(1, 2, 0)
        yield frame

def validate_frame(frame: np.ndarray) -> np.ndarray:
    """确保帧数据符合3通道RGB格式"""
    if len(frame.shape) == 2:  # 灰度图转RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:  # 带Alpha通道
        frame = frame[:, :, :3]
    return frame

def tensor_to_frames(tensor: torch.Tensor) -> Generator[np.ndarray, None, None]:
    """安全转换张量到帧序列"""
    # 确保输入是4D张量 [batch, channels, height, width]
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    
    frames = tensor.detach().cpu().numpy()
    frames = (255 * np.clip(frames, 0, 1)).astype(np.uint8)
    
    for i in range(frames.shape[0]):
        frame = frames[i].transpose(1, 2, 0)  # CHW -> HWC
        yield validate_frame(frame)

def render_set(model_path, name, iteration, viewpoint_stack, gaussians, pipeline, 
              audio_dir, batch_size, args, audio_arr=None) -> Optional[BytesIO]:
    """渲染视频集合并返回内存中的视频流"""
    try:
        stream_buffer = BytesIO()
        
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='512x512', framerate=25)
            .output('pipe:', format='flv', vcodec='libx264', pix_fmt='yuv420p',
                   preset='fast', tune='zerolatency', r=25, g=50)
            .run_async(pipe_stdin=True, pipe_stdout=True, overwrite_output=True)
        )

        seg_slicer = [0, 1, 2] if args.part == "all" else [1, 3] if args.part == "face" else [0, 1]
        smooth_coef = 1.0 if args.background_type in ["bg_w_torso", "scene"] else args.smooth_coef
        delay_wrapper = camera_delay_wrapper(coef=smooth_coef)

        with torch.no_grad():
            for idx in range((len(viewpoint_stack) // batch_size) + 1):
                viewpoint_batch = viewpoint_stack[idx * batch_size:(idx + 1) * batch_size]
                
                if viewpoint_stack.lazy_load:
                    with ThreadPoolExecutor(max_workers=batch_size) as executor:
                        futures = [executor.submit(viewpoint_cam) for viewpoint_cam in viewpoint_batch]
                        viewpoint_cams = [delay_wrapper.send(future.result()) for future in futures]
                else:
                    viewpoint_cams = [delay_wrapper.send(cam) for cam in viewpoint_batch]

                outputs = render_from_batch(viewpoint_cams, gaussians, pipeline.debug,
                                          background=args.background_type,
                                          visualize_attention=args.visualize_attention,
                                          feature_inputs=["aud", "eye", "cam", "uid"])

                mask_pred = outputs["gt_segs_tensor"][:, seg_slicer, ...].any(dim=1, keepdim=True).float() if args.use_gt_mask else outputs["rend_alpha_tensor"]
                output_tensor = outputs["bg_tensor"] * (1 - mask_pred) + outputs["rendered_image_tensor"] * mask_pred

                # 安全处理每帧数据
                for frame in tensor_to_frames(output_tensor):
                    try:
                        process.stdin.write(frame.tobytes())
                        while chunk := process.stdout.read(4096):
                            stream_buffer.write(chunk)
                    except BrokenPipeError as e:
                        raise RuntimeError(f"FFmpeg管道错误: {e}") from e

        process.stdin.close()
        while chunk := process.stdout.read(4096):
            stream_buffer.write(chunk)
        
        if process.wait() != 0:
            raise RuntimeError("视频编码失败")

        stream_buffer.seek(0)
        return stream_buffer

    except Exception as e:
        print(f"渲染错误: {str(e)}")
        if 'process' in locals():
            process.terminate()
        return None

def render_sets(dataset: ModelParams, hyperparam, iteration: int, 
               pipeline: PipelineParams, args, audio_arr=None) -> Optional[BytesIO]:
    """主渲染入口"""
    try:
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False,
                     custom_aud=audio_arr, start_frame=time_to_frame(args.ss),
                     smooth_win=args.smooth_win)
        gaussians.eval()

        if args.custom_aud:
            return render_set(
                dataset.model_path, "custom", scene.loaded_iter,
                scene.getCustomCameras(), gaussians, pipeline,
                args.custom_aud, args.batch, args, audio_arr
            )
        return None

    except Exception as e:
        print(f"渲染集错误: {str(e)}")
        return None

def main(custom_cmd_list, audio_arr):
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str, default="/root/chat2video/sync-gaussian-talker/arguments/custom.py")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--custom_aud", type=str, default='1')
    parser.add_argument("--background_type", type=str, default='torso')
    parser.add_argument("--use_gt_mask", action="store_true")
    parser.add_argument("--smooth_coef", type=float, default=1.0)
    parser.add_argument("--erode_size", type=int, default=0)
    parser.add_argument("--ss", type=str, default='00:00:00')
    parser.add_argument("--smooth_win", type=int, default=1)
    
    args = get_combined_args(parser, custom_cmd_list)
    print("Rendering ", args.model_path)
    
    if args.configs:
        from utils_talker.params_utils import merge_hparams, load_from_file
        config = load_from_file(args.configs)
        args = merge_hparams(args, config)
    
    safe_state(args.quiet)
    args.only_infer = True
    
    # Get the video stream in memory
    video_stream = render_sets(model.extract(args), hyperparam.extract(args), 
                             args.iteration, pipeline.extract(args), args, 
                             audio_arr=audio_arr)
    print(type(video_stream))
    return video_stream.getvalue() if video_stream else None