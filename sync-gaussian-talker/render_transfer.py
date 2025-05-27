#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_from_batch_infer as render_from_batch
from utils_talker.render_utils import interpolate_viewpoint
import torchvision
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils_talker.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from utils_talker.webRTC_transfer import GaussianVideoStream
import concurrent.futures
import gc
import ffmpeg
from torch import nn
from io import BytesIO
import av
from typing import Generator, Optional

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

def gaussian_blur_masked(image, edge, kernel_size=15, sigma=5):    
    from torchvision.transforms.functional import gaussian_blur
    # 创建高斯核
    channels = image.size(1)
    kernel = torch.ones(channels, 1, kernel_size, kernel_size)
    kernel = kernel.to(image.device)
    # 应用高斯模糊
    blurred = gaussian_blur(image, kernel_size, sigma)
    # 使用mask混合原始图像和模糊图像
    result = image * (1 - edge) + blurred * edge
    return result

def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def time_to_frame(time, fps=25):
    try:
        hour, minute, second = [int(x) for x in time.split(":")]
    except:
        raise ValueError(f"Invalid time format of {time}. Expected 'HH:MM:SS'")
    seconds = hour * 3600 + minute * 60 + second
    return int(seconds * fps)

def transcode_vp8_bytes_to_h264_bytes(vp8_bytes):
    try:
        # 输入流配置
        input_stream = ffmpeg.input('pipe:', format='webm')
        # 输出流配置，使用 H264 编码，输出格式改为 FLV
        output_stream = ffmpeg.output(input_stream, 'pipe:', format='flv', vcodec='libx264')
        # 运行 FFmpeg 命令并捕获输出
        process = ffmpeg.run_async(output_stream, pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        out, err = process.communicate(input=vp8_bytes)
        if process.returncode != 0:
            raise Exception(f"FFmpeg 转码出错: {err.decode()}")
        return out
    except Exception as e:
        print(f"转码过程中出现错误: {e}")
        return None

def render_set(model_path, name, iteration, viewpoint_stack, gaussians, pipeline, 
              audio_dir, batch_size, args, audio_arr=None) -> Optional[BytesIO]:
    """渲染视频集合并返回内存中的H264视频流"""
    try:
        output = BytesIO()
        
        # 使用裸 H.264 流格式（Annex B）
        with av.open(output, mode='w', format='h264') as container:
            # 配置 H.264 编码器
            stream = container.add_stream('libx264', rate=25)
            stream.width = 512
            stream.height = 512
            stream.pix_fmt = 'yuv420p'
            
            # 关键编码参数设置
            stream.options = {
                'profile': 'baseline',
                'preset': 'fast',
                'tune': 'zerolatency',
                'x264-params': 'keyint=25:min-keyint=25:annexb=1'
            }

            # 强制生成全局头信息
            stream.codec_context.flags |= 0x00400000  # AV_CODEC_FLAG_GLOBAL_HEADER
            
            # 存储原始 NAL 单元
            h264_nalus = bytearray()

            # 渲染设置（保持不变）
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

                    # 处理每一帧
                    for frame in tensor_to_image(output_tensor):
                        # 转换颜色空间到 YUV420P
                        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
                        
                        # 创建视频帧并编码
                        av_frame = av.VideoFrame.from_ndarray(yuv_frame, format='yuv420p')
                        av_frame.pts = None  # 清空时间戳
                        
                        for packet in stream.encode(av_frame):
                            # 转换长度前缀为 Annex B 格式
                            data = bytes(packet)
                            while data:
                                nalu_size = int.from_bytes(data[:4], byteorder='big')
                                nalu = data[4:4+nalu_size]
                                h264_nalus += b'\x00\x00\x00\x01' + nalu
                                data = data[4+nalu_size:]

            # 刷新编码器获取剩余数据
            for packet in stream.encode(None):
                data = bytes(packet)
                while data:
                    nalu_size = int.from_bytes(data[:4], byteorder='big')
                    nalu = data[4:4+nalu_size]
                    h264_nalus += b'\x00\x00\x00\x01' + nalu
                    data = data[4+nalu_size:]

            # 插入 SPS/PPS 到流起始位置
            if stream.codec_context.extradata:
                sps_pps = stream.codec_context.extradata.replace(b'\x00\x00\x00\x01', b'')
                h264_nalus = sps_pps + h264_nalus
            
            # 写入最终数据
            output.write(h264_nalus)
            output.seek(0)
            return output

    except Exception as e:
        print(f"渲染异常: {str(e)}")
        return None

def render_sets(dataset: ModelParams, hyperparam, iteration: int, 
               pipeline: PipelineParams, args, audio_arr=None) -> Optional[BytesIO]:
    """渲染视频集合并返回内存中的视频流"""
    try:
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False,
                     custom_aud=audio_arr, start_frame=time_to_frame(args.ss),
                     smooth_win=args.smooth_win)
        gaussians.eval()

        if args.custom_aud:
            return render_set(
                dataset.model_path,
                "custom",
                scene.loaded_iter,
                scene.getCustomCameras(),
                gaussians,
                pipeline,
                args.custom_aud,
                args.batch,
                args,
                audio_arr=audio_arr
            )
        return None

    except Exception as e:
        print(f"Error in render_sets: {str(e)}")
        return None

def tensor_to_image(tensor, normalize=True):
    if torch.is_tensor(tensor):
        image = tensor.detach().cpu().numpy().squeeze()
    else:
        image = tensor
        
    if normalize:
        image = 255 * image
        image = image.clip(0, 255).astype(np.uint8)

    if len(image.shape) == 3:
        image = image.transpose(1, 2, 0)
    elif len(image.shape) == 4:
        image = image.transpose(0, 2, 3, 1)
    return image
        
def tensor_to_videoframe(tensor):
    """将张量直接转换为VideoFrame"""
    frame = (255 * np.clip(tensor.squeeze().cpu().numpy().transpose(1, 2, 0), 0, 1)).astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return VideoFrame.from_ndarray(frame, format="bgr24") 

            
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
    parser.add_argument("--smooth_coef", type=float, default=1.0) # 抖动较厉害可以减小该系数获得平滑性
    parser.add_argument("--erode_size", type=int, default=0) # 边缘较明显则可以适当加大该系数，当存在该系数时不会有smooth_win，必须为奇数
    parser.add_argument("--ss", type=str, default='00:00:00')
    parser.add_argument("--smooth_win", type=int, default=1)
    
    args = get_combined_args(parser, custom_cmd_list)
    print("Rendering ", args.model_path)
    if args.configs:
        from utils_talker.params_utils import merge_hparams, load_from_file
        config = load_from_file(args.configs)
        args = merge_hparams(args, config)    
    # Initialize system state (RNG)
    safe_state(args.quiet)
    args.only_infer = True
    
    # 返回视频流而不是保存文件
    video_stream = render_sets(model.extract(args), hyperparam.extract(args), 
                               args.iteration, pipeline.extract(args), args, 
                               audio_arr=audio_arr).getvalue()
    return video_stream