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
import concurrent.futures
import gc
from torch import nn
from io import BytesIO

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

def render_set(model_path, name, iteration, viewpoint_stack, gaussians, pipeline, audio_dir, batch_size, args, audio_arr):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    inf_audio_dir = audio_dir
    
    makedirs(render_path, exist_ok=True)
    if name != 'custom':
        makedirs(gts_path, exist_ok=True)
    
    if name == "train":
        process_until = 1000
        print(" -------------------------------------------------")
        print("        train set rendering  :   {} frames   ".format(process_until))
        print(" -------------------------------------------------")
    else:
        process_until = len(viewpoint_stack) 
        print(" -------------------------------------------------")
        print("        test set rendering  :   {}  frames  ".format(process_until))
        print(" -------------------------------------------------") 
    print("point nums:", gaussians._xyz.shape[0])
    
    iterations = process_until // batch_size
    if process_until % batch_size != 0:
        iterations += 1
    total_time = 0
    total_frame = 0
    
    seg_slicer = slice(None, None)
    if args.part == "head":
        seg_slicer = [0, 1]
    elif args.part == "face":
        seg_slicer = [1, 3]
    elif args.part == "all":
        seg_slicer = [0, 1, 2]

    # 使用 PyAV 创建 FLV 容器（RTMP 兼容）
    import av
    from io import BytesIO

    output_buffer = BytesIO()
    container = av.open(output_buffer, mode='w', format='flv')
    stream = container.add_stream('libx264', rate=25)
    stream.pix_fmt = 'yuv420p'
    stream.options = {
        'preset': 'fast',
        'crf': '23',
        'tune': 'zerolatency',
    }

    smooth_coef = 1.0 if args.background_type in ["bg_w_torso", "scene"] else args.smooth_coef
    delay_wrapper = camera_delay_wrapper(coef=smooth_coef)
    ksize = args.erode_size
    if ksize > 0 and ksize % 2 == 1:
        pooling = nn.MaxPool2d(kernel_size=ksize, stride=1, padding=int((ksize - 1)//2))
        erode = lambda x, y: gaussian_blur_masked(x, pooling(y) + pooling(-y))
    else:
        erode = lambda x, y: x

    with torch.no_grad():
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            for idx in tqdm(range(iterations), desc="Rendering image", total=iterations):
                viewpoint_batch = viewpoint_stack[idx * batch_size:(idx + 1) * batch_size]
                if viewpoint_stack.lazy_load:
                    futures = [executor.submit(viewpoint_cam) for viewpoint_cam in viewpoint_batch]
                viewpoint_cams = [delay_wrapper.send(future.result()) for future in futures]
                outputs = render_from_batch(viewpoint_cams, gaussians, pipeline.debug, 
                                    background=args.background_type, 
                                    visualize_attention=args.visualize_attention,
                                    feature_inputs=["aud", "eye", "cam", "uid"],
                                    )
                B, C, H, W = outputs["rendered_image_tensor"].shape
                if stream.width is None:
                    stream.width = W
                    stream.height = H
                
                total_time += outputs["inference_time"]
                total_frame += B
                
                if args.use_gt_mask:
                    mask_pred = outputs["gt_segs_tensor"][:, seg_slicer, ...].any(dim=1, keepdim=True).float()
                else:
                    mask_pred = outputs["rend_alpha_tensor"]
                
                output_tensor = outputs["bg_tensor"] * (1 - mask_pred) + outputs["rendered_image_tensor"] * mask_pred
                output_tensor = erode(output_tensor, mask_pred)
                
                # 将帧写入 PyAV 容器（转换为 YUV420P）
                for frame in tensor_to_image(output_tensor):
                    av_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
                    av_frame = av_frame.reformat(format='yuv420p')  # 转换为 YUV420P
                    for packet in stream.encode(av_frame):
                        container.mux(packet)

    # 刷新流
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    # 获取 FLV 封装的 H.264 视频流（YUV420P）
    video_bytes = output_buffer.getvalue()
    output_buffer.close()

    print("total frame:", total_frame)
    print("Render FPS:", total_frame / total_time)

    return video_bytes  # 返回 FLV 封装的 H.264 视频流（适用于 RTMP 推流）
        
def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, args, audio_arr=None):
    skip_train, skip_test, batch_size= args.skip_train, args.skip_test, args.batch
    print("render_sets", audio_arr.shape)
    with torch.no_grad():
        data_dir = dataset.source_path
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, custom_aud=audio_arr, start_frame=time_to_frame(args.ss), smooth_win=args.smooth_win)
        gaussians.eval()
        
        if args.custom_aud != '':
            skip_train, skip_test = True, True
            audio_dir = args.custom_aud
            return render_set(dataset.model_path, "custom", scene.loaded_iter, scene.getCustomCameras(), gaussians, pipeline, audio_dir, batch_size, args, audio_arr=audio_arr)
        
        if not skip_train:
            audio_dir = os.path.join(data_dir, "aud.wav")
            return render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, audio_dir, batch_size, args, audio_arr=None)

        if not skip_test:
            audio_dir = os.path.join(data_dir, "aud.wav")
            return render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, audio_dir, batch_size, args, audio_arr=None)

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

            
def main(custom_cmd_list, audio_arr):
    print("main", audio_arr.shape)
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
    # Initialize system state (RNG)
    safe_state(args.quiet)
    args.only_infer = True
    print(args)

    # 调用 render_sets 并返回视频流
    video_bytes = render_sets(
        model.extract(args),
        hyperparam.extract(args),
        args.iteration,
        pipeline.extract(args),
        args,
        audio_arr=audio_arr
    )
    return video_bytes  # 返回 H.264 编码的 bytes