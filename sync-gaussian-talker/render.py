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

def render_set(model_path, name, iteration, viewpoint_stack, gaussians, pipeline,audio_dir, batch_size):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    inf_audio_dir = audio_dir
    
    makedirs(render_path, exist_ok=True)
    if name != 'custom':
        makedirs(gts_path, exist_ok=True)
    
    if name == "train" :
        process_until = 1000
        print(" -------------------------------------------------")
        print("        train set rendering  :   {} frames   ".format(process_until))
        print(" -------------------------------------------------")
    else:
        process_until = len(viewpoint_stack) 
        print(" -------------------------------------------------")
        print("        test set rendering  :   {}  frames  ".format(process_until))
        print(" -------------------------------------------------") 
    print("point nums:",gaussians._xyz.shape[0])
    # %%        
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

    codec = 'mp4v'
    fps = 25
    fourcc = cv2.VideoWriter_fourcc(*codec)
    videos = None
    smooth_coef = 1.0 if args.background_type in ["bg_w_torso", "scene"] else args.smooth_coef # 系数越小越靠向前一帧
    delay_wrapper = camera_delay_wrapper(coef=smooth_coef) # 用于平滑视角移动
    ksize = args.erode_size
    if ksize > 0 and ksize % 2 == 1:
        pooling = nn.MaxPool2d(kernel_size=ksize, stride=1, padding=int((ksize - 1)//2)) # erode kernel
        erode = lambda x, y:gaussian_blur_masked(x, pooling(y) + pooling(-y))
    # elif ksize < 0 and abs(ksize) % 2 == 1:
    #     ksize = abs(ksize)
    #     pooling = nn.MaxPool2d(kernel_size=ksize, stride=1, padding=int((ksize - 1)//2)) # erode kernel
    #     erode = lambda x:pooling(x)
    else:
        erode = lambda x, y:x
    with torch.no_grad():
        #render image and write into video
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            for idx in tqdm(range(iterations), desc="Rendering image",total = iterations):
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
                if videos is None:
                    videos = {}
                    videos["render"] = cv2.VideoWriter(f'{render_path}/renders.mp4', fourcc, fps, (W, H))
                    if args.visualize_attention:
                        videos["audio"] = cv2.VideoWriter(f'{render_path}/audio.mp4', fourcc, fps, (W, H))
                        videos['eye'] = cv2.VideoWriter(f'{render_path}/eye.mp4', fourcc, fps, (W, H))
                        videos['null'] = cv2.VideoWriter(f'{render_path}/null.mp4', fourcc, fps, (W, H))
                        videos['cam'] = cv2.VideoWriter(f'{render_path}/cam.mp4', fourcc, fps, (W, H))
                total_time += outputs["inference_time"]
                total_frame += B
                if args.use_gt_mask:
                    mask_pred = outputs["gt_segs_tensor"][:, seg_slicer, ...].any(dim=1, keepdim=True).float()
                else:
                    mask_pred = outputs["rend_alpha_tensor"]
                output_tensor = outputs["bg_tensor"] * (1 - mask_pred) + outputs["rendered_image_tensor"] * mask_pred
                output_tensor = erode(output_tensor, mask_pred)
                if args.visualize_attention:
                    for frame, attn_stacks in zip(tensor_to_image(output_tensor), tensor_to_image(outputs["attention_tensor"])):
                        videos["render"].write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        videos["audio"].write(cv2.cvtColor(attn_stacks[...,0], cv2.COLOR_BGR2RGB))
                        videos['eye'].write(cv2.cvtColor(attn_stacks[...,1], cv2.COLOR_BGR2RGB))
                        videos['cam'].write(cv2.cvtColor(attn_stacks[...,2], cv2.COLOR_BGR2RGB))
                        videos['null'].write(cv2.cvtColor(attn_stacks[...,3], cv2.COLOR_BGR2RGB))
                else:
                    for frame in tensor_to_image(output_tensor):
                        videos["render"].write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    print("total frame:",total_frame)
    print("Render FPS:",total_frame/total_time)
    for video in videos.values():
        video.release()  
    cmd = f'ffmpeg -loglevel quiet -y -i {render_path}/renders.mp4 -i {inf_audio_dir} -c:v copy -c:a aac {render_path}/{model_path.split("/")[-2]}_{name}_{iteration}iter_renders.mov'
    if os.system(cmd)==0:
        print("render done, check: ",render_path)
    else:
        print("render failed with cmd: ",cmd)
    if args.visualize_attention:
        if name != 'custom':
            cmd = f'ffmpeg -loglevel quiet -y -i {gts_path}/gt.mp4 -i {inf_audio_dir} -c:v copy -c:a aac {gts_path}/{model_path.split("/")[-2]}_{name}_{iteration}iter_gt.mov'
            os.system(cmd)
        cmd = f'ffmpeg -loglevel quiet -y -i {render_path}/renders.mp4 -i {inf_audio_dir} -c:v copy -c:a aac {render_path}/{model_path.split("/")[-2]}_{name}_{iteration}iter_renders.mov'
        os.system(cmd)
        cmd = f'ffmpeg -loglevel quiet -y -i {render_path}/audio.mp4 -i {inf_audio_dir} -c:v copy -c:a aac {render_path}/{model_path.split("/")[-2]}_{name}_{iteration}iter_audio.mov'
        os.system(cmd)
        cmd = f'ffmpeg -loglevel quiet -y -i {render_path}/eye.mp4 -i {inf_audio_dir} -c:v copy -c:a aac {render_path}/{model_path.split("/")[-2]}_{name}_{iteration}iter_eye.mov'
        os.system(cmd)
        cmd = f'ffmpeg -loglevel quiet -y -i {render_path}/null.mp4 -i {inf_audio_dir} -c:v copy -c:a aac {render_path}/{model_path.split("/")[-2]}_{name}_{iteration}iter_null.mov'
        os.system(cmd)
        cmd = f'ffmpeg -loglevel quiet -y -i {render_path}/cam.mp4 -i {inf_audio_dir} -c:v copy -c:a aac {render_path}/{model_path.split("/")[-2]}_{name}_{iteration}iter_cam.mov'
        os.system(cmd)
    gc.collect()
        
def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, args):
    skip_train, skip_test, batch_size= args.skip_train, args.skip_test, args.batch
    
    with torch.no_grad():
        data_dir = dataset.source_path
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, custom_aud=args.custom_aud, start_frame=time_to_frame(args.ss), smooth_win=args.smooth_win)
        gaussians.eval()
        
        if args.custom_aud != '':
            skip_train, skip_test = True, True
            audio_dir = args.custom_aud
            render_set(dataset.model_path, "custom", scene.loaded_iter, scene.getCustomCameras(), gaussians, pipeline, audio_dir, batch_size)
        
        if not skip_train:
            audio_dir = os.path.join(data_dir, "aud.wav")
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, audio_dir, batch_size)

        if not skip_test:
            audio_dir = os.path.join(data_dir, "aud.wav")
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, audio_dir, batch_size)

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

            
if __name__ == "__main__":
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
    parser.add_argument("--custom_aud", type=str, default='')
    parser.add_argument("--background_type", type=str, default='torso')
    parser.add_argument("--use_gt_mask", action="store_true")
    parser.add_argument("--smooth_coef", type=float, default=1.0) # 抖动较厉害可以减小该系数获得平滑性
    parser.add_argument("--erode_size", type=int, default=0) # 边缘较明显则可以适当加大该系数，当存在该系数时不会有smooth_win，必须为奇数
    parser.add_argument("--ss", type=str, default='00:00:00')
    parser.add_argument("--smooth_win", type=int, default=1)
    
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        from utils_talker.params_utils import merge_hparams, load_from_file
        config = load_from_file(args.configs)
        args = merge_hparams(args, config)    
    # Initialize system state (RNG)
    safe_state(args.quiet)
    args.only_infer = True
    print("hello hello, 听得到吗，聞こえいますが？", args)
    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args)