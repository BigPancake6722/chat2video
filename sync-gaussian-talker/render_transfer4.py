import cv2
import torch
import numpy as np
from tqdm import tqdm
from io import BytesIO
from os import makedirs
from scene import Scene
import concurrent.futures
from typing import Optional
from argparse import ArgumentParser
from gaussian_renderer import GaussianModel
from concurrent.futures import ThreadPoolExecutor
from utils_talker.general_utils import safe_state
from gaussian_renderer import render_from_batch_infer as render_from_batch
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams

def render_set(model_path, name, iteration, viewpoint_stack, gaussians, pipeline, 
              audio_dir, batch_size, args, audio_arr=None) -> Optional[BytesIO]:
    """生成VP8编码的视频流（WebRTC兼容格式）"""
    try:
        output = BytesIO()
        
        # WebM容器 + VP8编码
        with av.open(output, mode='w', format='webm') as container:
            stream = container.add_stream('libvpx', rate=25)  # VP8编码
            stream.width = 512
            stream.height = 512
            stream.pix_fmt = 'yuv420p'
            
            # WebRTC优化参数
            stream.options = {
                'quality': 'realtime',
                'cpu-used': '4',
                'lag-in-frames': '0',
                'error-resilient': '1',
                'crf': '30',
                'b:v': '1M'
            }

            # 原始渲染逻辑保持不变
            seg_slicer = [0, 1, 2] if args.part == "all" else [1, 3] if args.part == "face" else [0, 1]
            smooth_coef = 1.0 if args.background_type in ["bg_w_torso", "scene"] else args.smooth_coef
            delay_wrapper = camera_delay_wrapper(coef=smooth_coef)

            with torch.no_grad():
                for idx in tqdm(range((len(viewpoint_stack) // batch_size) + 1)):
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

                    # 转换并编码每一帧
                    for frame in tensor_to_image(output_tensor):
                        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        av_frame = av.VideoFrame.from_ndarray(bgr_frame, format='bgr24')
                        av_frame.pts = idx  # 关键帧间隔控制
                        
                        for packet in stream.encode(av_frame):
                            container.mux(packet)

            # 刷新编码器
            for packet in stream.encode(None):
                container.mux(packet)

            output.seek(0)
            return output

    except Exception as e:
        print(f"[VP8编码错误] {str(e)}")
        return None


def render_sets(dataset: ModelParams, hyperparam, iteration: int, 
               pipeline: PipelineParams, args, audio_arr=None) -> Optional[bytes]:
    """协调渲染流程并返回VP8视频流"""
    try:
        # 初始化模型（参数从main传入）
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False,
                     custom_aud=audio_arr, start_frame=time_to_frame(args.ss),
                     smooth_win=args.smooth_win)
        gaussians.eval()

        # 仅处理自定义音频的情况
        if args.custom_aud:
            video_bytesio = render_set(
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
            return video_bytesio.getvalue() if video_bytesio else None
            
        return None

    except Exception as e:
        print(f"[渲染流程错误] {str(e)}")
        return None


def main(custom_cmd_list, audio_arr) -> Optional[bytes]:
    """主入口函数，返回VP8视频流bytes"""
    # 参数解析（保持原有逻辑）
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
    
    # 加载配置文件（保持原有逻辑）
    if args.configs:
        from utils_talker.params_utils import merge_hparams, load_from_file
        config = load_from_file(args.configs)
        args = merge_hparams(args, config)    
    
    # 初始化系统状态
    safe_state(args.quiet)
    args.only_infer = True
    
    # 调用渲染流程并返回VP8视频流
    return render_sets(
        model.extract(args), 
        hyperparam.extract(args), 
        args.iteration, 
        pipeline.extract(args), 
        args, 
        audio_arr=audio_arr
    )