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
import numpy as np
import random
import os, sys
import torch
from torch.nn import functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import network_gui, render_from_batch_train as render_from_batch
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
import lpips
import copy
import pdb

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
from utils.loss_utils import VGGPerceptualLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg_perceptual_loss = VGGPerceptualLoss().to(device)
    
def scene_reconstruction(opt:OptimizationParams, pipe:PipelineParams, testing_iterations, 
                         checkpoint_iterations, first_iter, debug_from,
                         gaussians:GaussianModel, scene:Scene, stage, tb_writer, train_iter, timer):
    # if stage == "fine" and first_iter == 0:
    #     gaussians.mlp2cpu()
    gaussians.training_setup(opt)
        
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    checkpoint_iterations = checkpoint_iterations + [final_iter]
    first_iter += 1
    train_cams = scene.getTrainCameras()
    
    if not viewpoint_stack:
        viewpoint_stack = [cam for cam in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)
    
    batch_size = opt.batch_size
    if stage == 'coarse':batch_size=1
    if args.part == "head":
        seg_slicer = [0, 1]
    elif args.part == "face":
        seg_slicer = [3]
    elif args.part == "all":
        seg_slicer = [0, 1, 2]
    
    print("data loading done")
    progress_bar = tqdm(range(final_iter), desc=f"Training {stage} progress", initial=first_iter)
    for iteration in range(first_iter, final_iter+1):
        
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        idx = 0
        viewpoint_cams = []

        while idx < batch_size :    
            viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))
            if not viewpoint_stack :
                viewpoint_stack =  temp_list.copy()
            viewpoint_cams.append(viewpoint_cam() if scene.lazy_load else viewpoint_cam)
            idx +=1
        if len(viewpoint_cams) == 0:
            continue

        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        reg_surface = stage=="coarse" and iteration > opt.coarse_surface_reg_from_iter
        reg_attn = stage == 'fine' and iteration > opt.attn_reg_from_iter # 最终阶段微调注意力区域
        background_type = args.background_type
        outputs = render_from_batch(viewpoint_cams, gaussians, pipe.debug, background_type, stage=stage, 
                                    canonical_tri_plane_factor_list = opt.canonical_tri_plane_factor_list,
                                    visualize_attention=True if reg_attn else False,
                                    )
        generated_mask = outputs["rend_alpha_tensor"]
        gt_mask = outputs["gt_segs_tensor"][:, seg_slicer, ...].any(dim=1, keepdim=True).float() # head/ neck/ torso/ face(no hair)
        if args.part == "face": # 此处膨胀面部以适配部分脖子
            head_mask = outputs["gt_segs_tensor"][:, [0, 1], ...].any(dim=1, keepdim=True).float()
            gt_mask = F.max_pool2d(gt_mask, kernel_size=13, stride=1, padding=6) * head_mask # dilate face mask fit face
            neck_intersect = F.max_pool2d(gt_mask, kernel_size=27, stride=1, padding=13) * (outputs["gt_segs_tensor"][:, [1], ...])
            gt_mask = ((neck_intersect + gt_mask) > 0).float()
        
        image_tensor = outputs["rendered_image_tensor"]
        gt_image_tensor = outputs["gt_tensor"]
        visibility_filter = outputs["visibility_filter_tensor"]
        radii = outputs["radii"]
        viewspace_point_tensor_list = outputs["viewspace_point_tensor_list"] 
        viewspace_point_tensor = viewspace_point_tensor_list[0]
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:], gt_mask)

        psnr_ = 1 - psnr(image_tensor, gt_image_tensor).mean().double()
        perceptual_loss = vgg_perceptual_loss(image_tensor, gt_image_tensor[:,:3,:,:])
    
        ssim_loss = 1 - ssim(image_tensor, gt_image_tensor)
        
        loss = 0.8 * Ll1 + 0.01* perceptual_loss + 0.2 * ssim_loss
            
        dist_loss, normal_loss = None, None
        if reg_surface:
            rend_normal = outputs["rend_normal_tensor"]
            rend_dist = outputs["rend_dist_tensor"]
            surf_normal = outputs["surf_normal_tensor"]
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=1))
            dist_loss = opt.lambda_dist * rend_dist.mean()
            normal_loss = opt.lambda_normal * normal_error.mean()
            loss += dist_loss+normal_loss
        
        lip_l1_loss, mask_loss = None, None
        if stage == "fine" and opt.lip_fine_tuning:
            lip_l1_loss = l1_loss(outputs["rendered_lips_tensor"], outputs["gt_lips_tensor"])*0.4
            loss+=lip_l1_loss

        if opt.depth_fine_tuning:
            mask_loss = F.binary_cross_entropy(generated_mask, gt_mask) * 0.4
            loss += mask_loss
            
        reg_loss_aud, reg_loss_exp, reg_loss_cam = None, None, None
        
        if reg_attn: 
            frame_attention = outputs["attention_tensor"][:, [3], ...] # b,1,h,w
            # 正则语音的attention
            audio_attention = outputs["attention_tensor"][:, [0], ...] # b,1,h,w
            outier_mask = (1 - outputs["gt_segs_tensor"][:, [1, 3], ...].any(dim=1, keepdim=True).float()) * gt_mask # !face or neck
            reg_loss_aud = ((audio_attention - frame_attention) * outier_mask).mean() * 0.4
            loss += reg_loss_aud
            
            # 正则表情的attention
            exp_attention = outputs["attention_tensor"][:, [1], ...] # b,1,h,w
            outier_mask = (1 - outputs["gt_segs_tensor"][:, [3], ...].any(dim=1, keepdim=True).float()) * gt_mask # !face
            reg_loss_exp = (exp_attention * outier_mask).mean() * 0.4
            loss += reg_loss_exp
            
            # 正则镜头和场景的attention
            # cam_attention = outputs["attention_tensor"][:, [2, 3], ...] # b,1,h,w
            # outier_mask = outputs["gt_segs_tensor"][:, [3], ...].any(dim=1, keepdim=True).float() # face
            # reg_loss_cam = (cam_attention * outier_mask).mean() * 0.4
            # loss += reg_loss_cam
            
        losses = {
            "l1_loss": Ll1.item(),
            "lip_l1_loss": lip_l1_loss.item() if lip_l1_loss else None,
            "mask_loss": mask_loss.item() if mask_loss else None,
            "total_loss": loss.item(),
            "regular_attn_aud": reg_loss_aud.item() if reg_loss_aud else None,
            "regular_attn_exp": reg_loss_exp.item() if reg_loss_exp else None,
            "regular_attn_cam": reg_loss_cam.item() if reg_loss_cam else None,
            "psnr":psnr_.item(), 
            "perceptual_loss":perceptual_loss.item(),
            "ssim_loss":ssim_loss.item(),
            "normal_loss":normal_loss.item() if normal_loss else None,
            "dist_loss":dist_loss.item() if dist_loss else None
        }
                
        loss.backward()
        
        if torch.isnan(loss).any():
            assert False, "loss is nan,end training"
            os.execv(sys.executable, [sys.executable] + sys.argv)
                        
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()
        
    
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            
                            
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
                
            timer.start()
            # Densification
            if iteration < opt.densify_until_iter:
                if stage == "coarse" or (stage=="fine"and opt.split_gs_in_fine_stage):
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                    if stage == "coarse":
                        opacity_threshold = opt.opacity_threshold_coarse
                        densify_threshold = opt.densify_grad_threshold_coarse
                    else:    
                        opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                        densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  
                    
                    if  iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<args.max_points:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)
                        
                    if  iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        # print("pruning")
                        # print(f"densify_threshold:{densify_threshold}, opacity_threshold:{opacity_threshold}, size_threshold:{size_threshold}")
                        gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                        
                    # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                    if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<args.max_points and opt.add_point: 
                        gaussians.grow(5,5, scene.model_path, iteration, stage)
                        # torch.cuda.empty_cache()
                    # if stage == 'fine' and iteration % opt.opacity_reset_interval == 0:
                    #     print("reset opacity")
                    #     gaussians.reset_opacity() 
                

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                ckp_path = scene.model_path + f"/chkpnt/{stage}_{iteration}.pth"
                os.makedirs(os.path.dirname(ckp_path), exist_ok=True)
                torch.save((gaussians.capture(), iteration), ckp_path)
                
            training_report(tb_writer, iteration, losses, iter_start.elapsed_time(iter_end), testing_iterations, scene, 
                            render_from_batch, {"stage":stage, "canonical_tri_plane_factor_list":opt.canonical_tri_plane_factor_list, "background":background_type}, 
                            stage, save_iterations=args.save_iterations)
                
                
def training(dataset, hyper, opt, pipe, args):
    # first_iter = 0
    testing_iterations, checkpoint_iterations = args.test_iterations, args.checkpoint_iterations
    checkpoint, debug_from, expname = args.start_checkpoint, args.debug_from, args.expname
    tb_writer = prepare_output_and_logger(expname)
        
    timer = Timer()
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    scene = Scene(dataset, gaussians, load_coarse=None, smooth_win=0)
    
    dataset.model_path = args.model_path
    first_iter = 0
    stage_load = None
    if not checkpoint is None:
        if os.path.exists(checkpoint):
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)
        else:
            import glob
            files_c = sorted(glob.glob(scene.model_path + f"/chkpnt/coarse_*.pth"), key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
            files = files_c + sorted(glob.glob(scene.model_path + f"/chkpnt/fine_*.pth"), key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
            if len(files):
                filename = os.path.basename(files[checkpoint])
                stage_load, iteration_load = filename.split(".")[0].split("_")
                first_iter = int(iteration_load)
                if stage_load=="fine":
                    args.skip_coarse = True
                (model_params, _) = torch.load(files[checkpoint])
                gaussians.restore(model_params, opt)
                print(f"loaded checkpoint: {files[checkpoint]}, continue from iteration: {first_iter}")
    
    timer.start()

    train_l_temp=opt.train_l
    if not args.skip_coarse:
        # opt.train_l=["xyz","deformation","grid","f_dc","f_rest","opacity","scaling","rotation"]
        print(opt.train_l)
        testing_iterations_input = [first_iter + 1] + [x for x in testing_iterations if x > first_iter]
        scene_reconstruction(opt, pipe, testing_iterations_input,
                                checkpoint_iterations, first_iter, debug_from,
                                gaussians, scene, "coarse", tb_writer, opt.coarse_iterations, timer)
    
    if not args.skip_fine:
        if stage_load == "coarse":
            first_iter = 0
        opt.train_l = ["deformation","grid"] # freeze gaussian3d
        print(opt.train_l)
        testing_iterations_input = [first_iter + 1] + [x for x in testing_iterations if x > first_iter]
        scene_reconstruction(opt, pipe, testing_iterations_input,
                            checkpoint_iterations, first_iter, debug_from,
                            gaussians, scene, "fine", tb_writer, opt.iterations, timer)

def prepare_output_and_logger(expname):    
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, loss_dict:dict, elapsed, testing_iterations, scene : Scene, renderFunc, renderKargs, stage, save_iterations=[]):
    if tb_writer:
        for k in loss_dict:
            v = loss_dict[k]
            if v is None:
                continue
            tb_writer.add_scalar(f'{stage}/{k}', v, iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
        
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # 
        validation_configs = ({'name': 'test', 'cameras' : [scene.test_camera[idx] for idx in np.linspace(0, len(scene.test_camera)-1, 10).astype(int)]},
                            #   {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]}
                              )

        for config in validation_configs:
            viewpoint_cameras = [x() if scene.lazy_load else x for x in config['cameras']]
            outputs = renderFunc(viewpoint_cameras, scene.gaussians, **renderKargs)
            pred_masks = outputs["rend_alpha_tensor"]
            gt_image_tensor = outputs["gt_tensor"]
            image_tensor = outputs["rendered_image_tensor"] * pred_masks + gt_image_tensor * (1 - pred_masks)
            l1_test = 0.0
            psnr_test = 0.0
            for idx, viewpoint in enumerate(viewpoint_cameras):
                gt_image = torch.clamp(gt_image_tensor[idx], 0.0, 1.0)
                image = torch.clamp(image_tensor[idx], 0.0, 1.0)
                pred_mask = pred_masks[idx]
                if tb_writer and (idx < 5):
                    tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                    tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/pred_mask".format(viewpoint.image_name), pred_mask[None], global_step=iteration)
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                l1_test += l1_loss(image, gt_image).mean().double()
                mask=None
                psnr_test += psnr(image, gt_image, mask=mask).mean().double()
                
            psnr_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])          
            print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
            # print("sh feature",scene.gaussians.get_features.shape)
            if tb_writer:
                tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1)/100, iteration,max_bins=500)
            
        # Log and save
        if iteration in save_iterations:
            print("\n[ITER {}] Saving Gaussians".format(iteration))
            scene.save(iteration, stage, torch.cat([gt_image, image], dim=2), image_idx=viewpoint.image_name, args=args)
        
        torch.cuda.empty_cache()
            
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     
if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[500*i for i in range(25)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1] + [1000*i for i in range(1,11)])
    parser.add_argument("--background_type", type=str, default="torso")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[2000*i for i in range(1,6)])
    parser.add_argument("--start_checkpoint", type=int, default = -1)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument("--skip_coarse", action="store_true", default=False)
    parser.add_argument("--skip_fine", action="store_true", default=False)
    parser.add_argument("--part", type=str, default="head") # head, face, all
    parser.add_argument("--max_points", type=int, default = 50_000)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        from utils.params_utils import merge_hparams, load_from_file
        config = load_from_file(args.configs)
        args = merge_hparams(args, config, is_training=True)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args)

    # All done
    print("\nTraining complete.")
