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
import torch
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils_talker.point_utils import depth_to_normal
from time import time 
import pdb

def get_from_allmap(allmap, viewpoint_camera:Camera):
    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    w2c = (viewpoint_camera.world_view_transform[:3,:3].T).to(render_normal.device)
    render_normal = (render_normal.permute(1,2,0) @ w2c).permute(2,0,1)

    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    depth_ratio = 0.0
    surf_depth = render_depth_expected * (1-depth_ratio) + depth_ratio * render_depth_median

    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()
    
    return {'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,}
    
def render_from_batch_train(viewpoint_cameras, gaussian : GaussianModel, pipe_debug=False, background="black", scaling_modifier = 1.0, stage="fine", 
                            visualize_attention=False, canonical_tri_plane_factor_list = [], feature_inputs = ["aud", "eye", "cam", "uid"]):
    time1 = time()
    assert background in ["white", "black", "scene", "random", "torso", "bg_w_torso", "gt"]
    batch_size = len(viewpoint_cameras)
    means3D = gaussian.get_xyz.unsqueeze(0).repeat(batch_size, 1, 1) # [B, N, 3]
    opacity = gaussian._opacity.unsqueeze(0).repeat(batch_size, 1, 1) # [B, N, 1]
    shs = gaussian.get_features.unsqueeze(0).repeat(batch_size, 1, 1, 1) # [B, N, 16, 3]
    scales = gaussian._scaling.unsqueeze(0).repeat(batch_size, 1, 1) # [B, N, 2]
    rotations = gaussian._rotation.unsqueeze(0).repeat(batch_size, 1, 1) # [B, N, 4] 
    cov3D_precomp = None
     
    aud_features = []
    eye_features = []
    cam_features = []
    frame_ids = []
    rasterizers = []
    gt_imgs = []
    viewspace_point_tensor_list = []
    means2Ds = []
    lips_list = []
    gt_masks = []
    bg_list = []
    gt_w_bg = []
    seg_masks = []
    
    for viewpoint_camera in viewpoint_cameras:
        screenspace_points = torch.zeros_like(gaussian.get_xyz, dtype=gaussian.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        means2Ds.append(screenspace_points)
        viewspace_point_tensor_list.append(screenspace_points)
        
        if background=="random":
            bg_color = torch.rand((3,), dtype=torch.float32, device="cuda")
            bg_image = bg_color[:, None, None] * torch.ones((1, viewpoint_camera.image_height, viewpoint_camera.image_width), device=bg_color.device)
        elif background=="black": 
            bg_image = torch.zeros((1, viewpoint_camera.image_height, viewpoint_camera.image_width), device="cuda")
        elif background=="white":
            bg_image = torch.ones((1, viewpoint_camera.image_height, viewpoint_camera.image_width), device="cuda")
        elif background=="scene":
            bg_image = viewpoint_camera.bg_image.cuda()
        elif background=="torso":
            bg_image = viewpoint_camera.torso_image[:3, ...].cuda()
        elif background=="bg_w_torso":
            bg_image = viewpoint_camera.bg_w_torso.cuda()
        elif background=="gt":
            bg_image = viewpoint_camera.original_image.cuda()
        
        aud_features.append(viewpoint_camera.aud_f.unsqueeze(0).to(means3D.device))
        eye_features.append(torch.from_numpy(np.array([viewpoint_camera.eye_f])).unsqueeze(0).to(means3D.device))
        cam_features.append(torch.from_numpy(np.concatenate((viewpoint_camera.R.reshape(-1), viewpoint_camera.T.reshape(-1))).reshape(1,-1)).to(means3D.device))
        frame_ids.append(viewpoint_camera.uid)
                
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
            
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx= tanfovx,
            tanfovy= tanfovy,
            bg=bg_image,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=gaussian.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe_debug
        )
        rasterizers.append(GaussianRasterizer(raster_settings=raster_settings))

        fg_mask = viewpoint_camera.fg_mask.cuda()
        gt_masks.append(fg_mask)
        ori_img = viewpoint_camera.original_image.cuda()
        gt_w_bg.append(ori_img)
        gt_image = viewpoint_camera.gt_image.cuda()
        gt_imgs.append(gt_image)
        lips_list.append(viewpoint_camera.lips_rect)
        seg_mask = viewpoint_camera.seg_mask.cuda()
        seg_masks.append(seg_mask)
        bg_list.append(bg_image)
    
    if stage == "coarse":
        aud_features, eye_features, cam_features, frame_ids = None, None, None, None
        means3D_final, scales_temp, rotations_temp, opacity_temp, shs_temp = gaussian._deformation(means3D, scales, rotations, opacity, shs, aud_features, eye_features, cam_features, frame_ids)
        if "scales" in canonical_tri_plane_factor_list:
            # scales_temp = scales_temp-2
            scales_temp = scales_temp
            scales_final = scales_temp
        else: 
            scales_final = scales
            scales_temp = None
        if "rotations" in canonical_tri_plane_factor_list:
            rotations_final = rotations_temp
        else: 
            rotations_final = rotations
            rotations_temp = None
        if "opacity" in canonical_tri_plane_factor_list:
            opacity_final = opacity_temp
        else:
            opacity_final = opacity
            opacity_temp = None
        if "shs" in canonical_tri_plane_factor_list:
            shs_final = shs_temp
        else:
            shs_final = shs
            shs_temp = None
        gaussian.replace_gaussian(scales_temp, rotations_temp, opacity_temp, shs_temp)

    elif stage == "fine":
        aud_features = torch.cat(aud_features,dim=0) 
        eye_features = torch.cat(eye_features,dim=0)
        cam_features = torch.cat(cam_features,dim=0)
        frame_ids = torch.tensor(frame_ids, dtype=torch.int32).to(means3D.device)[:, None]
        
        aud_features = aud_features if "aud" in feature_inputs else torch.zeros_like(aud_features)
        eye_features = eye_features if "eye" in feature_inputs else torch.zeros_like(eye_features)
        cam_features = cam_features if "cam" in feature_inputs else torch.zeros_like(cam_features)
        frame_ids = frame_ids if 'uid' in feature_inputs else torch.zeros(batch_size, 1, dtype=torch.int32).to(means3D.device)
        
        means3D_final, scales_final, rotations_final, opacity_final, shs_final, attention = gaussian._deformation(means3D, scales, rotations, opacity, shs, aud_features, eye_features,cam_features,frame_ids)
                                                                                                    
    scales_final = gaussian.scaling_activation(scales_final[..., :2]) # (B, pts, 2) for 2dgs
    rotations_final = torch.nn.functional.normalize(rotations_final,dim=2) 
    opacity_final = gaussian.opacity_activation(opacity_final)
        
    
    rendered_image_list = []
    radii_list = []
    depth_list = []
    alpha_list = []
    visibility_filter_list = []
    rendered_lips_list = []
    gt_lips_list = []
    rend_dist_list = []
    rend_normal_list = []
    surf_normal_list = []
    attention_image_list = []
    
    for idx, rasterizer in enumerate(rasterizers):
        colors_precomp = None
        rendered_image, radii, allmap = rasterizer(
            means3D = means3D_final[idx],
            means2D = means2Ds[idx],
            shs = shs_final[idx],
            colors_precomp = colors_precomp,
            opacities = opacity_final[idx],
            scales = scales_final[idx],
            rotations = rotations_final[idx],
            cov3D_precomp = cov3D_precomp)
        rend_properties = get_from_allmap(allmap, viewpoint_cameras[idx])
        
        rendered_image_list.append(rendered_image)
        radii_list.append(radii)
        depth_list.append(rend_properties["surf_depth"])
        alpha_list.append(rend_properties["rend_alpha"])
        rend_dist_list.append(rend_properties["rend_dist"])
        rend_normal_list.append(rend_properties['rend_normal'])
        surf_normal_list.append(rend_properties['surf_normal'])
        visibility_filter_list.append((radii > 0))
        
        # lips
        y1,y2,x1,x2 = lips_list[idx]
        lip_crop = rendered_image[:, y1:y2,x1:x2]
        gt_lip_crop = gt_imgs[idx][:, y1:y2,x1:x2]
        rendered_lips_list.append(lip_crop.flatten())
        gt_lips_list.append(gt_lip_crop.flatten())
        
        if visualize_attention:
            colors_precomp = attention[idx, 0, :, :4]
            attn_image, radii, allmap = rasterizer(
                means3D = means3D_final[idx],
                means2D = means2Ds[idx],
                shs = None,
                colors_precomp = colors_precomp,
                opacities = opacity_final[idx], 
                scales = scales_final[idx],
                rotations = rotations_final[idx],
                cov3D_precomp = cov3D_precomp,)
            attention_image_list.append(attn_image)
            
    attention_tensor = None
    rendered_lips_tensor ,gt_lips_tensor = None, None
        
    radii = torch.stack(radii_list,0).max(dim=0).values
    visibility_filter_tensor = torch.stack(visibility_filter_list).any(dim=0)
    rendered_image_tensor = torch.stack(rendered_image_list,0)
    gt_tensor = torch.stack(gt_imgs,0)
    depth_tensor = torch.stack(depth_list,dim=0)
    gt_masks_tensor = torch.stack(gt_masks,dim=0)
    gt_w_bg_tensor = torch.stack(gt_w_bg,dim=0)
    gt_segs_tensor = torch.stack(seg_masks,dim=0)
    rend_alpha_tensor = torch.stack(alpha_list,dim=0)
    rend_dist_tensor = torch.stack(rend_dist_list,dim=0)
    rend_normal_tensor = torch.stack(rend_normal_list,dim=0)
    surf_normal_tensor = torch.stack(surf_normal_list,dim=0)
    if visualize_attention:
        attention_tensor = torch.stack(attention_image_list,dim=0)
        
    bg_tensor = torch.stack(bg_list,dim=0)
    rendered_lips_tensor = torch.cat(rendered_lips_list,0)
    gt_lips_tensor = torch.cat(gt_lips_list,0)     
    inference_time = time()-time1   
        
    return {
        "rendered_image_tensor": rendered_image_tensor,
        "gt_tensor":gt_tensor,
        "viewspace_points": screenspace_points,
        "visibility_filter_tensor" : visibility_filter_tensor,
        "viewspace_point_tensor_list" : viewspace_point_tensor_list,
        "radii": radii,
        "depth_tensor": depth_tensor,
        "rendered_lips_tensor":rendered_lips_tensor,
        "gt_lips_tensor":gt_lips_tensor,
        "rendered_w_bg_tensor":bg_tensor[:, :3, ...],
        "inference_time":inference_time,
        "gt_masks_tensor":gt_masks_tensor,
        "gt_w_bg_tensor":gt_w_bg_tensor,
        "gt_segs_tensor":gt_segs_tensor,
        "rend_dist_tensor":rend_dist_tensor,
        "rend_normal_tensor":rend_normal_tensor,
        "surf_normal_tensor":surf_normal_tensor,
        "rend_alpha_tensor":rend_alpha_tensor,
        "attention_tensor":attention_tensor,
        }


def render_from_batch_infer(viewpoint_cameras, gaussian : GaussianModel, pipe_debug=False, background="black", scaling_modifier = 1.0, 
                            visualize_attention=False, feature_inputs = ["aud", "eye", "cam", "uid"]):
    time1 = time()
    assert background in ["white", "black", "scene", "random", "torso", "bg_w_torso", "gt"]
    batch_size = len(viewpoint_cameras)
    means3D = gaussian.get_xyz.unsqueeze(0).repeat(batch_size, 1, 1) # [B, N, 3]
    opacity = gaussian._opacity.unsqueeze(0).repeat(batch_size, 1, 1) # [B, N, 1]
    shs = gaussian.get_features.unsqueeze(0).repeat(batch_size, 1, 1, 1) # [B, N, 16, 3]
    scales = gaussian._scaling.unsqueeze(0).repeat(batch_size, 1, 1) # [B, N, 2]
    rotations = gaussian._rotation.unsqueeze(0).repeat(batch_size, 1, 1) # [B, N, 4] 
    screenspace_points = torch.zeros_like(gaussian.get_xyz, dtype=gaussian.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    cov3D_precomp = None
     
    aud_features = []
    eye_features = []
    cam_features = []
    frame_ids = []
    rasterizers = []
    bg_list = []
    seg_masks = []
    
    for viewpoint_camera in viewpoint_cameras:
        
        if background=="random":
            bg_color = torch.rand((3,), dtype=torch.float32, device="cuda")
            bg_image = bg_color[:, None, None] * torch.ones((1, viewpoint_camera.image_height, viewpoint_camera.image_width), device=bg_color.device)
        elif background=="black": 
            bg_image = torch.zeros((1, viewpoint_camera.image_height, viewpoint_camera.image_width), device="cuda")
        elif background=="white":
            bg_image = torch.ones((1, viewpoint_camera.image_height, viewpoint_camera.image_width), device="cuda")
        elif background=="scene":
            bg_image = viewpoint_camera.bg_image.cuda()
        elif background=="torso":
            bg_image = viewpoint_camera.torso_image[:3, ...].cuda()
        elif background=="bg_w_torso":
            bg_image = viewpoint_camera.bg_w_torso.cuda()
        elif background=="gt":
            bg_image = viewpoint_camera.original_image.cuda()
        
        aud_features.append(viewpoint_camera.aud_f.unsqueeze(0).to(means3D.device))
        eye_features.append(torch.from_numpy(np.array([viewpoint_camera.eye_f])).unsqueeze(0).to(means3D.device))
        cam_features.append(torch.from_numpy(np.concatenate((viewpoint_camera.R.reshape(-1), viewpoint_camera.T.reshape(-1))).reshape(1,-1)).to(means3D.device))
        frame_ids.append(viewpoint_camera.uid)
                
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
            
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx= tanfovx,
            tanfovy= tanfovy,
            bg=bg_image,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=gaussian.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe_debug
        )
        rasterizers.append(GaussianRasterizer(raster_settings=raster_settings))
        bg_list.append(bg_image)
        seg_masks.append(viewpoint_camera.seg_mask.cuda())
    
    aud_features = torch.cat(aud_features,dim=0) 
    eye_features = torch.cat(eye_features,dim=0)
    cam_features = torch.cat(cam_features,dim=0)
    frame_ids = torch.tensor(frame_ids, dtype=torch.int32).to(means3D.device)[:, None]
    
    aud_features = aud_features if "aud" in feature_inputs else torch.zeros_like(aud_features)
    eye_features = eye_features if "eye" in feature_inputs else torch.zeros_like(eye_features)
    cam_features = cam_features if "cam" in feature_inputs else torch.zeros_like(cam_features)
    frame_ids = frame_ids if 'uid' in feature_inputs else torch.zeros(batch_size, 1, dtype=torch.int32).to(means3D.device)
    
    means3D_final, scales_final, rotations_final, opacity_final, shs_final, attention = gaussian._deformation(means3D, scales, rotations, opacity, shs, aud_features, eye_features,cam_features, frame_ids)
                                                                                                    
    scales_final = gaussian.scaling_activation(scales_final[..., :2]) # (B, pts, 2) for 2dgs
    rotations_final = torch.nn.functional.normalize(rotations_final,dim=2) 
    opacity_final = gaussian.opacity_activation(opacity_final)
        
    
    rendered_image_list = []
    radii_list = []
    depth_list = []
    alpha_list = []
    rend_dist_list = []
    rend_normal_list = []
    surf_normal_list = []
    attention_image_list = []
    
    for idx, rasterizer in enumerate(rasterizers):
        colors_precomp = None
        rendered_image, radii, allmap = rasterizer(
            means3D = means3D_final[idx],
            means2D = screenspace_points,
            shs = shs_final[idx],
            colors_precomp = colors_precomp,
            opacities = opacity_final[idx],
            scales = scales_final[idx],
            rotations = rotations_final[idx],
            cov3D_precomp = cov3D_precomp)
        
        rend_properties = get_from_allmap(allmap, viewpoint_cameras[idx])
        
        rendered_image_list.append(rendered_image)
        radii_list.append(radii)
        depth_list.append(rend_properties["surf_depth"])
        alpha_list.append(rend_properties["rend_alpha"])
        rend_dist_list.append(rend_properties["rend_dist"])
        rend_normal_list.append(rend_properties['rend_normal'])
        surf_normal_list.append(rend_properties['surf_normal'])
        
        if visualize_attention:
            colors_precomp = attention[idx, 0, :, :4]
            attn_image, radii, allmap = rasterizer(
                means3D = means3D_final[idx],
                means2D = screenspace_points,
                shs = None,
                colors_precomp = colors_precomp,
                opacities = opacity_final[idx], 
                scales = scales_final[idx],
                rotations = rotations_final[idx],
                cov3D_precomp = cov3D_precomp,)
            attention_image_list.append(attn_image)
            
    attention_tensor = None        
    radii = torch.stack(radii_list,0).max(dim=0).values
    rendered_image_tensor = torch.stack(rendered_image_list,0)
    depth_tensor = torch.stack(depth_list,dim=0)
    rend_alpha_tensor = torch.stack(alpha_list,dim=0)
    rend_dist_tensor = torch.stack(rend_dist_list,dim=0)
    rend_normal_tensor = torch.stack(rend_normal_list,dim=0)
    surf_normal_tensor = torch.stack(surf_normal_list,dim=0)
    bg_tensor = torch.stack(bg_list,dim=0)
    gt_segs_tensor = torch.stack(seg_masks,dim=0)
    if visualize_attention:
        attention_tensor = torch.stack(attention_image_list,dim=0)
        
    inference_time = time()-time1   
        
    return {
        "rendered_image_tensor": rendered_image_tensor,
        "radii": radii,
        "depth_tensor": depth_tensor,
        "bg_tensor":bg_tensor[:, :3, ...],
        "gt_segs_tensor":gt_segs_tensor,
        "inference_time":inference_time,
        "rend_dist_tensor":rend_dist_tensor,
        "rend_normal_tensor":rend_normal_tensor,
        "surf_normal_tensor":surf_normal_tensor,
        "rend_alpha_tensor":rend_alpha_tensor,
        "attention_tensor":attention_tensor,
        }