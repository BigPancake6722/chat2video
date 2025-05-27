import torch
import numpy as np
from scipy.spatial.transform import Rotation

@torch.no_grad()
def get_state_at_time(pc,viewpoint_camera):    
    means3D = pc.get_xyz
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = pc._scaling
    rotations = pc._rotation
    cov3D_precomp = None
    
    means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, 
                                                                 rotations, opacity, shs,
                                                                 time)
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
    return means3D_final, scales_final, rotations_final, opacity, shs

def matrix_to_quaternion(matrix):
    return Rotation.from_matrix(matrix).as_quat()

def quaternion_to_matrix(quaternion):
    return Rotation.from_quat(quaternion).as_matrix()

def slerp(q1, q2, t):
    """Spherical Linear Interpolation between quaternions."""
    # 确保四元数是单位四元数
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    dot = np.dot(q1, q2)
    
    # 如果四元数点积为负，取q2的反方向以确保最短路径
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # 如果四元数非常接近，使用线性插值
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return (s0 * q1) + (s1 * q2)

def interpolate_rotation_matrices(matrix1, matrix2, t):
    """Interpolate between two rotation matrices."""
    q1 = matrix_to_quaternion(matrix1)
    q2 = matrix_to_quaternion(matrix2)
    q_interpolated = slerp(q1, q2, t)
    return quaternion_to_matrix(q_interpolated)

def interpolate_viewpoint(viewpoint_now, viewpoint_last, coef=1.0):
    # last 0 -> now 1
    rot = viewpoint_now.world_view_transform[:3, :3].numpy()
    trans = viewpoint_now.world_view_transform[3, :].numpy()
    lastR = viewpoint_last.world_view_transform[:3, :3].numpy()
    lastT = viewpoint_last.world_view_transform[3, :].numpy()
    rot = torch.tensor(interpolate_rotation_matrices(lastR, rot, coef), dtype=torch.float)
    trans = torch.tensor((1 - coef) * lastT + coef * trans)
    viewpoint_now.R = rot.numpy()
    viewpoint_now.T = trans[:3].numpy()
    viewpoint_now.world_view_transform[:3, :3] = rot
    viewpoint_now.world_view_transform[3, :] = trans
    w2c = viewpoint_now.world_view_transform             
    viewpoint_now.full_proj_transform = (w2c.unsqueeze(0).bmm(viewpoint_now.projection_matrix.unsqueeze(0))).squeeze(0)
    viewpoint_now.camera_center = -rot@trans[:-1]
    return viewpoint_now
    