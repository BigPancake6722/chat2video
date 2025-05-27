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

import os
import sys
from PIL import Image
from scene.cameras import Camera

from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.hyper_loader import Load_hyper_data, format_hyper_data
import copy
from utils_talker.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np, math
import torch
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils_talker.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils_talker.general_utils import PILtoTorch
from tqdm import tqdm
from scene.utils import get_audio_features
from scipy.spatial.transform import Rotation
from io import BytesIO
import cv2

from scene.dataset_readers import read_timeline, getNerfppNorm

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    full_image: np.array
    full_image_path: str
    gt_image: np.array
    gt_image_path: str
    image_name: str
    width: int
    height: int
    torso_image: np.array
    torso_image_path: str
    bg_image: np.array
    bg_image_path: str
    mask: np.array
    mask_path: str
    trans: np.array
    face_rect: list
    lhalf_rect: list
    aud_f: torch.FloatTensor
    eye_f: np.array
    eye_rect: list
    lips_rect: list
   
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    torso_cameras: list
    custom_cameras: list
    nerf_normalization: dict
    ply_path: str
    maxtime: int

def read_and_extract_head(ori_image_path):
    ori_image = cv2.imread(ori_image_path, cv2.IMREAD_UNCHANGED)
    seg = cv2.imread(ori_image_path.replace('ori_imgs', 'parsing').replace('.jpg', '.png'))
    head_mask = (seg[..., 0] == 255) & (seg[..., 1] == 0) & (seg[..., 2] == 0)
    
    # Create an empty image with the same shape as the original image
    # head_image = np.zeros_like(ori_image)

    # Apply the mask to the original image to extract the head part
    # head_image[head_mask] = ori_image[head_mask]
    return ori_image, head_mask
    
def extract_array_continuous(arr, ids, idx, hwin):
    n = len(arr)
    result = []
    anchor = ids[idx]
    pre_i = idx
    for j, i0 in enumerate(range(idx, idx-hwin-1, -1)):
        i0 = max(i0, 0)
        i = ids[i0]
        if i + j == anchor:
            pre_i = i0
        result.insert(0, arr[pre_i])
    pre_i = idx
    for j, i0 in enumerate(range(idx+1, idx+hwin+1)):
        i0 = min(i0, n-1)
        i = ids[i0]
        if anchor + j + 1 == i:
            pre_i = i0
        result.append(arr[pre_i])
    return result

def generateCamerasFromTransforms(path, template_transformsfile, extension, maxtime):
    trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

    rot_phi = lambda phi : torch.Tensor([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]]).float()

    rot_theta = lambda th : torch.Tensor([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]]).float()
    def pose_spherical(theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(phi/180.*np.pi) @ c2w
        c2w = rot_theta(theta/180.*np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
        return c2w
    cam_infos = []
    # generate render poses and times
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)
    render_times = torch.linspace(0,maxtime,render_poses.shape[0])
    with open(os.path.join(path, template_transformsfile)) as json_file:
        template_json = json.load(json_file)
        try:
            fovx = template_json["camera_angle_x"]
        except:
            fovx = focal2fov(template_json["fl_x"], template_json['w'])
    print("hello!!!!")
    # breakpoint()
    # load a single image to get image info.
    for idx, frame in enumerate(template_json["frames"]):
        cam_name = os.path.join(path, frame["file_path"] + extension)
        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        image = PILtoTorch(image,(800,800))
        break
    # format information
    for idx, (time, poses) in enumerate(zip(render_times,render_poses)):
        time = time/maxtime
        matrix = np.linalg.inv(np.array(poses))
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
        # fovy = fovx
        FovY = fovy 
        FovX = fovx
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=None, image_name=None, width=image.shape[1], height=image.shape[2],
                            time = time, mask=None))
    return cam_infos

def euler2rot2(euler_angle):
    batch_size = euler_angle.shape[0]
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    one = torch.ones(batch_size, 1, 1).to(euler_angle.device)
    zero = torch.zeros(batch_size, 1, 1).to(euler_angle.device)
    rot_x = torch.cat(
        (
            torch.cat((one, zero, zero), 1),
            torch.cat((zero, theta.cos(), theta.sin()), 1),
            torch.cat((zero, -theta.sin(), theta.cos()), 1),
        ),
        2,
    )
    rot_y = torch.cat(
        (
            torch.cat((phi.cos(), zero, -phi.sin()), 1),
            torch.cat((zero, one, zero), 1),
            torch.cat((phi.sin(), zero, phi.cos()), 1),
        ),
        2,
    )
    rot_z = torch.cat(
        (
            torch.cat((psi.cos(), -psi.sin(), zero), 1),
            torch.cat((psi.sin(), psi.cos(), zero), 1),
            torch.cat((zero, zero, one), 1),
        ),
        2,
    )
    return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))

def euler2rot(euler_angle):
    batch_size = euler_angle.shape[0]
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    one = torch.ones((batch_size, 1, 1), dtype=torch.float32, device=euler_angle.device)
    zero = torch.zeros((batch_size, 1, 1), dtype=torch.float32, device=euler_angle.device)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def fetchPly(path):
    with open(path, 'rb') as f: # 可以释放句柄
        plydata = PlyData.read(f)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def readCamerasFromTracksTransforms(path, meshfile, transformsfile, aud_features, eye_features, 
                                    extension=".jpg", mapper = {}, preload=False, custom_aud =None):
    cam_infos = []
    mesh_path = os.path.join(path, meshfile)
    track_params = torch.load(mesh_path)
    trans_infos = track_params["trans"]
    
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        
    contents["fl_x"] = contents["focal_len"]
    contents["fl_y"] = contents["focal_len"]
    contents["w"] = contents["cx"] * 2
    contents["h"] = contents["cy"] * 2

    fovx = focal2fov(contents['fl_x'],contents['w'])
    fovy = focal2fov(contents['fl_y'],contents['h'])
    frames = contents["frames"]
    f_path = os.path.join(path, "ori_imgs")
    
    FovY = fovy 
    FovX = fovx
    
    # background_image
    bg_image_path = os.path.join(path, "bc.jpg")
    bg_img = cv2.imread(bg_image_path, cv2.IMREAD_UNCHANGED) # [H, W, 3]
    if bg_img.shape[0] != contents["h"] or bg_img.shape[1] != contents["w"]:
        bg_img = cv2.resize(bg_img, (contents["w"], contents["h"]), interpolation=cv2.INTER_AREA)
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
    bg_img = torch.from_numpy(bg_img).permute(2,0,1).float() / 255.0
        
    if custom_aud:
        auds = aud_features
    else:    
        auds = [aud_features[min(frame['aud_id'], aud_features.shape[0] - 1)] for frame in frames]
        auds = torch.stack(auds, dim=0)
        
    for idx, frame in enumerate(frames): # len(frames): 7272
        
        cam_name = os.path.join(f_path, str(frame["img_id"]) + extension)
        aud_feature = get_audio_features(auds, att_mode = 2, index = idx)     
        
        
        # Camera Codes  从3DMM获取
        euler = track_params["euler"][frame["img_id"]]
        R = euler2rot(euler.unsqueeze(0))
        
        flip_rot = torch.tensor(
            [[-1,  0,  0],  # This flips the X axis
            [ 0,  1,  0],  # Y axis remains the same
            [ 0,  0, 1]], # This flips the Z axis, maintaining the right-hand rule
            dtype=R.dtype,
            device=R.device
        ).view(1, 3, 3)
        # flip_rot = flip_rot.expand_as(R)  # Make sure it has the same batch size as R

        # Apply the flip rotation by matrix multiplication
        # Depending on your convention, you might need to apply the flip before or after the original rotation.
        # Use torch.matmul(flip_rot, R) if the flip should be applied globally first,
        # or torch.matmul(R, flip_rot) if the flip should be applied in the camera's local space.
        R = torch.matmul(flip_rot, R)
        R = R.squeeze(0).cpu().numpy()
        T = track_params["trans"][frame["img_id"]].unsqueeze(0).cpu().numpy()
        
        R = -np.transpose(R)
        
        T = -T
        T[:, 0] = -T[:, 0] 

        # Get Iamges for Facial 
        image_name = Path(cam_name).stem

        full_image_path = cam_name
        torso_image_path = os.path.join(path, 'torso_imgs', str(frame['img_id']) + '.png')
        mask_path = cam_name.replace('ori_imgs', 'parsing').replace('.jpg', '.png')
        
        # Landmark and extract face
        lms = np.loadtxt(os.path.join(path, 'ori_imgs', str(frame['img_id']) + '.lms')) # [68, 2]

        lh_xmin, lh_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max()) # actually lower half area
        xmin, xmax = int(lms[:, 1].min()), int(lms[:, 1].max())
        ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
        face_rect = [xmin, xmax, ymin, ymax]
        lhalf_rect = [lh_xmin, lh_xmax, ymin, ymax]
        
        # Eye Area and Eye Rect
        eye_area = eye_features[frame['img_id']]
        eye_area = np.clip(eye_area, 0, 2) / 2
        
        xmin, xmax = int(lms[36:48, 1].min()), int(lms[36:48, 1].max())
        ymin, ymax = int(lms[36:48, 0].min()), int(lms[36:48, 0].max())
        eye_rect = [xmin, xmax, ymin, ymax]
        
        # Finetune Lip Area
        lips = slice(48, 60)
        xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
        ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())
        cx = (xmin + xmax) // 2
        cy = (ymin + ymax) // 2
        l = max(xmax - xmin, ymax - ymin) // 2
        xmin = max(0, cx - l)
        xmax = min(contents["h"], cx + l)
        ymin = max(0, cy - l)
        ymax = min(contents["w"], cy + l)

        lips_rect = [xmin, xmax, ymin, ymax]
        
        if preload:
            ori_image = cv2.imread(cam_name, cv2.IMREAD_UNCHANGED)
            seg = cv2.imread(mask_path)
            # head_mask = (seg[..., 0] == 255) & (seg[..., 1] == 0) & (seg[..., 2] == 0)
            
            ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
            ori_image = torch.from_numpy(ori_image).permute(2,0,1).float() / 255.0
            
            # torso images 
            torso_img = cv2.imread(torso_image_path, cv2.IMREAD_UNCHANGED) # [H, W, 4]
            torso_img = cv2.cvtColor(torso_img, cv2.COLOR_BGRA2RGBA)
            torso_img = torso_img.astype(np.float32) / 255 # [H, W, 3/4]
            
        else:
            ori_image = None
            torso_img = None
            seg = None
        

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, full_image=ori_image, full_image_path=full_image_path,
                        image_name=image_name, width=contents["w"], height=contents["h"],
                        torso_image=torso_img, torso_image_path=torso_image_path, bg_image=bg_img, bg_image_path=bg_image_path,
                        mask=seg, mask_path=mask_path, trans=trans_infos[frame["img_id"]],
                        face_rect=face_rect, lhalf_rect=lhalf_rect, aud_f=aud_feature, eye_f=eye_area, eye_rect=eye_rect, lips_rect=lips_rect))
    return cam_infos     

def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    new_pose = np.array([
        [pose[0, 0], pose[0, 1], pose[0, 2], pose[0, 3] * scale + offset[0]],
        [pose[1, 0], pose[1, 1], pose[1, 2], pose[1, 3] * scale + offset[1]],
        [pose[2, 0], pose[2, 1], pose[2, 2], pose[2, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose

def smooth_camera_path(poses, kernel_size=5):
    # smooth the camera trajectory...
    # poses: [N, 4, 4], numpy array

    N = poses.shape[0]
    K = kernel_size // 2
    
    trans = poses[:, :3, 3].copy() # [N, 3]
    rots = poses[:, :3, :3].copy() # [N, 3, 3]

    for i in range(N):
        start = max(0, i - K)
        end = min(N, i + K + 1)
        poses[i, :3, 3] = trans[start:end].mean(0)
        poses[i, :3, :3] = Rotation.from_matrix(rots[start:end]).mean().as_matrix()

    return poses

def load_wav2ave_feature(aud_path):
    from scene.utils import AudDataset
    from scene.networks import AudioEncoder
    from torch.utils.data import DataLoader
    # ave aud features
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioEncoder().to(device).eval()
    ckpt = torch.load('/root/chat2video/sync-gaussian-talker/scene/checkpoints/audio_visual_encoder.pth')
    model.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt.items()})
    dataset = AudDataset(aud_path)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    outputs = []
    for mel in data_loader:
        mel = mel.to(device)
        with torch.no_grad():
            out = model(mel)
        outputs.append(out)
    outputs = torch.cat(outputs, dim=0).cpu()
    first_frame, last_frame = outputs[:1], outputs[-1:]
    aud_features = torch.cat([first_frame.repeat(2, 1), outputs, last_frame.repeat(2, 1)],
                                dim=0).unsqueeze(1)
    return aud_features    


def get_mirro_index(index, length):
    leng = length - 1
    len2 = leng * 2
    remainder = index % len2
    half = remainder // leng
    minus = remainder % leng
    if half == 0:
        return remainder
    else:
        return leng - minus

def readSyncTalkDatasetInfo(path, white_background, eval, extension=".jpg", custom_aud=None, train_torso=False, eval_torso=False, smooth_win=2, start_idx=0):
        
    # load action units
    # import pandas as pd
    # au_blink_info=pd.read_csv(os.path.join(path, 'au.csv'))
    # eye_features = au_blink_info[' AU45_r'].values
    bs_path = os.path.join(path, 'bs.npy')
    au_path = os.path.join(path, 'au.csv')
    if isinstance(custom_aud, np.ndarray):
        aud_features = load_wav2ave_feature(custom_aud)
        print(f"extract features from bytes stream")
    else:
        if custom_aud and os.path.exists(custom_aud):
            print("scene talking dataset reader", custom_aud)
            if custom_aud.endswith('.npy'):
                aud_features = np.load(custom_aud)
                aud_features = torch.from_numpy(aud_features).float()
                print(f"load features from {custom_aud}")
            elif custom_aud.endswith('.wav'):
                aud_ave = custom_aud.replace('.wav', '_ave.npy')
                if os.path.exists(aud_ave):
                    aud_features = np.load(aud_ave)
                    aud_features = torch.from_numpy(aud_features).float()
                    print(f"load features from {aud_ave}")
                else:
                    aud_features = load_wav2ave_feature(custom_aud)
                    print(f"extract features from {custom_aud}")
            else:
                raise ValueError('custom_aud should be either .npy or .wav')
            print("Reading Custom Audio Input")
        else:
            aud_path = os.path.join(path, 'aud.wav')
            aud_ave = os.path.join(path, 'aud_ave.npy')
            if os.path.exists(aud_ave):
                print(f"load features from {aud_ave}")
                aud_features = np.load(aud_ave)
                aud_features = torch.from_numpy(aud_features).float()
            else:
                print("loading audio features from ", aud_path)
                aud_features = load_wav2ave_feature(aud_path)
            print(f'load aud_features: {aud_features.shape}')
            custom_cam_infos=None
    
    exp_features = None
    if os.path.exists(bs_path):
        bs = np.load(bs_path)
        # eye_features = bs[:,8:10]
        eye_features = bs[:,:22]
        # if self.opt.bs_area == "upper":
        #     bs = np.hstack((bs[:, 0:5], bs[:, 8:10]))
        # elif self.opt.bs_area == "single":
        #     bs = np.hstack((bs[:, 0].reshape(-1, 1),bs[:, 2].reshape(-1, 1),bs[:, 3].reshape(-1, 1), bs[:, 8].reshape(-1, 1)))
    elif os.path.exists(au_path):
        import pandas as pd
        au_blink_info=pd.read_csv(os.path.join(path, 'au.csv'))
        eye_features = au_blink_info[' AU45_r'].values
    else:
        print("bs.npy or au.npy not found, have no eye features")
        eye_features = np.zeros(aud_features.shape[0])
        
    
    ply_path = os.path.join(path, "fused.ply")
    if train_torso:
        ply_path = os.path.join(path, "torso.ply")
    mesh_path = os.path.join(path, "track_params.pt")

    timestamp_mapper, max_time = read_timeline(path)
    print("Use SyncTalk dataset")
    # video_cam_infos = None 
    torso_cam_infos = None
    if eval_torso:
        torso_cam_infos = readCamerasFromNerfTransform(path, "track_params.pt", "transforms_torso.json", aud_features, 
                                                        eye_features, extension, timestamp_mapper, preload=False, smooth_win=smooth_win,
                                                        start_idx=start_idx)
    if not (len(custom_aud.shape) == 0 or all(s == 0 for s in custom_aud.shape)):
        print("Reading Custom Transforms")
        custom_cam_infos = readCamerasFromNerfTransforms(path, "track_params.pt", ["transforms_train.json", "transforms_val.json"], 
                                                        aud_features, eye_features, extension, timestamp_mapper, preload=False, 
                                                        smooth_win=smooth_win, infer_mode=True, start_idx=start_idx)
        nerf_normalization = getNerfppNorm(custom_cam_infos)
        train_cam_infos, test_cam_infos = None, None
    else:
        if train_torso:
            print(f"Reading Torso Track Transforms: transforms_torso.json")
            test_cam_infos = train_cam_infos = readCamerasFromNerfTransform(path, "track_params.pt", "transforms_torso.json", aud_features, 
                                                        eye_features, extension, timestamp_mapper, preload=False, smooth_win=smooth_win, start_idx=start_idx)
        else:
            print("Reading Training Transforms")
            train_cam_infos = readCamerasFromNerfTransform(path, "track_params.pt", "transforms_train.json", aud_features, 
                                                        eye_features, extension, timestamp_mapper, preload=False, smooth_win=smooth_win, start_idx=start_idx)
            print("Reading Test Transforms")
            test_cam_infos = readCamerasFromNerfTransform(path, "track_params.pt", "transforms_val.json", aud_features, 
                                                        eye_features, extension, timestamp_mapper, preload=False, smooth_win=smooth_win, start_idx=start_idx)
            if not eval:
                train_cam_infos.extend(test_cam_infos)
                test_cam_infos = train_cam_infos

        nerf_normalization = getNerfppNorm(train_cam_infos)

    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        # num_pts = 2000
        # print(f"Generating random point cloud ({num_pts})...")
        # We create random points inside the bounds of the synthetic Blender scenes
        # Initialize with 3DMM Vertices
        facial_mesh = torch.load(mesh_path).get("vertices")
        if facial_mesh is None:
            num_pts = 20_000
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            # assert False, "track_params.pt have no vertices"
            print(f"Generating random point cloud ({num_pts})...")
        else:
            average_facial_mesh = torch.mean(facial_mesh, dim=0)
            xyz = average_facial_mesh.cpu().numpy()
            print(f"Load point cloud from {mesh_path}")
        shs = np.random.random((xyz.shape[0], 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((xyz.shape[0], 3)))        
    else:
        pcd = fetchPly(ply_path)
        print(f"Load point cloud from {ply_path}")

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           torso_cameras=torso_cam_infos,
                           custom_cameras=custom_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time
                           )
    return scene_info

def readCamerasFromNerfTransforms(path, meshfile, transformsfiles, aud_features, eye_features, 
                                    extension=".jpg", mapper = {}, preload=False,
                                    scale=1., offset=[0, 0, 0], smooth_win=2, infer_mode=False, start_idx = 0):
    inputs = None
    for transformsfile in transformsfiles:
        with open(os.path.join(path, transformsfile)) as json_file:
            contents = json.load(json_file)
        if inputs is None:
            inputs = contents
        else:
            inputs["frames"] += contents["frames"]
    cam_infos = readCamerasFromNerfTransform(path, meshfile, inputs, aud_features, eye_features, 
                                                extension, mapper, preload, scale, offset, smooth_win, infer_mode, start_idx=start_idx)
    return cam_infos

def readCamerasFromNerfTransform(path, meshfile, transformsfile, aud_features, eye_features, 
                                    extension=".jpg", mapper = {}, preload=False, scale=1., offset=[0, 0, 0],
                                    smooth_win=2, infer_mode=False, start_idx=0):
    cam_infos = []
    mesh_path = os.path.join(path, meshfile)
    track_params = torch.load(mesh_path)
    trans_infos = track_params["trans"]
    
    if isinstance(transformsfile, dict):
        contents = transformsfile
    elif isinstance(transformsfile, str):
        with open(os.path.join(path, transformsfile)) as json_file:
            contents = json.load(json_file)
        
    contents["fl_x"] = contents["focal_len"]
    contents["fl_y"] = contents["focal_len"]
    contents["w"] = int(contents["cx"] * 2)
    contents["h"] = int(contents["cy"] * 2)

    fovx = focal2fov(contents['fl_x'],contents['w'])
    fovy = focal2fov(contents['fl_y'],contents['h'])
    frames = contents["frames"]
    f_path = os.path.join(path, "ori_imgs")
    
    FovY = fovy 
    FovX = fovx
    
    # background_image
    bg_image_path = os.path.join(path, "bc.jpg")
    bg_img = cv2.imread(bg_image_path, cv2.IMREAD_UNCHANGED) # [H, W, 3]
    if bg_img.shape[0] != contents["h"] or bg_img.shape[1] != contents["w"]:
        bg_img = cv2.resize(bg_img, (contents["w"], contents["h"]), interpolation=cv2.INTER_AREA)
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
    bg_img = torch.from_numpy(bg_img).permute(2,0,1).float() / 255.0
        
    startidx = 0
    if infer_mode:
        auds = aud_features
        num_frames = len(auds)
        slicer = [get_mirro_index(idx, len(frames)) for idx in range(start_idx, start_idx + num_frames)]
        frames = [frames[x] for x in slicer]
    else:
        # auds = aud_features
        startidx = frames[0]['aud_id']
        endidx = frames[-1]['aud_id']
        auds = aud_features[startidx: endidx + 1, ...]
        if len(auds)==0:
            return cam_infos
    print("start frame idx: ", frames[0]["img_id"])
    Rs, Ts, uids = [], [], []
    for idx, frame in enumerate(frames):
        # Camera Codes  从transform_xx.json中获得R和T
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w) # 转w2c为了得到相机中心点
        R = np.transpose(w2c[:3, :3]) # 实际输出为c2w
        R[:, 1:3] *= -1 # 翻转y/z轴，和渲染方向有关
        T = -w2c[:3, 3]
        T[0] *= -1 # T[1:3]*-1 / -T 同样是渲染方向相关
        
        Rs.append(R)
        Ts.append(T)
        uids.append(frame["img_id"])
        
    if smooth_win > 0:
        print("doing camera smooth.")
        for idx in tqdm(range(len(Rs))):
            quats = [Rotation.from_matrix(R).as_quat() for R in extract_array_continuous(Rs, uids, idx, smooth_win)]
            trans = [tran for tran in extract_array_continuous(Ts, uids, idx, smooth_win)]
            quats_avg = np.mean(quats, axis=0)
            Rs[idx] = Rotation.from_quat(quats_avg).as_matrix().astype(np.float32)
            Ts[idx] = np.mean(trans, axis=0).astype(np.float32)
        
    print("preloading data..")
    for idx, frame in enumerate(tqdm(frames)):
        jdx = frame['img_id']
        fdx = idx if infer_mode else jdx - startidx
        if idx>auds.shape[0]:
            break
        cam_name = os.path.join(f_path, str(jdx) + extension)
        aud_feature = get_audio_features(auds, att_mode = 2, index = fdx)
        # get rot and trans
        R = Rs[idx]
        T = Ts[idx]

        # Get Iamges for Facial 
        image_name = Path(cam_name).stem

        full_image_path = cam_name
        gt_image_path = cam_name.replace('ori_imgs', 'gt_imgs')
        torso_image_path = os.path.join(path, 'torso_imgs', str(jdx) + '.png')
        mask_path = cam_name.replace('ori_imgs', 'parsing').replace('.jpg', '.png')
        
        # Landmark and extract face
        lms = np.loadtxt(os.path.join(path, 'ori_imgs', str(jdx) + '.lms')) # [68, 2]

        lh_xmin, lh_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max()) # actually lower half area
        xmin, xmax = int(lms[:, 1].min()), int(lms[:, 1].max())
        ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
        face_rect = [xmin, xmax, ymin, ymax]
        lhalf_rect = [lh_xmin, lh_xmax, ymin, ymax]
        
        # Eye Area and Eye Rect
        eye_area = np.clip(eye_features[jdx], 0, 2) / 2
        
        xmin, xmax = int(lms[36:48, 1].min()), int(lms[36:48, 1].max())
        ymin, ymax = int(lms[36:48, 0].min()), int(lms[36:48, 0].max())
        eye_rect = [xmin, xmax, ymin, ymax]
        
        # Finetune Lip Area
        lips = slice(48, 60)
        xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
        ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())
        cx = (xmin + xmax) // 2
        cy = (ymin + ymax) // 2
        l = max(xmax - xmin, ymax - ymin) // 2
        xmin = max(0, cx - l)
        xmax = min(contents["h"], cx + l)
        ymin = max(0, cy - l)
        ymax = min(contents["w"], cy + l)
        lips_rect = [xmin, xmax, ymin, ymax]
        
        if preload:
            ori_image = cv2.imread(cam_name, cv2.IMREAD_UNCHANGED)
            gt_image = cv2.imread(gt_image_path, cv2.IMREAD_UNCHANGED)
            seg = cv2.imread(mask_path)
            # head_mask = (seg[..., 0] == 255) & (seg[..., 1] == 0) & (seg[..., 2] == 0)
            
            ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
            ori_image = torch.from_numpy(ori_image).permute(2,0,1).float() / 255.0
            
            # torso images 
            torso_img = cv2.imread(torso_image_path, cv2.IMREAD_UNCHANGED) # [H, W, 4]
            torso_img = cv2.cvtColor(torso_img, cv2.COLOR_BGRA2RGBA)
            torso_img = torso_img.astype(np.float32) / 255 # [H, W, 3/4]
            
        else:
            gt_image = None
            ori_image = None
            torso_img = None
            seg = None
        

        cam_infos.append(CameraInfo(uid=jdx, R=R, T=T, FovY=FovY, FovX=FovX, full_image=ori_image, full_image_path=full_image_path,
                        gt_image=gt_image, gt_image_path=gt_image_path,
                        image_name=image_name, width=contents["w"], height=contents["h"],
                        torso_image=torso_img, torso_image_path=torso_image_path, bg_image=bg_img, bg_image_path=bg_image_path,
                        mask=seg, mask_path=mask_path, trans=trans_infos[idx],
                        face_rect=face_rect, lhalf_rect=lhalf_rect, aud_f=aud_feature, eye_f=eye_area, eye_rect=eye_rect, lips_rect=lips_rect))
    return cam_infos     


def readTalkingPortraitDatasetInfo(path, white_background, eval, extension=".jpg",custom_aud=None):
    # # Audio Information
    # aud_features = np.load(os.path.join(path, 'aud_ds.npy'))
    # aud_features = torch.from_numpy(aud_features)

    # # support both [N, 16] labels and [N, 16, K] logits
    # if len(aud_features.shape) == 3:
    #     aud_features = aud_features.float().permute(0, 2, 1) # [N, 16, 29] --> [N, 29, 16]    
    # else:
    #     raise NotImplementedError(f'[ERROR] aud_features.shape {aud_features.shape} not supported')

    aud_path = os.path.join(path, 'aud.wav')
    aud_ave_path = os.path.join(path, 'aud_ave.npy')
    if os.path.exists(aud_ave_path):
        aud_features = np.load(aud_ave_path)
    else:
        from scene.utils import AudDataset
        from scene.networks import AudioEncoder
        from torch.utils.data import DataLoader
        # ave aud features
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AudioEncoder().to(device).eval()
        ckpt = torch.load('/root/chat2video/sync-gaussian-talker/scene/checkpoints/audio_visual_encoder.pth')
        model.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt.items()})
        dataset = AudDataset(aud_path)
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        outputs = []
        for mel in data_loader:
            mel = mel.to(device)
            with torch.no_grad():
                out = model(mel)
            outputs.append(out)
        outputs = torch.cat(outputs, dim=0).cpu()
        first_frame, last_frame = outputs[:1], outputs[-1:]
        aud_features = torch.cat([first_frame.repeat(2, 1), outputs, last_frame.repeat(2, 1)],
                                    dim=0).unsqueeze(1)
        np.save(aud_ave_path, aud_features.detach().cpu().numpy())
    print(f'[INFO] load aud_features: {aud_features.shape}')
    
    # load action units
    # import pandas as pd
    # au_blink_info=pd.read_csv(os.path.join(path, 'au.csv'))
    # eye_features = au_blink_info[' AU45_r'].values
    bs_path = os.path.join(path, 'bs.npy')
    au_path = os.path.join(path, 'au.csv')
    if os.path.exists(bs_path):
        bs = np.load(bs_path)
        eye_features = bs[:,8:10]
        # if self.opt.bs_area == "upper":
        #     bs = np.hstack((bs[:, 0:5], bs[:, 8:10]))
        # elif self.opt.bs_area == "single":
        #     bs = np.hstack((bs[:, 0].reshape(-1, 1),bs[:, 2].reshape(-1, 1),bs[:, 3].reshape(-1, 1), bs[:, 8].reshape(-1, 1)))
    elif os.path.exists(au_path):
        import pandas as pd
        au_blink_info=pd.read_csv(os.path.join(path, 'au.csv'))
        eye_features = au_blink_info[' AU45_r'].values
    else:
        print("bs.npy or au.npy not found, have no eye features")
        eye_features = np.zeros(aud_features.shape[0])

    if custom_aud:
        aud_features = np.load(os.path.join(path, custom_aud))
        aud_features = torch.from_numpy(aud_features)
        if len(aud_features.shape) == 3:
            aud_features = aud_features.float().permute(0, 2, 1) # [N, 16, 29] --> [N, 29, 16]    
        else:
            raise NotImplementedError(f'[ERROR] aud_features.shape {aud_features.shape} not supported')
        print("Reading Custom Transforms")
        custom_cam_infos = readCamerasFromTracksTransforms(path, "track_params.pt", "transforms_val.json", aud_features, eye_features, extension, 
                                                           timestamp_mapper,custom_aud=custom_aud)
    else:
        custom_cam_infos=None
    
    ply_path = os.path.join(path, "fused.ply")
    mesh_path = os.path.join(path, "track_params.pt")
    
    timestamp_mapper, max_time = read_timeline(path)
    print("Use ER_NeRF dataset")
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTracksTransforms(path, "track_params.pt", "transforms_train.json", aud_features, eye_features, extension, timestamp_mapper, preload = False)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTracksTransforms(path, "track_params.pt", "transforms_val.json", aud_features, eye_features, extension, timestamp_mapper, preload=False)
    print("Generating Video Transforms")
    video_cam_infos = None 
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        # num_pts = 2000
        # print(f"Generating random point cloud ({num_pts})...")
        # We create random points inside the bounds of the synthetic Blender scenes
        # Initialize with 3DMM Vertices
        facial_mesh = torch.load(mesh_path).get("vertices")
        if facial_mesh is None:
            num_pts = 20_000
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            # assert False, "track_params.pt have no vertices"
            print(f"Generating random point cloud ({num_pts})...")
        else:
            average_facial_mesh = torch.mean(facial_mesh, dim=0)
            xyz = average_facial_mesh.cpu().numpy()
        shs = np.random.random((xyz.shape[0], 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((xyz.shape[0], 3)))        
    else:
        pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           custom_cameras=custom_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time
                           )
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", pre_load=False):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        content_keys = contents.keys()
        if "cx" in content_keys and "focal_len" in content_keys:
            half_pixels = contents["cx"]
            focal = contents["focal_len"]
            fovx = 2*math.atan(half_pixels/focal)
            h = int(2*contents["cy"])
            w = int(2*contents["cx"])

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            if "file_path" in frame:
                cam_name = os.path.join(path, frame["file_path"] + extension)
            elif "img_id" in frame:
                cam_name = os.path.join(path, f"images/{frame['img_id']}{extension}")

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            if pre_load:
                image = Image.open(image_path)
                w, h = image.size
                im_data = np.array(image.convert("RGBA"))
                bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            else:
                bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
                image = None            

            fovy = focal2fov(fov2focal(fovx, w), h)
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=w, height=h, bg=bg))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".jpg"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "ER-NeRF": readTalkingPortraitDatasetInfo,
    "Blender" : readNerfSyntheticInfo,
    "SyncTalk": readSyncTalkDatasetInfo,
}
