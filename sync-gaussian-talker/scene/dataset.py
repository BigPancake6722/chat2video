from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils_talker.general_utils import PILtoTorch
from utils_talker.graphics_utils import fov2focal, focal2fov
import torch
from utils_talker.camera_utils import loadCam
from utils_talker.graphics_utils import focal2fov
from .talking_dataset_readers import CameraInfo
import os

import cv2

import sys

def create_transformation_matrix(R, T):
    T_homogeneous = np.hstack((R, T.reshape(-1, 1)))  # Concatenate R and T horizontally
    T_homogeneous = np.vstack((T_homogeneous, [0, 0, 0, 1]))  # Add the last row for homogeneous coordinates
    return T_homogeneous

class FourDGSdataset():
    def __init__(
        self,
        dataset,
        args,
        dataset_type,
        lazy_load=True
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
        self.lazy_load = lazy_load
        
    def mirro_index(self, index):
        leng = len(self) - 1
        len2 = leng * 2
        remainder = index % len2
        half = remainder // leng
        minus = remainder % leng
        if half == 0:
            return remainder
        else:
            return leng - minus
    
    def collate(self, index):    
        caminfo:CameraInfo = self.dataset[index]
        R = caminfo.R  # (3, 3)
        T = caminfo.T
        FovX = caminfo.FovX
        FovY = caminfo.FovY
        trans = caminfo.trans.cpu().numpy()
        mask = caminfo.mask
        
        full_image = caminfo.full_image
        if full_image is None:
            full_image = cv2.imread(caminfo.full_image_path, cv2.IMREAD_UNCHANGED)
            full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
            full_image = torch.from_numpy(full_image).permute(2,0,1).float() / 255.0
            
        gt_image = caminfo.gt_image
        if gt_image is None:
            gt_image = cv2.imread(caminfo.gt_image_path, cv2.IMREAD_UNCHANGED)
            gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
            gt_image = torch.from_numpy(gt_image).permute(2,0,1).float() / 255.0
            
        torso_image = caminfo.torso_image
        if torso_image is None:
            torso_image = cv2.imread(caminfo.torso_image_path, cv2.IMREAD_UNCHANGED) # [H, W, 4]
            torso_image = cv2.cvtColor(torso_image, cv2.COLOR_BGRA2RGBA)
            torso_image = torso_image.astype(np.float32) / 255 # [H, W, 3/4]
            torso_image = torch.from_numpy(torso_image) # [3/4, H, W]
            torso_image = torso_image.permute(2, 0, 1)
            
        bg_image = caminfo.bg_image
        if bg_image is None:
            bg_img = cv2.imread(caminfo.bg_image_path, cv2.IMREAD_UNCHANGED) # [H, W, 3]
            if bg_img.shape[0] != caminfo.height or bg_img.shape[1] != caminfo.width:
                bg_img = cv2.resize(bg_img, (caminfo.width, caminfo.height), interpolation=cv2.INTER_AREA)
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            bg_image = torch.from_numpy(bg_img).permute(2,0,1).float() / 255.0
        
        seg = caminfo.mask
        if seg is None:
            seg = cv2.imread(caminfo.mask_path)
            face_mask = cv2.imread(caminfo.full_image_path.replace('ori_imgs', 'face_mask').replace('.jpg', '.png'))
        # BGR -> head / neck / torso
        # BGR (head and neck)
        head_mask = (seg[..., 0] == 255) & (seg[..., 1] == 0) & (seg[..., 2] == 0)
        neck_mask = (seg[..., 0] == 0) & (seg[..., 1] == 255) & (seg[..., 2] == 0)
        torso_mask = (seg[..., 0] == 0) & (seg[..., 1] == 0) & (seg[..., 2] == 255)
        face_mask = (face_mask[:, :, 0]==255) & (face_mask[:, :, 1]==255) & (face_mask[:, :, 2]==255)
        seg_mask = torch.from_numpy(np.stack([head_mask, neck_mask, torso_mask, face_mask], axis=-1)).permute(2,0,1).float()
        
        bg_w_torso = torso_image[:3,...] * torso_image[3:,...] + bg_image * (1-torso_image[3:,...])
                    
        face_rect = caminfo.face_rect
        lhalf_rect = caminfo.lhalf_rect
        eye_rect = caminfo.eye_rect
        lips_rect = caminfo.lips_rect
        
        
        return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,orig_image=full_image, gt_image=gt_image, seg_mask=seg_mask, bg_image = bg_image,
                image_name=caminfo.image_name,uid=index,data_device=torch.device("cuda"), #trans=trans,
                aud_f = caminfo.aud_f, eye_f = caminfo.eye_f, face_rect=face_rect, lhalf_rect=lhalf_rect, 
                eye_rect=eye_rect, lips_rect=lips_rect, bg_w_torso=bg_w_torso,
                torso_image=torso_image)
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            start, end = index.start, index.stop
            start = 0 if start is None else start
            end = len(self) if end is None else end
            return [self[self.mirro_index(i)] for i in range(start, end)]
        if index >= len(self):
            raise IndexError("index out of range")
        if self.lazy_load:
            return lambda: self.collate(index)
        else:
            return self.collate(index)
        
    
    def __len__(self):
        return len(self.dataset)
