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
import random
import json
from utils_talker.system_utils import searchForMaxIteration
# from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.talking_dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.dataset import FourDGSdataset
from arguments import ModelParams
from utils_talker.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.utils.data import Dataset
from scene.dataset_readers import add_points
import numpy as np
import glob

from torchvision.utils import save_image
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, 
                 shuffle=True, resolution_scales=[1.0], load_coarse=False, custom_aud=None, lazy_load=True,
                 train_torso=False, eval_torso=False, smooth_win=2, start_frame=0):
        """b
        :param path: Path to colmap scene main folder.
        """
        print("scene_init", custom_aud.shape)
        self.args = args
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.lazy_load = lazy_load
        
        # if load_iteration:
        #     if load_iteration == -1:
        #         self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
        #     else:
        #         self.loaded_iter = load_iteration
        # if not self.loaded_iter is None:       
        #     print("Loading trained model at iteration {}".format(self.loaded_iter))
        if load_iteration:
            self.load(load_iteration, "fine")
            args = self.args

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}
        self.torso_camera = {}

        # dataset_type = "ER-NeRF"
        dataset_type = "SyncTalk"
        self.scene_info = scene_info = sceneLoadTypeCallbacks[dataset_type](args.source_path, False, args.eval, 
                                                                            custom_aud=custom_aud, train_torso=train_torso, 
                                                                            eval_torso=eval_torso, smooth_win=smooth_win,
                                                                            start_idx=start_frame)
        
        self.maxtime = scene_info.maxtime
        self.dataset_type = dataset_type
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        if not (len(custom_aud.shape) == 0 or all(s == 0 for s in custom_aud.shape)):
            print("Loading Custom Cameras")
            self.custom_camera = FourDGSdataset(scene_info.custom_cameras, args, dataset_type, lazy_load=lazy_load)
        else:
            if not self.loaded_iter:
                # 将cameras写入json
                json_cams = []
                camlist = []
                if scene_info.test_cameras:
                    camlist.extend(scene_info.test_cameras)
                if scene_info.train_cameras:
                    camlist.extend(scene_info.train_cameras)
                for id, cam in enumerate(camlist):
                    json_cams.append(camera_to_JSON(id, cam))
                with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                    json.dump(json_cams, file)
                    
            print("Loading Training Cameras")
            self.train_camera = FourDGSdataset(scene_info.train_cameras, args, dataset_type, lazy_load=lazy_load)
            print("Loading Test Cameras")
            self.test_camera = FourDGSdataset(scene_info.test_cameras, args, dataset_type, lazy_load=lazy_load)
        if eval_torso:
            self.torso_camera = FourDGSdataset(scene_info.torso_cameras, args, dataset_type, lazy_load=lazy_load)
            
        xyz_max = scene_info.point_cloud.points.max(axis=0)
        xyz_min = scene_info.point_cloud.points.min(axis=0)
        if args.add_points:
            print("add points.")
            scene_info = scene_info._replace(point_cloud=add_points(scene_info.point_cloud, xyz_max=xyz_max, xyz_min=xyz_min))
        self.gaussians._deformation.deformation_net.set_aabb(xyz_max,xyz_min)
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.load_model(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                   ))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime)
        
    def getTrainCameras(self, scale=1.0):
        return self.train_camera
    def getTestCameras(self, scale=1.0):
        return self.test_camera
    def getCustomCameras(self, scale=1.0):
        return self.custom_camera
    def getTorsoCameras(self, scale=1.0):
        return self.torso_camera    

    def save(self, iteration, stage, image=None, image_idx = None, args=None):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))

        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)
        
        if image is not None:
            save_image(image, os.path.join(point_cloud_path, str(image_idx) + ".png"))
            
        if not args is None:
            if not isinstance(args, dict):
                args = args.__dict__
            with open(os.path.join(point_cloud_path, "arguments.json"), 'w') as fp:
                json.dump(args, fp)
    
    def load(self, iteration, stage):
        if stage == "coarse":
            if iteration == -1:
                iteration = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), "coarse_iteration")
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))
        else:
            if iteration == -1:
                iteration = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), "iteration")
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        if not os.path.exists(point_cloud_path):
            print("Point cloud not found at {}".format(point_cloud_path))
            return False
        self.gaussians.load_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.load_model(os.path.join(point_cloud_path))
        if os.path.exists(os.path.join(point_cloud_path, "arguments.json")):
            with open(os.path.join(point_cloud_path, "arguments.json"), 'r') as file:
                args = json.load(file)
                for k in args:
                    setattr(self.args, k, args[k])
        print("Loaded point cloud from {}".format(point_cloud_path))
        self.loaded_iter = iteration
        return True
    
def gaussian_model_load(gaussians : GaussianModel, model_path, load_iteration=-1):
    model_path = model_path
    gaussians = gaussians
    xyz_max = np.ones(3)
    xyz_min = -np.ones(3)
    gaussians._deformation.deformation_net.set_aabb(xyz_max, xyz_min)
    loaded_iter = None
    if load_iteration:
        if load_iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
        else:
            loaded_iter = load_iteration
    if not loaded_iter is None:       
        print("Loading trained model at iteration {}".format(loaded_iter))
        gaussians.load_ply(os.path.join(model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(loaded_iter),
                                                        "point_cloud.ply"))
        gaussians.load_model(os.path.join(model_path,
                                                "point_cloud",
                                                "iteration_" + str(loaded_iter),
                                                ))
    else:
        print("have no trained model")
        