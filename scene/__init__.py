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
import numpy as np
import torch
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.ref_gaussian_model import RefGaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


class Scene:

    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel,
                 load_iteration=None, shuffle=True, resolution_scales=[1.0]):

        self.model_path = args.model_path
        self.batch_size = args.batch_size
        self.loaded_iter = None
        self.gaussians = gaussians

        self.ray_device = torch.device("cpu")

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print(f"Loading trained model at iteration {self.loaded_iter}")

        self.train_cameras = {}
        self.test_cameras = {}

        self.light_rotate = False

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval
            )

        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):

            if "blender_LDR" in args.source_path:
                print("Found keyword blender_LDR, assuming Stanford ORB data set!")
                scene_info = sceneLoadTypeCallbacks["StanfordORB"](
                    args.source_path, args.white_background, args.eval
                )

            elif "Synthetic4Relight" in args.source_path:
                print("Found Synthetic4Relight, assuming Synthetic4Relight data set!")
                scene_info = sceneLoadTypeCallbacks["Synthetic4Relight"](
                    args.source_path, args.white_background, args.eval
                )
                self.light_rotate = True

            elif "TensoIR" in args.source_path:
                print("Found transforms_train.json file, assuming TensoIR data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](
                    args.source_path, args.white_background, args.eval
                )
                self.light_rotate = True

            else:
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](
                    args.source_path, args.white_background, args.eval
                )
        else:
            raise ValueError("Could not recognize scene type!")

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file:
                with open(os.path.join(self.model_path, "input.ply"), 'wb') as dest_file:
                    dest_file.write(src_file.read())

            json_cams = []
            camlist = []

            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)

            for idx, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(idx, cam))

            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args
            )

            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )

        # ------------------------------------------------------------
        # MEMORY-SAFE RAY STORAGE (CRITICAL CHANGE)
        # ------------------------------------------------------------
        self.train_rays = {}
        self.train_ray_sizes = {}

        for resolution_scale in resolution_scales:

            print("Building training rays (CPU-streamed safe mode)")

            rays_o_list = []
            rays_d_list = []
            rays_rgb_list = []

            for cam in self.train_cameras[resolution_scale]:
                rays_o, rays_d = cam.get_rays()
                rays_rgb = cam.get_rays_rgb()

                rays_o_list.append(rays_o)
                rays_d_list.append(rays_d)
                rays_rgb_list.append(rays_rgb)

            rays_o = torch.cat(rays_o_list, dim=0).contiguous().cpu()
            rays_d = torch.cat(rays_d_list, dim=0).contiguous().cpu()
            rays_rgb = torch.cat(rays_rgb_list, dim=0).contiguous().cpu()

            self.train_rays[resolution_scale] = (rays_o, rays_d, rays_rgb)
            self.train_ray_sizes[resolution_scale] = rays_o.shape[0]

        # ------------------------------------------------------------

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    f"iteration_{self.loaded_iter}",
                    "point_cloud.ply"
                )
            )
        else:
            self.gaussians.create_from_pcd(
                scene_info.point_cloud,
                self.cameras_extent,
                args
            )

    def save(self, iteration):
        point_cloud_path = os.path.join(
            self.model_path,
            f"point_cloud/iteration_{iteration}"
        )
        self.gaussians.save_ply(
            os.path.join(point_cloud_path, "point_cloud.ply")
        )

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    # ------------------------------------------------------------
    # MEMORY-SAFE BATCH SAMPLER (GPU ONLY PER BATCH)
    # ------------------------------------------------------------
    def get_batch_rays(self, scale=1.0, device="cuda"):
        train_rays_o, train_rays_d, train_rays_rgb = self.train_rays[scale]

        n = train_rays_o.shape[0]
        batch_size = self.batch_size

        ray_id = torch.randint(0, n, (batch_size,))

        rays_o = train_rays_o[ray_id].to(device, non_blocking=True)
        rays_d = train_rays_d[ray_id].to(device, non_blocking=True)
        rays_rgb = train_rays_rgb[ray_id].to(device, non_blocking=True)

        return rays_o, rays_d, rays_rgb