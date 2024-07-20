import os
import torch
import numpy as np

import utils.general as utils
from utils import rend_util

import cv2

class SceneDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id=0,
                 ):

        self.instance_dir = os.path.join('../data', data_dir, 'scan{0}'.format(scan_id))

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None

        image_dir = '{0}/image'.format(self.instance_dir)
        image_paths = sorted(utils.glob_imgs(image_dir))
        self.n_images = len(image_paths)

        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)

        # Extract scale matrix, otherwise identity
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) if 'scale_mat_%d' % idx in camera_dict else np.eye(4) for idx in range(self.n_images)]
        # scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)

            # print(f"shape of rgb: {rgb.shape}") # (C, H, W)
            rgb = rgb.transpose(1, 2, 0) # [H, W, C]
            rgb = cv2.resize(rgb, (self.img_res[1], self.img_res[0]), interpolation=cv2.INTER_LINEAR) # size is received as (W, H)
            rgb = rgb.transpose(2, 0, 1) # [C, H, W] as expected by the model
            # print(f"shape of rgb after resize: {rgb.shape}")

            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]
        }

        rgb = self.rgb_images[idx]

        ground_truth = {
            "rgb": rgb
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = rgb[self.sampling_idx, :]
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']
