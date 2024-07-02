import os
import torch
import numpy as np

import utils.general as utils
from utils import rend_util

import cv2

class MultiSceneDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, img_res, scan_ids):
        """
        Initialize the MultiSceneDataset.
        
        :param data_dir: Root directory containing all scene data
        :param img_res: Tuple of (height, width) for image resolution
        :param scan_ids: List of scene IDs to include in the dataset
        """
        self.data_dir = os.path.join('../data', data_dir)
        self.img_res = img_res
        self.scan_ids = scan_ids
        self.total_pixels = img_res[0] * img_res[1]

        # Lists to store data for all scenes
        self.all_rgb_images = []
        self.all_intrinsics = []
        self.all_poses = []
        self.all_instance_dirs = []
        self.scene_sample_counts = []

        # Iterate over all scan IDs to load data for each scene
        for scan_id in scan_ids:
            instance_dir = os.path.join(self.data_dir, 'scan{0}'.format(scan_id))
            self.all_instance_dirs.append(instance_dir)

            # Load image paths
            image_dir = '{0}/image'.format(instance_dir)
            image_paths = sorted(utils.glob_imgs(image_dir))
            n_images = len(image_paths)

            # Load camera parameters
            cam_file = '{0}/cameras.npz'.format(instance_dir)
            camera_dict = np.load(cam_file)
            scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
            world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

            # Process camera parameters
            intrinsics = []
            poses = []
            for scale_mat, world_mat in zip(scale_mats, world_mats):
                P = world_mat @ scale_mat
                P = P[:3, :4]
                intrinsic, pose = rend_util.load_K_Rt_from_P(None, P)
                intrinsics.append(torch.from_numpy(intrinsic).float())
                poses.append(torch.from_numpy(pose).float())

            self.all_intrinsics.append(intrinsics)
            self.all_poses.append(poses)

            # Load and process RGB images
            rgb_images = []
            for path in image_paths:
                rgb = rend_util.load_rgb(path)
                rgb = rgb.transpose(1, 2, 0)
                rgb = cv2.resize(rgb, (self.img_res[1], self.img_res[0]), interpolation=cv2.INTER_LINEAR)
                rgb = rgb.transpose(2, 0, 1)
                rgb = rgb.reshape(3, -1).transpose(1, 0)
                rgb_images.append(torch.from_numpy(rgb).float())
            
            self.all_rgb_images.append(rgb_images)
            self.scene_sample_counts.append(n_images)

        # Total number of samples across all scenes
        self.total_samples = sum(self.scene_sample_counts)
        self.sampling_idx = None

    def __len__(self):
        """Return the total number of samples across all scenes."""
        return self.total_samples

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        :param idx: Index of the sample to retrieve
        :return: Tuple of (idx, sample dict, ground truth dict)
        """
        # Determine which scene this sample belongs to
        local_idx = idx #the local index that will indicate the position of sample within the scene
        scene_idx = 0
        while local_idx >= self.scene_sample_counts[scene_idx]:
            local_idx -= self.scene_sample_counts[scene_idx]
            scene_idx += 1

        # Generate UV coordinates
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        # Prepare sample data
        sample = {
            "uv": uv,
            "intrinsics": self.all_intrinsics[scene_idx][local_idx],
            "pose": self.all_poses[scene_idx][local_idx]
        }

        # Prepare ground truth data
        rgb = self.all_rgb_images[scene_idx][local_idx]
        ground_truth = {
            "rgb": rgb
        }

        # Apply sampling if specified
        if self.sampling_idx is not None:
            ground_truth["rgb"] = rgb[self.sampling_idx, :]
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        """
        Collate function to be used with DataLoader.
        
        :param batch_list: List of samples to collate
        :return: Tuple of collated data
        """
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))
        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        """
        Change the sampling index for pixel sampling.
        
        :param sampling_size: Number of pixels to sample (-1 for all pixels)
        """
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        """
        Get the scale matrix for the first scene.
        
        :return: Scale matrix as a numpy array
        """
        return np.load(os.path.join(self.all_instance_dirs[0], 'cameras.npz'))['scale_mat_0']