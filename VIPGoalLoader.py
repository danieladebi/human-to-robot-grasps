import os
import json
import h5py
import numpy as np
import cv2

import robomimic
import robomimic.utils.file_utils as FileUtils

# the dataset registry can be found at robomimic/__init__.py
from robomimic import DATASET_REGISTRY

# import all utility functions

import numpy as np

import torch
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.test_utils as TestUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.utils.dataset import SequenceDataset

from robomimic.config import config_factory
from robomimic.algo import algo_factory
from robomimic.scripts.dataset_states_to_obs import dataset_states_to_obs

from collections import namedtuple

# This file is mostly adapting the robomimic Getting Started tutorial
# https://robomimic.github.io/docs/datasets/robomimic_v0.1.html

class VIPGoalLoader:
    def __init__(self, task_name):
        self.download_folder = "robomimic_data/" + task_name + '/'
        self.dataset_path = os.path.join(self.download_folder, "demo_v141.hdf5")
        self.processed_dataset_path = os.path.join(self.download_folder, "image_dense_v141.hdf5")
        self.task_name = task_name
    
    def get_data_loader(self, dataset_path):
        """
        Get a data loader to sample batches of data.
        Args:
            dataset_path (str): path to the dataset hdf5
        """
        dataset = SequenceDataset(
            hdf5_path=dataset_path,
            obs_keys=(                      # observations we want to appear in batches
                # "robot0_hand image (I forgot the name lol), but there is another camera",
                "agentview_image",
            ),
            dataset_keys=(                  # can optionally specify more keys here if they should appear in batches
                "actions",
                "rewards",
                "dones",
            ),
            load_next_obs=True,
            frame_stack=1,
            seq_length=1,                  
            pad_frame_stack=True,
            pad_seq_length=True,            # pad last obs per trajectory to ensure all sequences are sampled
            get_pad_mask=False,
            goal_mode='last',
            hdf5_cache_mode="all",          # cache dataset in memory to avoid repeated file i/o
            hdf5_use_swmr=True,
            hdf5_normalize_obs=False,
            filter_by_attribute=None,       # can optionally provide a filter key here
        )
        print("\n============= Created Dataset =============")
        print(dataset)
        print("")

        data_loader = DataLoader(
            dataset=dataset,
            sampler=None,       # no custom sampling logic (uniform sampling)
            batch_size=100,     # batches of size 100
            shuffle=True,
            num_workers=0,
            drop_last=True      # don't provide last batch in dataset pass if it's less than 100 in size
        )
        return data_loader
    
    def download_dataset(self):
        # set download folder and make it
        os.makedirs(self.download_folder, exist_ok=True)
        if not os.path.exists(self.dataset_path):
            # download the dataset
            task = self.task_name
            dataset_type = "ph"
            hdf5_type = "raw"
            FileUtils.download_url(
                url=DATASET_REGISTRY[task][dataset_type][hdf5_type]["url"],
                download_dir=self.download_folder,
            )

            # enforce that the dataset exists
            assert os.path.exists(self.dataset_path)

        if not os.path.exists(self.processed_dataset_path):
            Args = namedtuple('Args', ['done_mode', 'shaped', 'dataset', 
                                    'output_name', 'camera_names', 
                                    'camera_height', 'camera_width',
                                    'n', 'depth', 
                                    'copy_rewards', 'copy_dones', 
                                    'exclude_next_obs', 'compress'])
            args = Args(done_mode=0, shaped=True, dataset=self.dataset_path, 
                        output_name="image_dense_v141.hdf5", 
                        camera_names=['agentview', 'robot0_eye_in_hand'],
                        camera_height=84, camera_width=84,
                        n=None, depth=False,
                        copy_rewards=False, copy_dones=False,
                        exclude_next_obs=False, compress=False)
            # NOTE: For code below, I had to do extra steps
            # I had to do this: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/cluster/mmunje/projects/vip/human-to-robot-grasps/mujoco210/bin
            # and this: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
            # also did: pip install "cython<3"
            dataset_states_to_obs(args)
            assert os.path.exists(self.processed_dataset_path)
            
    def load_dataset(self):
        self.download_dataset()
        self.dataset = self.get_data_loader(self.processed_dataset_path)
        self.data_loader_iter = iter(self.dataset)
        
    def get_random_goal_image(self):
        next_sample = next(self.data_loader_iter)
        img = next_sample['goal_obs']['agentview_image'][0].numpy()
        # image should be upside down
        img = np.flipud(img)
        # save img to debug
        rgb_img = np.copy(img)
        # swap blues and reds
        rgb_img[:, :, [0, 2]] = rgb_img[:, :, [2, 0]]
        goal_filepath = f'goal_{self.task_name}.png'
        cv2.imwrite(goal_filepath, rgb_img)
        return img