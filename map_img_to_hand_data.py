import os
import pickle
import torch
import torchvision

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import re

import argparse

class HandPoseDataset(Dataset):
    def __init__(self, mocap_dir, rendered_dir, transform=None):
        self.pose_dir = mocap_dir
        self.img_dir = rendered_dir
        self.transform = transform

        self.load_data()
    
    def load_data(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

def load_data(mocap_dir, rendered_dir):
    hand_poses = {}
    imgs = {}

    for pkl_filename in tqdm(os.listdir(mocap_dir)):
        pkl_filepath = os.path.join(mocap_dir, pkl_filename)
        id = re.findall(r'\d+', pkl_filename)[0]
        with open(pkl_filepath, "rb") as f:
            try:
                hand_poses[id] = pickle.load(f)
            except UnicodeDecodeError:
                f.seek(0)
                hand_poses[id] = pickle.load(f, encoding='latin1')


    for img_filename in tqdm(os.listdir(rendered_dir)):
        img_filepath = os.path.join(rendered_dir, img_filename)
        img = torchvision.io.read_image(img_filepath)
        id = re.findall(r'\d+', img_filename)[0]
        imgs[id] = img
            
    return hand_poses, imgs

if __name__ == "__main__":
    mocap_dir = "/home/ida338/frankmocap/demo/results/mocap"
    rendered_dir = "/home/ida338/frankmocap/demo/results/rendered"

    hand_pose_dict, img_dict = load_data(mocap_dir, rendered_dir)
    count = 0
    empty = 0
    for i in hand_pose_dict:
        d = hand_pose_dict[i]['pred_output_list'][0]["right_hand"]
        if d:
            count += 1
        else: 
            empty += 1
    print(count, empty)
    print(hand_pose_dict['0000010000']["pred_output_list"][0]["right_hand"])

    data = [(img_dict[id], hand_pose_dict[id]["pred_output_list"][0]["right_hand"]) for id in hand_pose_dict.keys()] 

    # Learn model that gets directly from image to hand joints
    