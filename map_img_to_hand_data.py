import os
import pickle
import torch
import torchvision
import pickle
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.io import read_image

import re
from tqdm import tqdm

import argparse

class HandPoseDataset(Dataset):
    def __init__(self, mocap_dir, rendered_dir, transform=None):
        self.pose_dir = mocap_dir
        self.img_dir = rendered_dir
        self.transform = transform

        self.img_ids = []  
        self.handpose_ids = []

        self.get_file_ids()
    
    def get_file_ids(self):

        for pkl_filename in tqdm(os.listdir(self.pose_dir)):
            pkl_filepath = os.path.join(self.pose_dir, pkl_filename)
            id = re.findall(r'\d+', pkl_filename)[0]

            self.handpose_ids.append(id) 

        for img_filename in tqdm(os.listdir(self.img_dir)):
            img_filepath = os.path.join(self.img_dir, img_filename)
            id = re.findall(r'\d+', img_filename)[0]
            
            self.img_ids.append(id)

        ids = list(set(self.handpose_ids) & set(self.img_ids))

        self.handpose_ids = sorted(ids)
        self.img_ids = sorted(ids)

        assert(self.handpose_ids == self.img_ids)
        for hp_id, img_id in zip(self.handpose_ids, self.img_ids):
            assert(hp_id == img_id)
        

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id, hand_pose_id = self.img_ids[idx], self.handpose_ids[idx]
        img_filename = f"frame_{img_id}.jpg"
        img_filepath = os.path.join(self.img_dir, img_filename)
        
        img = read_image(img_filepath)

        pkl_filename = f"frame_{hand_pose_id}_prediction_result.pkl"
        pkl_filepath = os.path.join(self.pose_dir, pkl_filename)
        with open(pkl_filepath, "rb") as f:
            try:
                hand_pose_dict = pickle.load(f)
            except UnicodeDecodeError:
                f.seek(0)
                hand_pose_dict = pickle.load(f, encoding='latin1')

        hand_pose = hand_pose_dict["pred_output_list"][0]["right_hand"]["pred_joints_img"]

        if self.transform:
            img = self.transform(img)

        return img, hand_pose


if __name__ == "__main__":
    mocap_dir = "/mnt/zhang-nas/ikadebi/handpose-data/mocap"
    rendered_dir = "/mnt/zhang-nas/ikadebi/handpose-data/imgs"

    dataset = HandPoseDataset(mocap_dir, rendered_dir)
    print(len(dataset))
    plt.imshow(dataset[0][0].permute(1,2,0))
    print(dataset[0][0].shape)
    plt.savefig("sample.png")

    count = 0 
    for i in tqdm(range(len(dataset))):
        if dataset[1]:
            count += 1

    print(count)
