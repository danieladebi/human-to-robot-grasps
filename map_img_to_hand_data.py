import os
import pickle
import torch
import torchvision
import pickle

from torch.utils.data import Dataset, DataLoader
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

        self.handpose_ids.sort()
        self.img_ids.sort()

        assert(self.handpose_ids == self.img_ids)
        for hp, img in zip(self.handpose_ids, self.img_ids):
            assert(hp == img)
        

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

    dataset = HandPoseDataset(mocap_dir, rendered_dir)
    print(dataset[0])

    # hand_pose_dict, img_dict = load_data(mocap_dir, rendered_dir)
    # count = 0
    # empty = 0
    # for i in hand_pose_dict:
    #     d = hand_pose_dict[i]['pred_output_list'][0]["right_hand"]
    #     if d:
    #         count += 1
    #     else: 
    #         empty += 1
    # print(count, empty)
    # print(hand_pose_dict['0000010000']["pred_output_list"][0]["right_hand"])

    # data = [(img_dict[id], hand_pose_dict[id]["pred_output_list"][0]["right_hand"]) for id in hand_pose_dict.keys()] 

    # Learn model that gets directly from image to hand joints
    