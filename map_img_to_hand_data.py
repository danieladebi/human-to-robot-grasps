import os
import pickle
import torch
import torchvision
from tqdm import tqdm
import re

import argparse

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

    data = [(img_dict[id], hand_pose_dict[id]["pred_output_list"][0]["right_hand"]) for id in hand_pose_dict.keys()] 