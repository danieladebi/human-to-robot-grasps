import gymnasium as gym 
import robosuite
from robosuite.controllers import load_controller_config

import numpy as np
import os
import cv2

from robosuite.environments.base import register_env

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.save_util import save_to_zip_file, load_from_zip_file
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from VIPWrapper import VIPWrapper
from VIPGoalLoader import VIPGoalLoader
from vip import load_vip
import argparse

from stable_baselines3.common.callbacks import ProgressBarCallback


# task_name_to_env_name_map = {"lift" : "Lift",
#                              "door" : "Door",
#                              "nut" : "NutAssembly",
#                              "pick": "PickPlace",
#                              "stack": "Stack"}

task_name_to_env_name_map = {"lift" : "Lift",
                             "can" : "Can",
                             "square" : "Square",
                             "transport": "Transport",
                             "tool_hang": "Tool_Hang"}


def trainer(args):
    controller_config = load_controller_config(default_controller="OSC_POSE")

    task_name = args.task
    task_name_to_env_name = task_name_to_env_name_map[task_name] #task_name[0].upper() + task_name[1:]
    # rs_env = robosuite.make(
    #     task_name_to_env_name,
    #     robots="Panda",
    #     gripper_types="default",
    #     controller_configs=controller_config,
    #     has_renderer=False,
    #     render_camera="frontview",
    #     has_offscreen_renderer=True,
    #     control_freq=20,
    #     horizon=2000,
    #     use_object_obs=False,
    #     use_camera_obs=True,
    #     camera_names="agentview",
    #     camera_heights=84,
    #     camera_widths=84,
    #     reward_shaping=True
    # )

    vip_goal_loader = VIPGoalLoader(task_name)
    vip_goal_loader.load_dataset()
    vip_goal = vip_goal_loader.get_random_goal_image()

    # Might be useful later
    env_meta = FileUtils.get_env_metadata_from_dataset(vip_goal_loader.processed_dataset_path)

    rs_env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta["env_name"],
        render=False,
        render_offscreen=True,
        use_image_obs=True,
    )
    rs_env = rs_env.env
            
    vip_model = load_vip()
    vip_model.eval()
    env = VIPWrapper(rs_env, vip_model, vip_goal,
                     use_vip_embedding_obs=args.use_vip_embedding_obs,
                     use_hand_pose_obs=args.use_hand_pose_obs,
                     use_vip_reward=args.use_vip_reward,
                     vip_reward_type=args.vip_reward_type,
                     vip_reward_min=args.vip_reward_min,
                     vip_reward_max=args.vip_reward_max,
                     vip_reward_interval=args.vip_reward_interval)

    def wrap_env(env):
        wrapped_env = Monitor(env) # needed for extracting eprewmean and eplenmean
        wrapped_env = DummyVecEnv([lambda: wrapped_env]) # Needed for all environments (e.g. used for multi-processing)
        # getting some gym box error when uncommenting below
        wrapped_env = VecNormalize(wrapped_env) # Needed for improving training when using MuJoCo envs?
        return wrapped_env

    #wrap_env(env)
    # get the number of folders in the trained_models directory
    num_models = len(os.listdir('trained_models'))
    base_folder = os.path.join('trained_models', task_name)
    filepath = os.path.join(base_folder, str(num_models))

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"./ppo_{task_name}_tensorboard/") # PPO
    model.learn(total_timesteps=3e5, tb_log_name=filepath, progress_bar=True) #  3e5, tb_log_name=filename)

    model.save(filepath)
    # let's also save the command line arguments as csv
    args_filepath = filepath + '_args.csv'
    with open(args_filepath, 'w') as f:
        for arg in vars(args):
            f.write("%s,%s\n"%(arg,getattr(args, arg)))
    
    # env.save('trained_models/vec_normalize_' + filename + '.pkl') # Save VecNomralize statistics

    rs_test_env = robosuite.make(
            task_name_to_env_name,
            robots="Panda",
            gripper_types="default",
            controller_configs=controller_config,
            has_renderer=False,
            render_camera="frontview",
            has_offscreen_renderer=True,
            control_freq=20,
            horizon=2000,
            use_object_obs=False,
            use_camera_obs=True,
            camera_names="agentview",
            camera_heights=84,
            camera_widths=84,
            reward_shaping=True
        )
    env_test = VIPWrapper(rs_test_env, vip_model, vip_goal,
                          use_vip_embedding_obs=args.use_vip_embedding_obs,
                        use_hand_pose_obs=args.use_hand_pose_obs,
                        use_vip_reward=False,
                        vip_reward_type=args.vip_reward_type,
                        vip_reward_min=args.vip_reward_min,
                        vip_reward_max=args.vip_reward_max,
                        vip_reward_interval=args.vip_reward_interval)

    model = PPO.load(filepath)
    env = DummyVecEnv([lambda : env_test])
    # env = VecNormalize.load("trained_models/vec_normalize_" + filename + ".pkl", env)

    env.training = False
    env.norm_reward = False

    obs = env.reset()
    image_folder = os.path.join(filepath, 'images')
    os.makedirs(image_folder, exist_ok=True)

    i = 0
    total_r = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_r += reward
        image_filepath = os.path.join(image_folder, 'img' + str(i) + '.png')
        cv2.imwrite(image_filepath, env.envs[0].gym_env.latest_obs_dict['agentview_image'])
        i += 1

        if done: 
            obs = env.reset()
            break
    # save images as video
    image_files = image_folder + '/*.png'
    frame_rate = 30 
    os.system('ffmpeg -pattern_type glob -r ' + str(frame_rate) + ' -i ' + '\'' + image_files + '\'' + ' -vcodec mpeg4 -y ' + filepath + '/demo.mp4')

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='lift')
    parser.add_argument('--seed', type=int, default=0)
    
    # vip arguments
    parser.add_argument('--use_vip_embedding_obs', type=bool, default=True)
    parser.add_argument('--use_hand_pose_obs', type=bool, default=True)
    parser.add_argument('--use_vip_reward', type=bool, default=True)
    parser.add_argument('--vip_reward_type', type=str, default='add')
    parser.add_argument('--vip_reward_min', type=float, default=-1)
    parser.add_argument('--vip_reward_max', type=float, default=1)
    parser.add_argument('--vip_reward_interval', type=int, default=1)
    
    args = parser.parse_args()
    trainer(args)
    
    