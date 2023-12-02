import os
from stable_baselines3 import PPO, SAC, A2C, PPO2
from utils import get_vip_wrapped_env, task_name_to_env_name_map
import argparse
from stable_baselines3.common.callbacks import ProgressBarCallback
import tqdm
import cv2
import numpy as np

def load_args(model_folder):
    args_filepath = os.path.join(model_folder, 'args.csv')
    args_dict = {}
    with open(args_filepath, 'r') as f:
        for line in f.readlines():
            arg, val = line.split(',')
            val = val.strip()
            # check if it is a bool
            if val == 'True':
                val = True
            elif val == 'False':
                val = False
            else:
                try:
                    val = int(val)
                except ValueError:
                    val = val
            args_dict[arg] = val

    # convert dict to attributes
    class Args:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    return Args(**args_dict) 

def run_episode(env, model, max_horizon=1000, save_imgs=False, image_folder=None):
    obs = env.reset()
    total_r = 0
    frames = []
    done = False
    i = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_r += reward
        if save_imgs:
            img = env.venv.envs[0].env.latest_obs_dict['frontview_image']
            frames.append(img)

        if i >= max_horizon:
            break
        i += 1

    num_digits = len(str(len(frames)))
    if len(frames) > 0:
        for i, img in enumerate(frames):
            file_idx = str(i).zfill(num_digits)
            image_filepath = os.path.join(image_folder, 'img_' + file_idx + '.png')
            img = np.flipud(img)
            img[:, :, [0, 2]] = img[:, :, [2, 0]] # swap blues and reds
            cv2.imwrite(image_filepath, img)

    return total_r

def evaluator(model_folder, n_eval_eps, n_vids):
    model_filepath = os.path.join(model_folder, 'model')
    evaluation_folder = os.path.join(model_folder, 'evaluation')
    os.makedirs(evaluation_folder, exist_ok=True)
    
    training_args = load_args(model_folder)
    task_name = training_args.task
    training_args.use_vip_reward = False # let's not use the VIP reward for evaluation for equal comparison
    env = get_vip_wrapped_env(task_name, training_args, 
                              load_vec_normalize=True, model_filepath=model_filepath,
                              use_frontview=True)

    model = PPO.load(model_filepath)
    
    if args.model == 'ppo':
        model = PPO.load(model_filepath)
    elif args.model == 'a2c':
        model = A2C.load(model_filepath)
    else:
        raise Exception('Model not supported')

    env.training = False
    env.norm_reward = False

    horizon = env.venv.envs[0].env.horizon
    
    # run evaluation episodes
    print('Running evaluation episodes...')
    total_rewards = []
    for i in tqdm.tqdm(range(n_eval_eps)):
        total_r = run_episode(env, model, max_horizon=horizon)
        total_rewards.append(total_r)
    print(f'Average reward over {n_eval_eps} episodes: {sum(total_rewards)/len(total_rewards)}')

    # save this data into csv
    csv_filepath = os.path.join(evaluation_folder, 'reward_eval.csv')
    with open(csv_filepath, 'w') as f:
        f.write('episode,reward\n')
        for i, r in enumerate(total_rewards):
            f.write(f'{i},{r}\n')
    
    print('Saving videos...')
    # save videos
    for i in tqdm.tqdm(range(n_vids)):
        num_digits = len(str(horizon))
        image_folder = os.path.join(evaluation_folder, 'images_' + str(i))
        os.makedirs(image_folder, exist_ok=True)
        run_episode(env, model, save_imgs=True, max_horizon=horizon, image_folder=image_folder)
        # save images as video
        image_files = image_folder + f'/img_%0{num_digits}d.png'
        control_freq = 20 # equivalent to fps
        vid_filename = os.path.join(evaluation_folder, f'vid_{i}.mp4')
        os.system('ffmpeg -r ' + str(control_freq) + ' -i ' + image_files + ' -vcodec mpeg4 -y ' + vid_filename)
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', type=str)
    parser.add_argument('--n_evaluation_eps', type=int, default=10)
    parser.add_argument('--n_vids', type=int, default=5)
    
    args = parser.parse_args()
    evaluator(args.model_folder, args.n_evaluation_eps, args.n_vids)
    
    
    