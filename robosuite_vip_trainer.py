import os
from stable_baselines3 import PPO, A2C
from utils import get_vip_wrapped_env
import argparse
from stable_baselines3.common.callbacks import ProgressBarCallback
from robosuite_vip_evaluator import evaluator


def trainer(args):
    task_name = args.task
    env = get_vip_wrapped_env(task_name, args)

    # get the number of folders in the trained_models directory
    base_folder = os.path.join('trained_models', task_name)
    children_files = os.listdir(base_folder)
    children_folders = [f for f in children_files if os.path.isdir(os.path.join(base_folder, f))]
    num_models = len(children_folders)
    
    model_folder = os.path.join(base_folder, str(num_models + 1))
    # make model folder
    os.makedirs(model_folder, exist_ok=True)
    model_filepath = os.path.join(model_folder, 'model')

    # let's also save the command line arguments as csv
    args_filepath = os.path.join(model_folder, 'args.csv')
    with open(args_filepath, 'w') as f:
        for arg in vars(args):
            f.write("%s,%s\n"%(arg,getattr(args, arg)))

    if args.model == 'ppo':
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"ppo_{task_name}_tensorboard")
    elif args.model == 'a2c':
        model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=f"sac_{task_name}_tensorboard") 
    else:
        raise Exception('Model not supported')
    model.learn(total_timesteps=args.n_steps, tb_log_name=model_folder, progress_bar=True)

    model.save(model_filepath)
    env.save(os.path.join(model_filepath + '_vec_normalize.pkl')) # Save VecNormalize statistics
    env.close()
    
    # evaluate the model
    evaluator(model_folder=model_folder, n_eval_eps=30, n_vids=5)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='lift')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_steps', type=int, default=5e5)
    parser.add_argument('--model', type=str, default='ppo')
    parser.add_argument('--discrete_actions', action='store_true', default=True)
    parser.add_argument('--evaluate', action='store_true', default=True)
    
    # vip arguments
    parser.add_argument('--use_vip_embedding_obs', action='store_true', default=False)
    parser.add_argument('--use_hand_pose_obs', action='store_true', default=False)
    parser.add_argument('--use_vip_reward', action='store_true', default=False)
    parser.add_argument('--vip_reward_type', type=str, default='replace')
    parser.add_argument('--vip_reward_min', type=float, default=-1)
    parser.add_argument('--vip_reward_max', type=float, default=1)
    parser.add_argument('--vip_reward_interval', type=int, default=1)
    
    args = parser.parse_args()
    trainer(args)
    
    