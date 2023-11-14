import gymnasium as gym 
import robosuite
from robosuite.controllers import load_controller_config

import numpy as np

from robosuite.environments.base import register_env
from robosuite.wrappers import GymWrapper

from stable_baselines3 import PPO 
from stable_baselines3.common.save_util import save_to_zip_file, load_from_zip_file
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

controller_config = load_controller_config(default_controller="OSC_POSE")

rs_env = robosuite.make(
    "Lift",
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

# TODO: Write PPO algorithm to test
env = GymWrapper(rs_env)

def wrap_env(env):
    wrapped_env = Monitor(env) # needed for extracting eprewmean and eplenmean
    wrapped_env = DummyVecEnv([lambda: wrapped_env]) # Needed for all environments (e.g. used for multi-processing)
    wrapped_env = VecNormalize(wrapped_env) # Needed for improving training when using MuJoCo envs?
    return wrapped_env

wrap_env(env)
filename = 'lift_test'

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_lift_tensorboard/")
model.learn(total_timesteps=3e5, tb_log_name=filename)

model.save('trained_models/' + filename)
env.save('trained_models/vec_normalize_' + filename + '.pkl') # Save VecNomralize statistics

env_test = GymWrapper(
    robosuite.make(
        'Lift', #Block Lifting | Block Stacking | Pick and Place | Nut Assembly | Door Opening | Table Wiping
        robots="Panda",
        controller_configs=controller_config,
        gripper_typers="default",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        control_freq=20,
        render_camera=None,
        horizon=2000,
        reward_shaping=True
    )
)

model = PPO.load("trained_models/"+filename)
env = DummyVecEnv([lambda : env_test])
env = VecNormalize.load("trained_models/vec_normalize_" + filename + ".pkl", env)

env.training = False
env.norm_reward = False

obs = env.reset()

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    env_test.render()
    if done: 
        obs = env.reset()
        break

env.close()