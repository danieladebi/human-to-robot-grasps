from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from VIPWrapper import VIPWrapper
from VIPGoalLoader import VIPGoalLoader
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from vip import load_vip

task_name_to_env_name_map = {"lift" : "Lift",
                             "can" : "Can",
                             "square" : "Square",
                             "transport": "Transport",
                             "tool_hang": "Tool_Hang"}

class CustomDummyVecEnv(DummyVecEnv):
    def reset(self):
        for env_idx in range(self.num_envs):
            obs, *_ = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return self._obs_from_buf()

    def step_wait(self):
        for env_idx in range(self.num_envs):
            obs, rew, terminated, truncated, info = self.envs[env_idx].step(self.actions[env_idx])
            self.buf_rews[env_idx] = rew
            self.buf_dones[env_idx] = terminated | truncated
            self.buf_infos[env_idx] = info
            if terminated | truncated:
                obs, *_ = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), self.buf_rews, self.buf_dones, self.buf_infos)
    
def get_vip_wrapped_env(task_name, args):
    vip_goal_loader = VIPGoalLoader(task_name)
    vip_goal_loader.load_dataset()
    vip_goal = vip_goal_loader.get_random_goal_image()

    env_meta = FileUtils.get_env_metadata_from_dataset(vip_goal_loader.processed_dataset_path)
    env_meta['env_kwargs']['use_object_obs'] = False
    env_meta['env_kwargs']['camera_names'] = 'agentview'

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
    
    def make_env(rank, seed=0):
        """
        Utility function for multiprocessed env.
        :param rank: (int) index of the subprocess
        :param seed: (int) the initial seed for RNG
        """
        def _init():
            env = VIPWrapper(rs_env, vip_model, vip_goal,
                    use_vip_embedding_obs=args.use_vip_embedding_obs,
                    use_hand_pose_obs=args.use_hand_pose_obs,
                    use_vip_reward=args.use_vip_reward,
                    vip_reward_type=args.vip_reward_type,
                    vip_reward_min=args.vip_reward_min,
                    vip_reward_max=args.vip_reward_max,
                    vip_reward_interval=args.vip_reward_interval)
            # env.seed(seed + rank)
            env = Monitor(env)
            return env
        return _init

    num_envs = 1
    env = CustomDummyVecEnv([make_env(i) for i in range(num_envs)])
    return VecNormalize(env, norm_obs=True, norm_reward=True)

