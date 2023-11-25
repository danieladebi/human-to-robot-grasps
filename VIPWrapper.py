from robosuite.wrappers import GymWrapper
from gym.core import Env
import numpy as np
from gymnasium import spaces
import torchvision.transforms as T
from PIL import Image
import cv2
import torch
import hand_pose_model

class VIPWrapper(GymWrapper, Env):
    """
    Initializes the VIPWrapper.
    Builds upon the Robosuite GymWrapper.

    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env, vip_model, goal_image, 
                 use_vip_embedding_obs=True,
                 use_hand_pose_obs=True,
                 use_vip_reward=True, vip_reward_type='add', 
                 vip_reward_min=-1, vip_reward_max=1,
                 vip_reward_interval=1, keys=None):
        self.vip_model = vip_model
        self.vip_model.to('cuda')
        self.vip_model.eval()
        self.transform = T.Compose([T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor()])
        self.goal_embedding = self.get_vip_embedding(goal_image)
        self.use_vip_embedding_obs = use_vip_embedding_obs
        self.use_hand_pose_obs = use_hand_pose_obs
        self.use_vip_reward = use_vip_reward
        self.vip_reward_type = vip_reward_type
        self.vip_reward_interval = vip_reward_interval
        self.curr_vip_reward_interval = 1
        
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        # Get reward range (MAY need to change this based upon VIP reward)
        self.reward_range = (0, self.env.reward_scale)
        # vip reward is [-1, 1]
        if use_vip_reward:
            if vip_reward_type == 'add':
                self.reward_range = (-1, self.env.reward_scale + 1)
            else: # multiply
                self.reward_range = (-self.env.reward_scale, self.env.reward_scale)
                
        if self.use_hand_pose_obs is True:
            model_filepath = 'hand_pose_model/best_model'
            self.hand_pose_model = hand_pose_model.load_from_file(model_filepath)

        embedding_keys = []
        if keys is None:
            keys = []
            # Add object obs if requested
            if self.env.use_object_obs:
                keys += ["object-state"]
            # Add image obs if requested
            if self.env.use_camera_obs:
                cam_data = [f"{cam_name}_image" for cam_name in self.env.camera_names]
                if self.use_vip_embedding_obs:
                    embedding_keys += cam_data
                else:
                    keys += cam_data
            # Iterate over all robots to add to state
            for idx in range(len(self.env.robots)):
                keys += ["robot{}_proprio-state".format(idx)]
        self.keys = keys

        # Gym specific attributes
        self.env.spec = None
        self.metadata = None

        # set up observation and action spaces
        obs = self.env.reset()
        self.latest_obs_dict = obs
        self.modality_dims = {key: obs[key].shape for key in self.keys}
        
        for key in embedding_keys:
            self.modality_dims[key] = 1024
            self.keys += [key]
        if self.use_hand_pose_obs:
            self.modality_dims['hand_pose'] = 21 * 3
        # let's modify the modality dims to be the VIP embedding size, which is 1024
        # check if VIP is on the GPU
        flat_ob = self._flatten_obs(obs)
        if self.use_vip_embedding_obs or self.use_hand_pose_obs: 
            flat_ob = self.add_embedding_flattened_obs(flat_ob, obs)
            
        # bound VIP reward
        if self.use_vip_reward:
            # initial distance from start -> goal
            start_img = obs['agentview_image']
            start_embedding = self.get_vip_embedding(start_img)
            self.max_embedding_dist = np.linalg.norm(self.goal_embedding - start_embedding)
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high)
        
    def get_vip_embedding(self, img):
        # look like robosuite already uses BGR so let's omit line below
        # cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        img = self.transform(img)  * 255
        img = img.unsqueeze(0)
        img.cuda()
        embedding = self.vip_model(img)
        embedding = embedding.cpu().detach().numpy()
        return embedding
    
    def get_hand_pose(self, img):
        img = Image.fromarray(img.astype(np.uint8))
        img = self.transform(img)  * 255
        img = img.unsqueeze(0)
        img.cuda()
        pred_hp = self.hand_pose_model(img).reshape(-1, 21, 3)
        pred_hp = pred_hp.cpu().detach().numpy()
        return pred_hp.flatten()


    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)

    def add_embedding_flattened_obs(self, flattened_obs, embedding_dict):
        ob_lst = []
        for key in embedding_dict:
            ob_lst.append(np.array(embedding_dict[key]).flatten())
        return np.concatenate([flattened_obs] + ob_lst)
    
    def reset(self):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict.

        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        ob_dict = self.env.reset()
        embedding_dict = {}
        for key in ob_dict:
            if self.use_vip_embedding_obs and 'image' in key:
                embedding_dict[key + '_embedding'] = self.get_vip_embedding(ob_dict[key])
            if self.use_hand_pose_obs and 'image' in key:
                embedding_dict['hand_pose'] = self.get_hand_pose(ob_dict[key])
        flattened_obs = self._flatten_obs(ob_dict)
        if self.use_vip_embedding_obs:
            flattened_obs = self.add_embedding_flattened_obs(flattened_obs, ob_dict)
        return flattened_obs

    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) flattened observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ob_dict, reward, done, info = self.env.step(action)
        flattened_obs = self._flatten_obs(ob_dict)
                
        use_vip_reward = self.use_vip_reward and (self.curr_vip_reward_interval == self.vip_reward_interval)
        self.curr_vip_reward_interval = (self.curr_vip_reward_interval + 1) % self.vip_reward_interval
        embedding_dict = {}
        
        if self.use_vip_embedding_obs or use_vip_reward:
            for key in ob_dict:
                if 'image' in key:
                    embedding_dict[key + '_embedding'] = self.get_vip_embedding(ob_dict[key])
        if self.use_hand_pose_obs:
            for key in ob_dict:
                if 'image' in key:
                    embedding_dict[key + '_hand_pose'] = self.get_hand_pose(ob_dict[key])
        obs = flattened_obs
        if self.use_vip_embedding_obs or self.use_hand_pose_obs:
            obs = self.add_embedding_flattened_obs(flattened_obs, ob_dict)
            
        if self.use_vip_reward:
            cur_embedding = embedding_dict['agentview_image_embedding']
            vip_distance = np.linalg.norm(cur_embedding - self.goal_embedding)
            # bound the reward
            vip_distance = np.clip(vip_distance, 0, self.max_embedding_dist)
            # range of vip_reward will be [-1, 1]
            vip_reward = 1 - 2 * vip_distance / self.max_embedding_dist
            if self.vip_reward_type == 'add':
                reward += vip_reward
            else:
                reward *= vip_reward
                
        self.latest_obs_dict = ob_dict
        return obs, reward, done, info

    def seed(self, seed=None):
        """
        Utility function to set numpy seed

        Args:
            seed (None or int): If specified, numpy seed to set

        Raises:
            TypeError: [Seed must be integer]
        """
        # Seed the generator
        if seed is not None:
            try:
                np.random.seed(seed)
            except:
                TypeError("Seed must be an integer type!")

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()