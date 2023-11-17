from robosuite.wrappers import GymWrapper
from gym.core import Env
import numpy as np
from gym import spaces
import torchvision.transforms as T
from PIL import Image
import cv2

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

    def __init__(self, env, vip_model, goal_image, keys=None):
        self.vip_model = vip_model
        self.vip_model.to('cuda')
        self.vip_model.eval()
        self.transform = T.Compose([T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor()])
        self.goal_embedding = self.get_vip_embedding(goal_image)
        
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        # Get reward range (MAY need to change this based upon VIP reward)
        self.reward_range = (0, self.env.reward_scale)

        embedding_keys = []
        if keys is None:
            keys = []
            # Add object obs if requested
            if self.env.use_object_obs:
                keys += ["object-state"]
            # Add image obs if requested
            if self.env.use_camera_obs:
                embedding_keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
            # Iterate over all robots to add to state
            for idx in range(len(self.env.robots)):
                keys += ["robot{}_proprio-state".format(idx)]
        self.keys = keys

        # Gym specific attributes
        self.env.spec = None
        self.metadata = None

        # set up observation and action spaces
        obs = self.env.reset()
        self.modality_dims = {key: obs[key].shape for key in self.keys}
        for key in embedding_keys:
            self.modality_dims[key] = 1024
            self.keys += [key]
        # let's modify the modality dims to be the VIP embedding size, which is 1024
        # check if VIP is on the GPU
        flat_ob = self._flatten_obs(obs)
        flat_ob = self.add_embedding_flattened_obs(flat_ob, obs)
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
            if 'image' in key:
                embedding_dict[key + '_embedding'] = self.get_vip_embedding(ob_dict[key])
        flattened_obs = self._flatten_obs(ob_dict)
        return self.add_embedding_flattened_obs(flattened_obs, ob_dict)

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
        embedding_dict = {}
        for key in ob_dict:
            if 'image' in key:
                embedding_dict[key + '_embedding'] = self.get_vip_embedding(ob_dict[key])
        flattened_obs = self._flatten_obs(ob_dict)
        obs = self.add_embedding_flattened_obs(flattened_obs, ob_dict)
        # let's add l2 distance between current embedding and goal embedding as reward
        cur_embedding = embedding_dict['agentview_image_embedding']
        vip_reward = -np.linalg.norm(cur_embedding - self.goal_embedding)
        # add it to current reward (now the reward is unbounded, not sure if it makes a huge difference?)
        # maybe we can normalize the difference vector
        reward += vip_reward
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