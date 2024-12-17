import mujoco
from gym.envs.mujoco import mujoco_env

from gym import utils, spaces

import os
import time
import numpy as np

class DualUR5eEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 500,
    }
    def __init__(self, render=True, render_mode="rgb_array", width=600, height=600):
        model_path = f"{os.path.dirname(__file__)}/../../universal_robots_ur5e/scene.xml" 
        frame_skip = 1
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(
            self, 
            model_path=model_path, 
            frame_skip=frame_skip, 
            observation_space=None, 
            render_mode=render_mode, 
            width=width, 
            height=height,
            camera_name="camera1"
            )
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.data.qpos.size + self.data.qvel.size

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)
        if render:
            self.render()
        # self.render = render

    def step(self, action):
        # Step the environment
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward = 1.0  # Placeholder reward
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info
    
    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation
    
    def _get_obs(self):
        # Return the observation
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

    def close(self):
        mujoco_env.MujocoEnv.close(self)

    def print_info(self):
        print("Model timestep:", self.model.opt.timestep)
        print("Set number of frames skipped: ", self.frame_skip)
        print("dt = timestep * frame_skip: ", self.dt)
        print("FPS: ", 1/self.dt)
        print("Actionspace: ", self.action_space)
        print("Observation space:", self.observation_space)
