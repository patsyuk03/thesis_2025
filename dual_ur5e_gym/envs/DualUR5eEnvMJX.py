from gym import utils, spaces
import gym

from mujoco import mjx

import os
import time
import numpy as np

class DualUR5eEnvMJX(gym.Env, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 500,
    }
    def __init__(self, model_path, render=True, render_mode="rgb_array", width=600, height=600):
        # model_path = f"{os.path.dirname(__file__)}/../../universal_robots_ur5e/scene.xml" 
        frame_skip = 1
        utils.EzPickle.__init__(self)
        
        self.model = mjx.MjModel.from_xml_path(model_path)
        self.sim = mjx.MjSim(self.model)

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.sim.model.nq + self.sim.model.nv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)


        if render:
            self.render_mode = render_mode
            self.render(width, height)
        # self.render = render

    def step(self, action):
        # self.do_simulation(action, self.frame_skip)
        self.sim.data.ctrl[:] = action
        self.sim.step()
        obs = self._get_obs()
        print(obs)
        reward = 1.0  
        return obs, reward
    
    def reset_model(self):
        self.sim.reset()
        self.sim.set_state(self.sim.model.init_qpos, self.sim.model.init_qvel)

        observation = self._get_obs()

        return observation
    
    def _get_obs(self):
        # Return the observation
        return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat])
    
    def render(self, width=600, height=600):
        # Render the environment (assuming `mjx` has render capabilities)
        if self.render_mode == "rgb_array":
            img = self.sim.render(width=width, height=height, mode="rgb_array")
            return img
        elif self.render_mode == "depth_array":
            img = self.sim.render(width=width, height=height, mode="depth_array")
            return img
        else:
            raise NotImplementedError("Render mode not supported.")

    def close(self):
        print('close')

    def print_info(self):
        print('info')
        # print("Model timestep:", self.model.opt.timestep)
        # print("Set number of frames skipped: ", self.frame_skip)
        # print("dt = timestep * frame_skip: ", self.dt)
        # print("FPS: ", 1/self.dt)
        # print("Actionspace: ", self.action_space)
        # print("Observation space:", self.observation_space)
