import numpy as np
from dm_control import suite
from dm_control.suite.wrappers import pixels
from dm_control.rl import control
import matplotlib.pyplot as plt
from dm_control import mjcf
import os
import cv2
from dm_env import specs
import torch

model_path = f"{os.path.dirname(__file__)}/../universal_robots_ur5e/dual_arm_gripper_scene.xml" 
mjcf_model = mjcf.from_path(model_path)
physics = mjcf.Physics.from_mjcf_model(mjcf_model)

class CustomTask(control.Task):
    def __init__(self):
        super().__init__()
        self._terminate = False
    
    def action_spec(self, physics):
        # Define a 1D action space with values between -1 and 1
        return specs.BoundedArray(
            shape=(1,), dtype=float, minimum=[-1], maximum=[1], name="action"
        )
    
    def before_step(self, action, physics):
        # Perform any necessary preprocessing on the action
        physics.data.ctrl[0] += action[0]
        if abs(physics.data.qpos[0]) > 2.0:  # Custom condition
            self._terminate = True
    
    def initialize_episode(self, physics):
        # Reset the physics for a new episode
        self._terminate = False 
        physics.reset()
    
    def get_observation(self, physics):
        # Return observations (e.g., position, velocity)
        obs = {
            "position": physics.data.qpos.copy(),
            "velocity": physics.data.qvel.copy(),
        }
        return obs
    
    def get_reward(self, physics):
        # Define a reward function
        return -abs(physics.data.qpos[0])  # Example: penalize distance from origin

    def get_termination(self, physics):
        return 1.0 if self._terminate else None

    
task = CustomTask()
env = control.Environment(physics, task, time_limit=1000)

time_step = env.reset()
num_episodes = 5
stop = False
for episode in range(num_episodes):
    time_step = env.reset()
    cumulative_reward = 0
    while not time_step.last():
        action = [0.1]  # Example action
        time_step = env.step(action)
        print("Reward:", time_step.reward)
        print("Observation:", time_step.observation)
        cumulative_reward += time_step.reward or 0
        frame = env.physics.render(height=240, width=320, camera_id=0)
        cv2.imshow("Simulation", frame[..., ::-1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop = True
            print("Stopping")
            break
    print("Last timestep reached. Episode is over.")
    if stop:
        break