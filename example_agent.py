#!/usr/bin/env python3

import gym
import numpy as np
import time
import cv2


env = gym.make("dual_ur5e_gym:DualUR5eEnv", render=True)

obs = env.reset()

for _ in range(10000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    image = env.render()

    if terminated or truncated:
        obs = env.reset()

    cv2.imshow("image", image)
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

env.close()
print("Finished.")
