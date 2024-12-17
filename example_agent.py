#!/usr/bin/env python3

import gym
import cv2


env = gym.make("dual_ur5e_gym:DualUR5eEnv", render=True)

obs = env.reset()
action = env.action_space.sample()
action[-1] = 0
env.print_info()
print(action)

# print(env.data.ctrl)
print(env.data.qpos[:7])


while True:
    obs, reward, terminated, truncated, info = env.step(action)
    
    image = env.render()

    # print(env.data.ctrl)
    if terminated or truncated:
        obs = env.reset()

    cv2.imshow("image", image)
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break
    elif (cv2.waitKey(1) & 0xFF) == ord("r"):
        action = env.action_space.sample()
        obs = env.reset()

print(env.data.qpos[:7])

env.close()
print("Finished.")
