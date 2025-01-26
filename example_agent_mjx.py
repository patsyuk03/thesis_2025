from jax import vmap
import jax
import os
import gym
import jax.numpy as jnp
import numpy as np

def step(env, action):
    print('AAAAAAAAAAAAAAAA')
    obs, rewards = env.step(action)
    return obs, rewards



# Number of parallel environments
batch_size = 3
action_space_dim = 7

# Create a batch of environments
model_path = f"{os.path.dirname(__file__)}/universal_robots_ur5e/scene.xml" 
env = gym.make("dual_ur5e_gym:DualUR5eEnvMJX", model_path=model_path)
# envs = [gym.make("dual_ur5e_gym:DualUR5eEnvMJX", model_path=model_path) for _ in range(batch_size)]

# print(envs)

# actions = jnp.zeros((batch_size, action_space_dim)) 

# step_batch = jax.vmap(step, in_axes=(0, 0)) 

# print(np.array(envs).shape)
# print(actions.shape)

# obs, rewards = step_batch(envs, actions)
