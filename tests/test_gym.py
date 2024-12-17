import gym
import time
from gym.envs.mujoco import mujoco_env


# Create the MuJoCo Gym environment
env = gym.make('HalfCheetah-v4')  # MuJoCo environment from Gym

# Reset the environment to its initial state
observation = env.reset()

# Number of steps to run the simulation
num_steps = 1000

print("Starting simulation...")

# Main loop to interact with the environment
for step in range(num_steps):
    env.render()  # Render the environment

    # Random action: Replace this with a trained policy for meaningful behavior
    action = env.action_space.sample()

    # Take a step in the environment
    observation, reward, done, info = env.step(action)
    
    print(f"Step: {step}, Reward: {reward:.2f}, Done: {done}")

    # If the simulation ends, reset the environment
    if done:
        print("Episode ended, resetting environment...")
        observation = env.reset()

    time.sleep(0.01)  # Slow down the simulation for better visualization

# Close the environment
env.close()
print("Simulation complete!")
