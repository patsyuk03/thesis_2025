from matplotlib import pyplot as plt
from dm_control import mjcf
from dm_control.rl import control
from dm_env import specs

import os

# Load custom XML
model_path = f"{os.path.dirname(__file__)}/universal_robots_ur5e/dual_arm_gripper_scene.xml" 
mjcf_model = mjcf.from_path(model_path)

# Create a Physics instance
physics = mjcf.Physics.from_mjcf_model(mjcf_model)

class CustomTask(control.Task):
    def __init__(self):
        super().__init__()
    
    def action_spec(self, physics):
        # Define a 1D action space with values between -1 and 1
        return specs.BoundedArray(
            shape=(1,), dtype=float, minimum=[-1], maximum=[1], name="action"
        )
    
    def before_step(self, action, physics):
        # Perform any necessary preprocessing on the action
        pass
    
    def initialize_episode(self, physics):
        # Reset the physics for a new episode
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

# Wrap in an Environment
task = CustomTask()
env = control.Environment(physics, task)
# Print the observation specs
print(env.observation_spec())

time_step = env.reset()
for _ in range(10):
    action = [0.1]  # Example action
    time_step = env.step(action)
    print("Reward:", time_step.reward)
    print("Observation:", time_step.observation)

    # Ren`der an image from the environment
    pixels = env.physics.render(height=240, width=320, camera_id=0)

    # Display the image
    plt.imshow(pixels)
    plt.axis("off")
    plt.show()
