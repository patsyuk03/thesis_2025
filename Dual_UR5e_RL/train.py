import numpy as np
from dm_control.rl import control
from dm_control import mjcf
import os
import cv2
from dm_env import specs
import torch
import torch.nn as nn
import torch.optim as optim

model_path = f"{os.path.dirname(__file__)}/../universal_robots_ur5e/dual_arm_gripper_scene.xml" 
mjcf_model = mjcf.from_path(model_path)
physics = mjcf.Physics.from_mjcf_model(mjcf_model)

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Tanh for action range [-1, 1]
        return x

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


# Initialize policy network, optimizer, and loss function
obs_dim = 44  # position + velocity
action_dim = 1
policy_net = PolicyNetwork(obs_dim, action_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

# Collect trajectories and optimize
gamma = 0.99  # Discount factor

task = CustomTask()
env = control.Environment(physics, task, time_limit=1000)

time_step = env.reset()
num_episodes = 5
stop = False
for episode in range(num_episodes):
    time_step = env.reset()
    log_probs = []
    rewards = []
    cumulative_reward = 0
    while not time_step.last():
        obs = np.concatenate((time_step.observation["position"].flatten(), time_step.observation["velocity"].flatten()))
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        # Get action from policy network
        action_prob = policy_net(obs_tensor)
        action = action_prob.detach().numpy().flatten()

        # Apply action
        time_step = env.step(action)
        reward = time_step.reward or 0
        cumulative_reward += reward

        # Store log probability and reward
        log_prob = torch.log(torch.tensor(action_prob + 1e-8))  # Add small value for stability
        log_probs.append(log_prob)
        rewards.append(reward)

        print("Reward:", time_step.reward)
        print("Observation:", time_step.observation)

        frame = env.physics.render(height=240, width=320, camera_id=0)
        cv2.imshow("Simulation", frame[..., ::-1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop = True
            print("Stopping")
            break

    discounted_rewards = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)

    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

    # Compute loss
    loss = 0
    for log_prob, G in zip(log_probs, discounted_rewards):
        loss += -log_prob * G  # Policy gradient los

    # Optimize the policy network
    optimizer.zero_grad()
    loss.requires_grad = True
    loss.backward()
    optimizer.step()

    print(f"Episode {episode + 1}: Cumulative Reward: {cumulative_reward}")
    if stop:
        break

torch.save(policy_net.state_dict(), f"{os.path.dirname(__file__)}/models/policy_network.pth")
print("Training completed and model saved.")