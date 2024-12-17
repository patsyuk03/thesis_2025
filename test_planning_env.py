import gym
import cv2
import numpy as np
import time

TIMEOUT = 3
NUM_JOINTS = 7
NUM_SAMPLES = 10

class MujocoPlanner:
    def __init__(self):
        self.env = gym.make("dual_ur5e_gym:DualUR5eEnv", render=True)
        self.env.reset()
        self.env.render()
        self.model = self.env.model
        self.data = self.env.data
        self.dt = self.model.opt.timestep
        self.target_position = list(self.data.xpos[self.model.body(name="object_0").id])
        self.target_position[0] += 0.2
        self.step = 0
        found_solution = False

        self.vel = self.sample_joint_velocities(NUM_JOINTS, NUM_SAMPLES)
        self.vel[:,-1] = 0

        while not found_solution:
            self.step = 0
            self.start = time.time()
            found_solution = self.run_simulation()


        self.close()

    def sample_joint_velocities(self, num_joints, num_samples, mu=0, sigma=1):
        return np.random.normal(mu, sigma, (num_samples, num_joints))
    
    def sample_joint_states(self):
        return self.data.qpos[:NUM_JOINTS] + self.vel[0]

    def evaluate_cost(self, current_position):    
        cost = np.linalg.norm(current_position - self.target_position)**2
        return cost
    
    def run_simulation(self):
        found_solution = False
        self.env.reset()
        # action = [1.5, -0.88, 1.83, -0.38, 1.76, 0, 0]
        # action = [1.26, -0.75, 2.42, -1.7, 1.38, 0, 0]
        while True:
            joint_states = self.sample_joint_states()
            obs, reward, terminated, truncated, info = self.env.step(joint_states)
            image = self.env.render()

            if terminated or truncated:
                obs = self.env.reset()
                print("terminated or truncated")
                break

            cost = self.evaluate_cost(self.data.xpos[self.model.body(name="hande").id])
            print(cost)
            self.step += 1
            if cost < 0.05:
                found_solution = True
                break
            elif time.time() - self.start > TIMEOUT:
                print("Timeout reached.")
                break
            # elif self.step == len(self.vel):
            #     print("Trajectory executed.")
            #     break

            cv2.imshow("image", image)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

        return found_solution

    def close(self):
        self.env.close()
        print("Finished.")


def main():
    MujocoPlanner()

if __name__ == "__main__":
    main()