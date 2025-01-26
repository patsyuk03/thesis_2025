import gym
import cv2
import numpy as np
import time
from multiprocessing import Process, Queue


TIMEOUT = 30
NUM_JOINTS = 7
NUM_STEPS = 1000
NUM_SAMPLES = 50

class MujocoPlanner:
    def __init__(self, seed, mean, std, output_queue=None, visualize=False, run_simulation=False):
        self.seed = seed
        np.random.seed(self.seed) 
        self.env = gym.make("dual_ur5e_gym:DualUR5eEnv", render=True)
        self.env.reset(seed=self.seed)
        self.env.render()

        print(f"Env #{self.seed} initialized.")

        self.model = self.env.model
        self.data = self.env.data
        
        self.visualize = visualize

        self.mean = mean
        self.std = std

        self.cost = np.empty((NUM_SAMPLES), float)
        self.best_sample = np.empty((NUM_STEPS, NUM_JOINTS), float)
        self.start = time.time()

        self.target_position = list(self.data.xpos[self.model.body(name="object_0").id])
        self.target_position[0] += 0.2

        self.vel = self.sample_joint_velocities(NUM_SAMPLES,NUM_STEPS,NUM_JOINTS)
        # print(self.vel)


        if run_simulation:
            self.run_simulation()

            out = [self.mean, self.std]

            output_queue.put(out)

            self.close()
        

    def sample_joint_velocities(self, num_samples, num_steps, num_joints):
        vel = np.empty((num_samples, num_steps, num_joints), float)
        for i in range(num_samples):
            vel[i] = np.random.normal(self.mean, self.std, (num_steps, num_joints))
        vel[:,:,-1] = 0
        return vel
    
    def sample_joint_states(self, sample_idx, step_idx):
        return self.data.qpos[:NUM_JOINTS] + self.vel[sample_idx][step_idx]

    def evaluate_cost(self, current_position):    
        return np.linalg.norm(current_position - self.target_position)**2
    
    def run_simulation(self):
        for sample_idx, steps in enumerate(self.vel):
            self.env.reset()
            self.start = time.time()
            cost_hist = list()
            for step_idx, v in enumerate(steps):
                joint_states = self.sample_joint_states(sample_idx, step_idx)
                obs, reward, terminated, truncated, info = self.env.step(joint_states)

                if terminated or truncated:
                    obs = self.env.reset()
                    print(f"Env #{self.seed} terminated or truncated.")
                    break

                cost_hist.append(self.evaluate_cost(self.data.xpos[self.model.body(name="hande").id]))


                if self.visualize:
                    image = self.env.render()
                    cv2.imshow(f"Env #{self.seed}", image)
                    cv2.waitKey(1)

            self.cost[sample_idx] = min(cost_hist)
        
        num_elites = 5
        elite_cost = np.sort(self.cost)[:num_elites]
        elite_idxs = np.argsort(self.cost)[:num_elites]
        elite_actions = self.vel[elite_idxs]
        elite_actions = elite_actions.reshape(num_elites*NUM_STEPS, NUM_JOINTS)
        # print(elite_actions.shape)

        self.mean = elite_actions.mean(axis=0)
        self.std = elite_actions.std(axis=0)

    
    def run_visualisation(self):
        self.env.reset()
        self.start = time.time()
        idx = 0
        target_reached = False
        while True:
            if not target_reached:
                joint_states = self.sample_joint_states(idx)
            obs, reward, terminated, truncated, info = self.env.step(joint_states)

            if terminated or truncated:
                obs = self.env.reset()
                print("terminated or truncated")
                break

            cost = self.evaluate_cost(self.data.xpos[self.model.body(name="hande").id])
            if cost < 0.05:
                target_reached = True
                joint_states = self.data.qpos[:NUM_JOINTS]


            image = self.env.render()
            cv2.imshow(f"Result", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("w"):
                idx = idx+1 if idx<len(self.vel)-1 else 0
                self.env.reset()
                self.start = time.time()
                target_reached = False
                print(f"Switshing to result #{idx}")
            elif key == ord("s"):
                idx = idx-1 if idx>0 else len(self.vel)-1
                self.env.reset()
                self.start = time.time()
                target_reached = False
                print(f"Switshing to result #{idx}")


    def close(self):
        self.env.close()
        print(f"Env #{self.seed} finished.")


def main():
    mean = np.ones(NUM_JOINTS)
    std = np.ones(NUM_JOINTS)


    for batch in range(10):
        num_envs = 10
        processes = []
        output_queue =  Queue()
        for i in range(num_envs):
            process = Process(target=MujocoPlanner, args=(i, mean, std, output_queue, False, True))
            processes.append(process)
            process.start()


        results = []
        for process in processes:
            process.join()  # Wait for each process to finish
            results.append(output_queue.get())

        
        results = np.array(results)
        mean = results[:,0,:].mean(axis=0)
        std = results[:,1,:].mean(axis=0)
        print(mean)
        print(std)

        # print(results)


    print("Done.")

    mp = MujocoPlanner(0, mean, std, None, True, True)

if __name__ == "__main__":
    main()