import gym
import cv2
import numpy as np
import time
from multiprocessing import Process, Queue


TIMEOUT = 1
NUM_JOINTS = 7
NUM_SAMPLES = 10

class MujocoPlanner:
    def __init__(self, seed=0, output_queue=None, visualize=False, run_simulation=False):
        self.seed = seed
        np.random.seed(self.seed) 
        self.env = gym.make("dual_ur5e_gym:DualUR5eEnv", render=True)
        self.env.reset(seed=self.seed)
        self.env.render()

        print(f"Env #{self.seed} initialized.")

        self.model = self.env.model
        self.data = self.env.data
        
        self.visualize = visualize

        self.best_cost = 1
        self.best_vel = [0]*NUM_JOINTS
        self.start = time.time()

        self.target_position = list(self.data.xpos[self.model.body(name="object_0").id])
        self.target_position[0] += 0.2

        self.vel = self.sample_joint_velocities(NUM_JOINTS, NUM_SAMPLES)
        self.vel[:,-1] = 0

        if run_simulation:
            self.run_simulation()

            out = {
                "cost" : self.best_cost,
                "vel" : self.best_vel
            }

            output_queue.put(out)

            self.close()
        

    def sample_joint_velocities(self, num_joints, num_samples, mu=0, sigma=1):
        return np.random.normal(mu, sigma, (num_samples, num_joints))
    
    def sample_joint_states(self, idx):
        return self.data.qpos[:NUM_JOINTS] + self.vel[idx]

    def evaluate_cost(self, current_position):    
        cost = np.linalg.norm(current_position - self.target_position)**2
        return cost
    
    def run_simulation(self):
        found_solution = False
        for idx, v in enumerate(self.vel):
            self.env.reset()
            self.start = time.time()
            while True:
                joint_states = self.sample_joint_states(idx)
                obs, reward, terminated, truncated, info = self.env.step(joint_states)

                if terminated or truncated:
                    obs = self.env.reset()
                    print(f"Env #{self.seed} terminated or truncated.")
                    break

                cost = self.evaluate_cost(self.data.xpos[self.model.body(name="hande").id])
                if cost < 0.05:
                    found_solution = True
                    if cost < self.best_cost:
                        print(f"Env #{self.seed} found new best cost: {cost}.")
                        self.best_cost = cost
                        self.best_vel = self.vel[idx]
                    break


                elif time.time() - self.start > TIMEOUT:
                    print(f"Env #{self.seed} timeout reached.")
                    break
                if self.visualize:
                    image = self.env.render()
                    cv2.imshow(f"Env #{self.seed}", image)
                    cv2.waitKey(1)

        return found_solution
    
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
    num_envs = 5
    processes = []
    output_queue = Queue()
    for i in range(num_envs):
        process = Process(target=MujocoPlanner, args=(i, output_queue, False, True))
        processes.append(process)
        process.start()

    results = []
    for process in processes:
        process.join()  # Wait for each process to finish
        results.append(output_queue.get())

    print(results)

    vels = [result['vel'] for result in results]    
    # vels = [[ 1.14472371,  0.90159072,  0.50249434,  0.90085595, -0.68372786,
    #    -0.12289023,  0.], [ 1.76405235,  0.40015721,  0.97873798,  2.2408932 ,  1.86755799,
    #    -0.97727788,  0.        ], [0, 0, 0, 0, 0, 0, 0]]

    mp = MujocoPlanner()
    mp.vel = vels
    mp.run_visualisation()

    print("Done.")

if __name__ == "__main__":
    main()