import multiprocessing
import mujoco
import mujoco.viewer

import time
import os
import numpy as np


NUM_JOINTS = 6
NUM_ENVS = 3
SAMP_PER_ENV = 5
NUM_SAMPLES = NUM_ENVS*SAMP_PER_ENV
TIMEOUT = 1

class MujocoPlanner:
    def __init__(self):
        self.model_path = f"{os.path.dirname(__file__)}/../universal_robots_ur5e/scene.xml" 
        model = mujoco.MjModel.from_xml_path(self.model_path)
        data = mujoco.MjData(model)
        self.target_position = list(data.xpos[model.body(name="object_0").id])

        manager = multiprocessing.Manager()
        self.best_cost = manager.Value('d', float('inf'))
        self.best_vel = manager.list()
        self.visualize = manager.Value('b', False)
        self.start = manager.Value('d', time.time())


        self.velocities = self.sample_joint_velocities(NUM_JOINTS, NUM_SAMPLES)

        with multiprocessing.Pool(processes=NUM_ENVS) as pool:
            # pool.map(self.run_simulation, np.arange(1, NUM_ENVS+1))
            async_results = []
            
            for env_id in np.arange(1, NUM_ENVS + 1):
                async_results.append(pool.apply_async(self.run_simulation, (env_id,)))
            for result in async_results:
                result.get()

        if self.visualize.value:
            print(self.best_vel[:])
            self.visualize_path()
        else:
            print("Solution was not found.")

        
    
    def sample_joint_velocities(self, num_joints, num_samples, mu=0, sigma=1):
        return np.random.normal(mu, sigma, (num_samples, num_joints))
    
    def evaluate_cost(self, current_position):    
        cost = np.linalg.norm(current_position - self.target_position)**2
        return cost
    
    def run_simulation(self, sim_id):
        model = mujoco.MjModel.from_xml_path(self.model_path)
        data = mujoco.MjData(model)

        velosities = self.velocities[(sim_id-1)*SAMP_PER_ENV:sim_id*SAMP_PER_ENV]
        print(f"Simulation #{sim_id} vel: {velosities}")
        for vel in velosities:
            if self.visualize.value:
                break
            found_solution = False
            while time.time() - self.start.value < TIMEOUT and not found_solution:
                data.qvel[:NUM_JOINTS] = vel
                mujoco.mj_step(model, data)
                cost = self.evaluate_cost(data.xpos[model.body(name="hande").id])
                if cost < 0.1:
                    found_solution = True


            model, data = self.reset_simulation(model, data)

            if found_solution:
                self.visualize.value = True
                if cost < self.best_cost.value:
                    self.best_cost.value = cost
                    self.best_vel[:] = vel
                print(f"Simulation #{sim_id}: Solution was found. Cost {cost}\n")
            else:
                print(f"Simulation #{sim_id}: Timeout. Solution was not found.\n")

    def reset_simulation(self, model, data):
        data.qpos = [0]*len(data.qpos)
        data.qvel = [0]*len(data.qvel)
        data.ctrl = [0]*len(data.ctrl)
        mujoco.mj_step(model, data)
        self.start.value = time.time()
        return model, data

    def visualize_path(self, model=None, data=None):
        model = mujoco.MjModel.from_xml_path(self.model_path)
        data = mujoco.MjData(model)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 4
            start = time.time()
            while viewer.is_running() and time.time() - start < 30:
                step_start = time.time()
                data.qvel[:NUM_JOINTS] = self.best_vel[:]
                mujoco.mj_step(model, data)
                viewer.sync()
                cost = self.evaluate_cost(data.xpos[model.body(name="hande").id])
                print(cost)
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        






def main():
    MujocoPlanner()

if __name__ == "__main__":
    main()