import os
import mujoco.mjx as mjx 
import mujoco
import cv2
import jax
from jax import numpy as jnp
import numpy as np
import time

class MJXPlanner:
    def __init__(self, model_path, n_batch=3, n_samples=10, n_steps=100, visualize=False):
        print("Initializing planner...")

        self.model_path = model_path
        self.visualize = visualize

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model)

        self.n_batch = n_batch #Number of times mean and std is updated
        self.n_samples = n_samples #Number of sample trajectories per environment
        self.n_steps = n_steps #Number of steps per trajectory
        self.n_joints = self.model.nu #Number of joints

        self.mjx_model = mjx.put_model(self.model)
        self.mjx_data = mjx.put_data(self.model, self.data)

        self.mean = np.zeros(self.n_joints)
        self.std = np.ones(self.n_joints)

        self.target_position = self.model.body(name="object_0").pos
        print("Position of the object:", self.target_position)

        print('JIT-compiling the model physics step...')
        start = time.time()
        self.jit_step = jax.jit(mjx.step)
        print(f'Compilation took {time.time() - start}s.')

        self.camera = mujoco.MjvCamera() 
        self.camera.lookat[:] = [0.0, 0.0, 0.0]
        self.camera.distance = 3.0  

        # self.prev_time = self.data.time
        # while self.visualize:
        #     self.visualization()

    def generate_samples(self):
        samples = np.zeros((self.n_samples, self.n_steps, self.n_joints), float)
        for trajectory in range(self.n_samples):
            samples[trajectory] = np.random.normal(self.mean, self.std, (self.n_steps, self.n_joints))
        samples[:,:,-1] = 0
        return jnp.array(samples)

    def single_trajectory_cost(self, trajectory, data):
        temp_mjx_data = mjx.put_data(self.model, data)
        total_cost = 0.0
        for idx, step in enumerate(trajectory):
            qvel = temp_mjx_data.qvel.at[:self.n_joints].set(step)
            temp_mjx_data = temp_mjx_data.replace(qvel=qvel)
            temp_mjx_data = self.jit_step(self.mjx_model, temp_mjx_data)
            current_position = temp_mjx_data.xpos[self.model.body(name="hande").id]
            total_cost += jnp.linalg.norm(self.target_position - current_position)

            if idx%100 == 0:
                print(f"Total cost at step #{idx}: {total_cost}")
                print("Current pose",current_position)

        return total_cost

    def evaluate_batch(self, batch_trajectories):
        costs = jax.vmap(self.single_trajectory_cost, in_axes=(0, None))(batch_trajectories, self.data)
        return np.array(costs)


    def optimizer(self):
        avg_batch_cost = list()
        for batch in range(self.n_batch):
            samples = self.generate_samples()

            costs = self.evaluate_batch(samples)
            
            elite_idxs = np.argsort(costs)[:self.n_samples // 5]
            elite_samples = samples[elite_idxs]
            elite_samples = elite_samples.reshape((elite_samples.shape[0]*elite_samples.shape[1], elite_samples.shape[2]))

            self.mean = np.mean(elite_samples, axis=0)
            self.std = np.std(elite_samples, axis=0)

            print(f"Batch {batch + 1}/{self.n_batch}: Mean cost = {np.mean(costs):.4f} \nMean: {self.mean}\nSTD: {self.std}")

            avg_batch_cost.append(np.mean(costs))
        return avg_batch_cost
    
    def run_trajectory(self, mean=None, std=None):
        if mean!=None and std!=None:
            self.mean = mean
            self.std = std
        samples = self.generate_samples()
        traj = 0
        while self.visualize:
            for vel in samples[traj]:
                self.data.qvel[:self.n_joints] = vel
                mujoco.mj_step(self.model, self.data)
                out = self.visualization(traj)
                if traj != out or not self.visualize:
                    traj = out
                    break                

    def visualization(self, traj):
        self.renderer.update_scene(self.data, camera=self.camera)
        image = self.renderer.render()


        cv2.imshow(f"Result", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.visualize = False
            cv2.destroyAllWindows()
        elif key == ord("w"):
            if traj < self.n_samples:
                traj += 1
            else:
                traj = 0
            print("Traj:", traj)
        elif key == ord("s"):
            if traj > 0:
                traj -= 1
            else:
                traj = self.n_samples-1
            print("Traj:", traj)

        return traj


def main():
    model_path = f"{os.path.dirname(__file__)}/../universal_robots_ur5e/scene_mjx.xml" 
    mp = MJXPlanner(model_path, n_batch=100, n_samples=10, n_steps=500, visualize=True)

    # mean = [0.57053006,  0.1621206,  -0.00401804, -0.01125557, -0.00737564, -0.20637755, 0. ]
    # std = [1.0005847,  1.0102792,  0.9040343,  0.9934344,  0.96778953, 1.0823556, 0. ]
    mean = [1.939485,    0.53497046,  0.35734025,  0.60128725, -0.5419982,  -0.1699366, 0. ]
    std = [0.81699705, 0.95854604, 1.2487191,  0.9681563,  1.1003195,  0.6305805, 0. ]
    mp.run_trajectory(mean=mean, std=std)

    # cost_list = mp.optimizer()
    # file_path = f"{os.path.dirname(__file__)}/costs1.txt" 
    # with open(file_path, 'w') as file:
    #     for cost in cost_list:
    #         file.write(f'{cost}\n')





if __name__ == "__main__":
    main()