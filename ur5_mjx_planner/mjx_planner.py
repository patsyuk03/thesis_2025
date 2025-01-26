import os
import mujoco.mjx as mjx 
import mujoco
import cv2
import jax
import numpy as np
import time

class MJXPlanner:
    def __init__(self, model_path, n_batch=3, n_envs=1, n_samples=10, n_steps=1000, visualize=False):
        print("Initializing planner...")

        self.model_path = model_path
        self.visualize = visualize

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model)

        self.n_batch = n_batch #Number of times mean and std is updated
        self.n_envs = n_envs #Number of parallel environments pre batch
        self.n_samples = n_samples #Number of sample trajectories per environment
        self.n_steps = n_steps #Number of steps per trajectory
        self.n_joints = self.model.nu #Number of joints

        self.mjx_model = mjx.put_model(self.model)
        self.mjx_data = mjx.put_data(self.model, self.data)

        self.mean = np.zeros(self.n_joints)
        self.std = np.ones(self.n_joints)
        self.samples = self.generate_samples()

        self.target_position = self.model.body(name="object_0").pos
        print("Position of the object:", self.target_position)

        print('JIT-compiling the model physics step...')
        start = time.time()
        self.jit_step = jax.jit(mjx.step)
        print(f'Compilation took {time.time() - start}s.')

        camera = mujoco.MjvCamera() 
        camera.lookat[:] = [0.0, 0.0, 0.0]
        camera.distance = 3.0  

        self.prev_time = self.data.time
        while self.visualize:
            self.visualization(camera)

    def generate_samples(self):

        samples = np.empty((self.n_samples, self.n_steps, self.n_joints), float)
        for trajectory in range(self.n_samples):
            samples[trajectory] = np.random.normal(self.mean, self.std, (self.n_steps, self.n_joints))
        samples[:,:,-1] = 0

    def single_trajectory_cost(self, trajectory, data):
        temp_data = mjx.put_data(self.model, data)
        total_cost = 0.0
        for t in range(self.n_steps):
            # Apply control
            temp_data.ctrl[:] = trajectory[t]
            # Step the simulation
            temp_data = self.jit_step(self.mjx_model, temp_data)
            # Compute cost as distance to the target
            current_position = self.data.xpos[self.model.body(name="hande").id]
            total_cost += jax.numpy.linalg.norm(self.target_position - current_position)
        return total_cost

    def evaluate_batch(self, batch_trajectories):
        costs = jax.vmap(self.single_trajectory_cost, in_axes=(0, None))(batch_trajectories, self.mjx_data)
        return np.array(costs)


    def optimizer(self):
        for batch in range(self.n_batch):
            # Step 1: Generate samples
            samples = self.generate_samples()

            # Step 2: Evaluate the cost of each trajectory in parallel
            costs = self.evaluate_batch(samples)

    def visualization(self, camera, fps=60):
        mujoco.mj_step(self.model, self.data)
        self.mjx_data = self.jit_step(self.mjx_model, self.mjx_data)
        if self.mjx_data.time - self.prev_time >= 1/fps:
            self.prev_time = self.data.time
            self.data = mjx.get_data(self.model, self.mjx_data)
            self.renderer.update_scene(self.data, camera=camera)

            image = self.renderer.render()

            cv2.imshow(f"Result", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.visualize = False
                cv2.destroyAllWindows()

        







def main():
    model_path = f"{os.path.dirname(__file__)}/../universal_robots_ur5e/scene_mjx.xml" 
    mp = MJXPlanner(model_path, visualize=True)




if __name__ == "__main__":
    main()