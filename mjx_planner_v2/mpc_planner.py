import numpy as np
from mjx_planner import cem_planner
import mujoco.mjx as mjx 
import mujoco
import time
import jax.numpy as jnp
import jax
import os
from mujoco import viewer
import matplotlib.pyplot as plt

start_time = time.time()
cem =  cem_planner(
    num_dof=6, 
    num_batch=100, 
    num_steps=10, 
    maxiter_cem=5,
    w_pos=5,
    num_elite=0.1,
    timestep=0.04
    )
print(f"Initialized CEM Planner: {round(time.time()-start_time, 2)}s")

model = cem.model
data = cem.data

xi_mean = jnp.zeros(cem.nvar)

start_time = time.time()
_ = cem.compute_cem(xi_mean, data.qpos[:6], data.qvel[:6])
print(f"Compute CEM: {round(time.time()-start_time, 2)}s")


thetadot = np.array([0]*6)

with viewer.launch_passive(model, data) as viewer_:
    viewer_.cam.distance = 4
    viewer_.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

    while viewer_.is_running():
        start_time = time.time()
        cost, best_cost_g, best_vels, best_traj, xi_mean = cem.compute_cem(xi_mean, data.qpos[:6], data.qvel[:6])
        thetadot = best_vels[1]

        data.qvel[:6] = thetadot
        mujoco.mj_step(model, data)

        eef_pos = data.xpos[cem.hande_id]
        cost_ = np.linalg.norm(eef_pos - cem.target_pos)
        print(f'Step Time: {round(time.time() - start_time, 2)}s | Cost: {round(float(cost_), 2)}')

        viewer_.sync()

        time_until_next_step = model.opt.timestep - (time.time() - start_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)  