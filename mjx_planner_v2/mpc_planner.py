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

def quaternion_distance(q1, q2):
    dot_product = np.abs(np.dot(q1, q2))
    return 2 * np.arccos(dot_product)

start_time = time.time()
cem =  cem_planner(
    num_dof=6, 
    num_batch=500, 
    num_steps=10, 
    maxiter_cem=5,
    w_pos=5,
    w_rot=2,
    w_col=10,
    num_elite=0.01,
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

        cost_g = np.linalg.norm(data.xpos[cem.hande_id] - cem.target_pos)   
        cost_r = quaternion_distance(data.xquat[cem.hande_id], cem.target_rot)  
        print(f'Step Time: {"%.0f"%((time.time() - start_time)*1000)}ms | Cost g: {"%.2f"%(float(cost_g))} | Cost r: {"%.2f"%(float(cost_r))} | Cost: {cost}')

        viewer_.sync()

        time_until_next_step = model.opt.timestep - (time.time() - start_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)  