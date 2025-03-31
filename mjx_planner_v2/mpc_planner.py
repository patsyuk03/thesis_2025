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
    dot_product = jnp.clip(dot_product, -1.0, 1.0)
    return 2 * np.arccos(dot_product)

start_time = time.time()
cem =  cem_planner(
    num_dof=6, 
    num_batch=1000, 
    num_steps=20, 
    maxiter_cem=2,
    w_pos=10,
    w_rot=2,
    w_col=10,
    num_elite=0.05,
    timestep=0.05
    )
print(f"Initialized CEM Planner: {round(time.time()-start_time, 2)}s")

model = cem.model
data = cem.data

xi_mean = jnp.zeros(cem.nvar)

start_time = time.time()
_ = cem.compute_cem(xi_mean, data.qpos[:6], data.qvel[:6], data.qacc[:6])
print(f"Compute CEM: {round(time.time()-start_time, 2)}s")


thetadot = np.array([0]*6)

cost_g_list = list()
cost_list = list()
cost_r_list = list()
cost_c_list = list()
thetadot_list = list()
theta_list = list()

with viewer.launch_passive(model, data) as viewer_:
    viewer_.cam.distance = 4
    viewer_.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

    while viewer_.is_running():
        start_time = time.time()
        cost, best_cost_g, best_cost_c, best_vels, best_traj, xi_mean = cem.compute_cem(xi_mean, data.qpos[:6], data.qvel[:6], data.qacc[:6])
        # thetadot = best_vels[1] 
        thetadot = np.mean(best_vels[1:3], axis=0)


        
        data.qvel[:6] = thetadot
        mujoco.mj_step(model, data)

        cost_g = np.linalg.norm(data.xpos[cem.hande_id] - cem.target_pos)   
        cost_r = quaternion_distance(data.xquat[cem.hande_id], cem.target_rot)  
        cost = np.round(cost, 2)
        print(f'Step Time: {"%.0f"%((time.time() - start_time)*1000)}ms | Cost g: {"%.2f"%(float(cost_g))} | Cost r: {"%.2f"%(float(cost_r))} | Cost c: {best_cost_c} | Cost: {cost}')

        viewer_.sync()

        cost_g_list.append(cost_g)
        cost_r_list.append(cost_r)
        cost_c_list.append(best_cost_c)
        thetadot_list.append(thetadot)
        theta_list.append(data.qpos[:6].copy())
        cost_list.append(cost[-1])

        time_until_next_step = model.opt.timestep - (time.time() - start_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)  

np.savetxt('data/costs.csv',cost_list, delimiter=",")
np.savetxt('data/thetadot.csv',thetadot_list, delimiter=",")
np.savetxt('data/theta.csv',theta_list, delimiter=",")
np.savetxt('data/cost_g.csv',cost_g_list, delimiter=",")
np.savetxt('data/cost_r.csv',cost_r_list, delimiter=",")
np.savetxt('data/cost_c.csv',cost_c_list, delimiter=",")