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

def rotation_quaternion(angle_deg, axis):
    axis = axis / np.linalg.norm(axis)
    angle_rad = np.deg2rad(angle_deg)
    w = np.cos(angle_rad / 2)
    x, y, z = axis * np.sin(angle_rad / 2)
    return (round(w, 5), round(x, 5), round(y, 5), round(z, 5))

def quaternion_multiply(q1, q2):
		w1, x1, y1, z1 = q1
		w2, x2, y2, z2 = q2
		
		w = w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1
		x = w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1
		y = w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1
		z = w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1
		
		return (round(w, 5), round(x, 5), round(y, 5), round(z, 5))

start_time = time.time()
cem =  cem_planner(
    num_dof=6, 
    num_batch=4000, 
    num_steps=10, 
    maxiter_cem=2,
    w_pos=5,
    w_rot=1.5,
    w_col=15,
    num_elite=0.05,
    timestep=0.05
    )
print(f"Initialized CEM Planner: {round(time.time()-start_time, 2)}s")

model = cem.model
data = cem.data
data.qpos[:6] = jnp.array([1.5, -1.8, 1.75, -1.25, -1.6, 0])
mujoco.mj_forward(model, data)

xi_mean = jnp.zeros(cem.nvar)
target_pos = model.body(name="target").pos
target_rot = model.body(name="target").quat


start_time = time.time()
_ = cem.compute_cem(xi_mean, data.qpos[:6], data.qvel[:6], data.qacc[:6], target_pos, target_rot)
print(f"Compute CEM: {round(time.time()-start_time, 2)}s")


thetadot = np.array([0]*6)

cost_g_list = list()
cost_list = list()
cost_r_list = list()
cost_c_list = list()
thetadot_list = list()
theta_list = list()

init_position = data.xpos[model.body(name="hande").id].copy()
init_rotation = data.xquat[model.body(name="hande").id].copy()

target_positions = [
    [-0.3, 0.3, 0.8],
    [-0.2, -0.4, 1.0],
    [-0.3, -0.1, 0.8],
    init_position
]

target_rotations = [
    rotation_quaternion(-135, np.array([1,0,0])),
    quaternion_multiply(rotation_quaternion(90, np.array([0,0,1])),rotation_quaternion(135, np.array([1,0,0]))),
    quaternion_multiply(rotation_quaternion(180, np.array([0,0,1])),rotation_quaternion(-90, np.array([0,1,0]))),
    init_rotation
]

target_idx = 0

with viewer.launch_passive(model, data) as viewer_:
    viewer_.cam.distance = 4
    viewer_.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    viewer_.opt.sitegroup[:] = False  
    viewer_.opt.sitegroup[1] = True 

    while viewer_.is_running():
        start_time = time.time()
        target_pos = model.body(name="target").pos
        target_rot = model.body(name="target").quat

        cost, best_cost_g, best_cost_c, best_vels, best_traj, xi_mean = cem.compute_cem(xi_mean, data.qpos[:6], data.qvel[:6], data.qacc[:6], target_pos, target_rot)
        thetadot = np.mean(best_vels[1:5], axis=0)
        # thetadot = best_vels[1]

        data.qvel[:6] = thetadot
        mujoco.mj_step(model, data)

        cost_g = np.linalg.norm(data.xpos[cem.hande_id] - target_pos)   
        cost_r = quaternion_distance(data.xquat[cem.hande_id], target_rot)  
        cost = np.round(cost, 2)
        print(f'Step Time: {"%.0f"%((time.time() - start_time)*1000)}ms | Cost g: {"%.2f"%(float(cost_g))} | Cost r: {"%.2f"%(float(cost_r))} | Cost c: {"%.2f"%(float(best_cost_c))} | Cost: {cost}')
        viewer_.sync()

        if cost_g<0.03 and cost_r<0.3:
            model.body(name="target").pos = target_positions[target_idx]
            model.body(name="target").quat = target_rotations[target_idx]
            if target_idx<len(target_positions)-1:
                target_idx += 1

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