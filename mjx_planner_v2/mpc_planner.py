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


num_dof = 6
num_batch = 100
mpc_steps = 20

cem =  cem_planner(num_dof, num_batch)
model = cem.model
data = cem.data

qvel = cem.compute_cem(data.qpos[:6], data.qvel[:6])

pos = list()
vel = list()

pos.append(data.qpos[:6])
vel.append(data.qvel[:6])

with viewer.launch_passive(model, data) as viewer_:
    viewer_.cam.distance = 4
    viewer_.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

    while viewer_.is_running():
        # print(data.qvel[:6])
        qvel = cem.compute_cem(data.qpos[:6], data.qvel[:6])
        # print(f"Time: {round(time.time() - start, 2)}, Cost: {cost}")
        data.qvel[:6] = qvel
        start = time.time()
        mujoco.mj_step(model, data)
        print(data.qpos)
        pos.append(data.qpos[:6])
        vel.append(data.qvel[:6])
        viewer_.sync()

        time_until_next_step = model.opt.timestep - (time.time() - start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)  

# print(np.array(pos).T.shape)
plt.plot(np.array(pos))
plt.savefig(f'{os.path.dirname(__file__)}/plots/pos.png')
plt.clf()

plt.plot(np.array(vel))
plt.savefig(f'{os.path.dirname(__file__)}/plots/vel.png')
plt.clf()