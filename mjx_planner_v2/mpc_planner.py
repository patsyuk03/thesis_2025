import numpy as np
from mjx_planner import cem_planner
import mujoco.mjx as mjx 
import mujoco
import time
import jax.numpy as jnp
import jax
import os
from mujoco import viewer

num_dof = 6
num_batch = 250
mpc_steps = 20

cem =  cem_planner(num_dof, num_batch)
model = cem.model
data = cem.data

qvel = cem.compute_cem(data.qpos[:6], data.qvel[:6])

with viewer.launch_passive(model, data) as viewer_:
    viewer_.cam.distance = 4
    viewer_.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

    start = time.time()
    while viewer_.is_running():
        qvel, cost = cem.compute_cem(data.qpos[:6], data.qvel[:6])
        # step_start = time.time()
        print(f"Cost: {cost}")
        data.qvel[:6] = qvel

        mujoco.mj_step(model, data)
        viewer_.sync()

        # time_until_next_step = model.opt.timestep - (time.time() - step_start)
        # if time_until_next_step > 0:
        #     time.sleep(time_until_next_step)  