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

# def update_plot(fig, axs, vels_plot, cost_plot, vels, costs, steps):
#     for i, joint in enumerate(vels_plot):
#         joint.set_data(steps, vels[i])

#     cost_plot.set_data(steps, costs)

#     # Set plot limits
#     # self.ax1.set_xlim([max(0, len(self.time_stamps) - 100), len(self.time_stamps)])
#     # self.ax2.set_xlim([max(0, len(self.time_stamps) - 100), len(self.time_stamps)])

#     # Draw the updated plot
#     axs[0].relim()
#     axs[1].relim()
#     axs[0].autoscale_view(True, True, True)
#     axs[1].autoscale_view(True, True, True)
#     fig.canvas.draw()
#     fig.canvas.flush_events()

start_time = time.time()
cem =  cem_planner(
    num_dof=6, 
    num_batch=500, 
    num_steps=10, 
    maxiter_cem=5,
    w_pos=5,
    w_rot=2,
    w_col=100,
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

# plt.ion()
# fig, axs = plt.subplots(2, 1, figsize=(12, 6))
# axs[0].set_title('Cost')
# axs[0].set_ylabel('Cost')
# axs[1].set_title('Velocities')
# axs[1].set_ylabel('Velocity')
# axs[1].set_xlabel('Step')

# vel_plot = [axs[1].plot([], [], label=f'Joint {joint}')[0] for joint in np.arange(1,7)]
# cost_plot = axs[0].plot([], [])[0]

# axs[0].legend()
# axs[1].legend()

# vels = list()
# costs = list()
# steps = list()

# vels.append(thetadot)
# costs.append(np.linalg.norm(data.xpos[cem.hande_id] - cem.target_pos))
# steps.append(len(steps))

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
        cost_r = np.linalg.norm(data.xmat[cem.hande_id][-3:] - cem.target_rot)  
        print(f'Step Time: {round(time.time() - start_time, 2)}s | Cost g: {round(float(cost_g), 2)} | Cost r: {round(float(cost_r), 2)}')

        viewer_.sync()

        # vels.append(thetadot)
        # costs.append(cost_)
        # steps.append(len(steps))
        # update_plot(fig, axs, vel_plot, cost_plot, np.transpose(vels), costs, steps)

        time_until_next_step = model.opt.timestep - (time.time() - start_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)  