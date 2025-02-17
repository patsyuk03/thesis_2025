import mujoco
from mujoco import viewer
from mujoco import mjx
import os
import numpy as np
import time


model_path = f"{os.path.dirname(__file__)}/../universal_robots_ur5e/scene_mjx.xml" 
model = mujoco.MjModel.from_xml_path(model_path)
model.opt.timestep = 0.01
data = mujoco.MjData(model)

file_path = f"{os.path.dirname(__file__)}/best_vels.csv" 
thetadot = np.genfromtxt(file_path, delimiter=',').T
print(thetadot.shape)

with viewer.launch_passive(model, data) as viewer_:
    viewer_.cam.distance = 4
    start = time.time()
    i = 0
    while viewer_.is_running():
        step_start = time.time()
        data.qvel[:6] = thetadot[i]
        mujoco.mj_step(model, data)
        viewer_.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        if i < thetadot.shape[0]-1:
            i+=1
        else:
            data.qvel[:6] = np.zeros(6)
            data.qpos[:6] = np.zeros(6)
            i=0

# print(data.qpos[:6])
