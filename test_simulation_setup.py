import mujoco
import mujoco.viewer
import os
import time


model_path = f"{os.path.dirname(__file__)}/universal_robots_ur5e/dual_arm_gripper_scene.xml" 
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 4
    start = time.time()
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)