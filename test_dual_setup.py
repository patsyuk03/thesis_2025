import mujoco
import mujoco.viewer
import os
import time
import json

JSON_PATH = os.path.join(os.path.expanduser("~"), "joint_states.json")
with open(JSON_PATH, "r") as f:
    recorded_path = list(reversed(json.load(f)))

model_path = f"{os.path.dirname(__file__)}/universal_robots_ur5e/dual_arm_scene.xml" 
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

initial_joint_positions = recorded_path[1]['position']
data.qpos[:6] = initial_joint_positions
data.qpos[6:] = initial_joint_positions

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 4  # Camera distance
    start = time.time()
    step = 0
    while viewer.is_running() and time.time() - start < 120:
        data.ctrl[:6] = recorded_path[step]['position'] 
        data.ctrl[6:] = recorded_path[step]['position'] 
        if step < len(recorded_path)-1:
            step+=1
        
        mujoco.mj_step(model, data)
        if step%20==0:
            viewer.sync()