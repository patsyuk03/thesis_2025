import mujoco  # Import MuJoCo (or dm_control if using DeepMind's version)
import time
import os
import json
import mujoco.viewer
import numpy as np

def get_joint_positions(data, step, recorded_path):
    joint_positions = recorded_path[step]['position']
    names = recorded_path[step]['name']

    positions = dict()
    for idx, name in enumerate(names):
        positions[name] = joint_positions[idx]

    data.qpos[:] = [
        positions['arm_0_shoulder_pan_joint'],
        positions['arm_0_shoulder_lift_joint'],
        positions['arm_0_elbow_joint'],
        positions['arm_0_wrist_1_joint'],
        positions['arm_0_wrist_2_joint'],
        positions['arm_0_wrist_3_joint'],
    ]
    return data


JSON_PATH = os.path.join(os.path.expanduser("~"), "joint_states_arm_0.json")
with open(JSON_PATH, "r") as f:
    recorded_path = list(reversed(json.load(f)))

model_path = f"{os.path.dirname(__file__)}/universal_robots_ur5e/scene.xml" 
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

data = get_joint_positions(data, 0, recorded_path)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 4
    start = time.time()
    step = 0
    while viewer.is_running() and time.time() - start < 120:
        data = get_joint_positions(data, step, recorded_path)
        if step < len(recorded_path)-1:
            step+=1
        
        mujoco.mj_step(model, data)
        viewer.sync()

print("Playback finished!")
