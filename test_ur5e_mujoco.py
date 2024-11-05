import mujoco  # Import MuJoCo (or dm_control if using DeepMind's version)
import time
import os
import json
import mujoco.viewer
import numpy as np


JSON_PATH = os.path.join(os.path.expanduser("~"), "joint_states.json")
with open(JSON_PATH, "r") as f:
    recorded_path = json.load(f)

# Load your MuJoCo model
model_path = f"{os.path.dirname(__file__)}/universal_robots_ur5e/scene.xml" 
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

initial_joint_positions = recorded_path[-1]['position']
data.qpos[:] = initial_joint_positions

# model.opt.timestep = 0.0001

# Initialize viewer
viewer = mujoco.viewer.launch_passive(model, data)
# viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1
renderer = mujoco.Renderer(model)

# Play through each recorded joint state
# tolerance = np.zeros(model.njnt)+0.1
counter = 0
for joint_position in reversed(recorded_path):
    # target = joint_position['position']
    data.ctrl[:] = joint_position['position'] 
    # delta = np.absolute(data.qpos[:]-target)
    # steps = 0
    # while not (delta>tolerance).all():
        # delta = data.qpos[:]-target
    mujoco.mj_step(model, data)
    counter+=1
    if counter>20:
        counter = 0
        viewer.sync()
        # time.sleep(0.01) 
        # steps += 1
        # if steps>100:
        #     print('Max steps')
        #     break

    # time.sleep(0.001) 

print("Playback finished!")
