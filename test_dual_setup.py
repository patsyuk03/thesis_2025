import mujoco
import mujoco.viewer
import os
import time
import json

JSON_PATH_0 = os.path.join(os.path.expanduser("~"), "joint_states_arm_0.json")
with open(JSON_PATH_0, "r") as f:
    recorded_path_0 = list(reversed(json.load(f)))
JSON_PATH_1 = os.path.join(os.path.expanduser("~"), "joint_states_arm_1.json")
with open(JSON_PATH_1, "r") as f:
    recorded_path_1 = list(reversed(json.load(f)))

def get_joint_positions(data, step):
    joint_positions = recorded_path_0[step]['position'] + recorded_path_1[step]['position']
    names = recorded_path_0[step]['name'] + recorded_path_1[step]['name']

    positions = dict()
    for idx, name in enumerate(names):
        positions[name] = joint_positions[idx]

    data.qpos[:6] = [
        positions['arm_0_shoulder_pan_joint'],
        positions['arm_0_shoulder_lift_joint'],
        positions['arm_0_elbow_joint'],
        positions['arm_0_wrist_1_joint'],
        positions['arm_0_wrist_2_joint'],
        positions['arm_0_wrist_3_joint'],
    ]
    data.qpos[8:14] = [
        positions['arm_1_shoulder_pan_joint'],
        positions['arm_1_shoulder_lift_joint'],
        positions['arm_1_elbow_joint'],
        positions['arm_1_wrist_1_joint'],
        positions['arm_1_wrist_2_joint'],
        positions['arm_1_wrist_3_joint'],
    ]
    return data

model_path = f"{os.path.dirname(__file__)}/universal_robots_ur5e/dual_arm_gripper_scene.xml" 
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]

data = get_joint_positions(data, 0)


with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 4
    start = time.time()
    step = 0
    while viewer.is_running() and time.time() - start < 120:

        data = get_joint_positions(data, step)

        if step < len(recorded_path_0)-2:
            step+=1
        
        mujoco.mj_step(model, data)
        viewer.sync()
        # if step%20==0:
        #     viewer.sync()