import mujoco
from mujoco import viewer
from mujoco import mjx
import os
import numpy as np
import time
import jax

def quaternion_distance(q1, q2):
    dot_product = np.abs(np.dot(q1, q2))
    return 2 * np.arccos(dot_product)


model_path = f"{os.path.dirname(__file__)}/ur5e_hande_mjx/scene.xml" 
model = mujoco.MjModel.from_xml_path(model_path)
model.opt.timestep = 0.04
data = mujoco.MjData(model)

init_joint_state = [1.5, -1.8, 1.75, -1.25, -1.6, 0]
# init_joint_state = [-1.07386628, -2.21057648, -2.35288999, -1.76991373, -1.09578535, -2.99189036]
# init_joint_state = [1.3, -0.8, 0, 0, 0, 0]
data.qpos[:6] = init_joint_state
data.ctrl[:6] = init_joint_state

mjx_model = mjx.put_model(model)
mjx_data = mjx.put_data(model, data)

# jit_step = jax.jit(mjx.step)

file_path = f"{os.path.dirname(__file__)}/data/best_vels.csv" 
thetadot = np.genfromtxt(file_path, delimiter=',')
# thetadot = np.tile(np.zeros(6), (300, 1))
# thetadot[:,2] = -0.9
# print(thetadot[1])

geom_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'robot_{i}') for i in range(10)]) # [33  7 12 13 18 19 23 27 28 30]

with viewer.launch_passive(model, data) as viewer_:
    viewer_.cam.distance = 4
    viewer_.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

    start = time.time()
    i = 0
    while viewer_.is_running():
        step_start = time.time()
        # data.qvel[:6] = thetadot[i]
        data.qpos[:6] = data.ctrl[:6]

        # quat_gripper = data.xquat[model.body(name="hande").id]

        # qpos = mjx_data.qpos.at[:6].set(data.ctrl[:6])
        # mjx_data = mjx_data.replace(qpos=qpos)
        # mjx_data = jit_step(mjx_model, mjx_data)

        rot_matrix = data.xmat[model.body(name="hande").id].reshape(3,3)
        # print(mjx_data.xmat[model.body(name="hande").id].shape)
        # rot_matrix = log_map_rotation_matrix(rot_matrix)
        rot_matrix = np.round(rot_matrix, 1)
        # rot_matrix = data.xmat[model.body(name="wrist_3_link_1").id].reshape(3,3).T @ rot_matrix
        print(rot_matrix)

        mujoco.mj_step(model, data)
        viewer_.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)    

        # if i < thetadot.shape[0]-1:
        #     i+=1
        # else:
        #     data.qvel[:6] = np.zeros(6)
        #     data.qpos[:6] = init_joint_state
        #     # mjx_data = mjx.put_data(model, data)
        #     i=0