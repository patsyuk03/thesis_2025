import mujoco
import mujoco.viewer
import os
import time
import numpy as np

NUM_JOINTS = 6
NUM_SAMPLES = 100

model_path = f"{os.path.dirname(__file__)}/../universal_robots_ur5e/scene.xml" 
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
mujoco.mj_step(model, data)

print("EEF pose:", data.xpos[model.body(name="hande").id])
print("Obj pos:", data.xpos[model.body(name="object_0").id])

def sample_joint_velocities(num_joints, num_samples, mu=0, sigma=1):
    return np.random.normal(mu, sigma, (num_samples, num_joints))

def integrate_trajectory(q_start, velocities, dt=0.1):
    trajectory = [q_start]
    for v in velocities:
        q_next = trajectory[-1] + v * dt
        trajectory.append(q_next)
    return np.array(trajectory)



def evaluate_cost(target_position, current_position):    
    cost = np.linalg.norm(current_position - target_position)**2
    return cost


target_position = data.xpos[model.body(name="object_0").id]
step = 0
q_start = np.zeros(NUM_JOINTS)
velocities = sample_joint_velocities(NUM_JOINTS, NUM_SAMPLES)
best_cost = float('inf')
best_trajectory = None
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 4
    start = time.time()
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()
        data.qvel[:NUM_JOINTS] = velocities[0]
        mujoco.mj_step(model, data)
        viewer.sync()

        cost = evaluate_cost(target_position, data.xpos[model.body(name="hande").id])
        print(cost)
        
        # if cost < best_cost:
        #     best_cost = cost
        #     # best_trajectory = trajectory
        if step < len(velocities)-1:
            step+=1
            print(step)

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)