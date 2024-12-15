import multiprocessing
import mujoco
import mujoco.viewer

import time
import os


def run_simulation(model_file):
    model = mujoco.MjModel.from_xml_path(model_file)
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


if __name__ == '__main__':
    model_path = f"{os.path.dirname(__file__)}/../universal_robots_ur5e/scene.xml" 
    model_files = [model_path, model_path, model_path]
    with multiprocessing.Pool(processes=3) as pool:
        pool.map(run_simulation, model_files)
