import jax
import mujoco
from mujoco import mjx
import mujoco.viewer
import logging
import time
from jax import numpy as jp
import os

model_path = f"{os.path.dirname(__file__)}/../universal_robots_ur5e/scene_mjx.xml" 

_VIEWER_GLOBAL_STATE = {
    'running': True,
    'quit': False
}

def key_callback(key: int) -> None:
    print(key)
    if key == 32:  # Space bar
        _VIEWER_GLOBAL_STATE['running'] = not _VIEWER_GLOBAL_STATE['running']
        logging.info('RUNNING = %s', _VIEWER_GLOBAL_STATE['running'])
    elif key == 81:
        _VIEWER_GLOBAL_STATE['quit'] = not _VIEWER_GLOBAL_STATE['quit']
        print("Quitting.")

def main():
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)

    print(f'Default backend: {jax.default_backend()}')

    step_fn = mjx.step
    print('JIT-compiling the model physics step...')
    start = time.time()
    step_fn = jax.jit(step_fn).lower(mx, dx).compile()
    elapsed = time.time() - start
    print(f'Compilation took {elapsed}s.')


    viewer = mujoco.viewer.launch_passive(m, d, key_callback=key_callback)
    step = 0
    with viewer:
        while True:
            step+=1
            start = time.time()
            # print(d.qpos)
            # qvel = d.qvel.at[0].set(vel)
            # d = d.replace(qvel=qvel)

            dx = dx.replace(
                ctrl=jp.array(d.ctrl),
                act=jp.array(d.act),
                xfrc_applied=jp.array(d.xfrc_applied),
            )
            dx = dx.replace(
                qpos=jp.array(d.qpos), qvel=jp.array(d.qvel), time=jp.array(d.time)
            ) 
            mx = mx.tree_replace({
                'opt.gravity': m.opt.gravity,
                'opt.tolerance': m.opt.tolerance,
                'opt.ls_tolerance': m.opt.ls_tolerance,
                'opt.timestep': m.opt.timestep,
            })

            if _VIEWER_GLOBAL_STATE['running']:
                dx = step_fn(mx, dx)


            mjx.get_data_into(d, m, dx)
            viewer.sync()

            elapsed = time.time() - start
            if elapsed < m.opt.timestep:
                time.sleep(m.opt.timestep - elapsed)


            if _VIEWER_GLOBAL_STATE['quit']:
                break



if __name__ == "__main__":
    main()