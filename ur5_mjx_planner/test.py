# Throw a ball at 100 different velocities.

import jax
import mujoco
from mujoco import mjx
import mujoco.viewer
import logging
import time
from jax import numpy as jp

XML=r"""
<mujoco>
  <worldbody>
    <body>
      <freejoint/>
      <geom size=".15" mass="1" type="sphere"/>
    </body>
  </worldbody>
</mujoco>
"""

# model = mujoco.MjModel.from_xml_string(XML)
# mjx_model = mjx.put_model(model)

# @jax.vmap
# def batched_step(vel):
#   mjx_data = mjx.make_data(mjx_model)
#   qvel = mjx_data.qvel.at[0].set(vel)
#   mjx_data = mjx_data.replace(qvel=qvel)
#   pos = mjx.step(mjx_model, mjx_data).qpos[0]
#   return pos

# vel = jax.numpy.arange(0.0, 1.0, 0.01)
# pos = jax.jit(batched_step)(vel)

# print(pos)

_VIEWER_GLOBAL_STATE = {
    'running': True,
}

def key_callback(key: int) -> None:
    if key == 32:  # Space bar
        _VIEWER_GLOBAL_STATE['running'] = not _VIEWER_GLOBAL_STATE['running']
        logging.info('RUNNING = %s', _VIEWER_GLOBAL_STATE['running'])

def main():
    m = mujoco.MjModel.from_xml_string(XML)
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


    vel = jp.arange(0.0, 1.0, 0.01)
    i = 0

    viewer = mujoco.viewer.launch_passive(m, d, key_callback=key_callback)
    with viewer:
        while True:
            start = time.time()
            # print(d.qvel)
            # qvel = d.qvel.at[0].set(vel)
            # d = d.replace(qvel=qvel)

            dx = dx.replace(
                ctrl=jp.array(d.ctrl),
                act=jp.array(d.act),
                xfrc_applied=jp.array(d.xfrc_applied),
            )
            d.qvel[0] = vel[i]
            i+=1
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
    

if __name__ == "__main__":
    main()