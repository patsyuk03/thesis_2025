# Throw a ball at 100 different velocities.

import jax
import mujoco
from mujoco import mjx

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

model = mujoco.MjModel.from_xml_string(XML)
mjx_model = mjx.put_model(model)

def batched_step(vel):
  mjx_data = mjx.make_data(mjx_model)
  print(mjx_data.qvel)
  print(vel)
  print(mjx_data.qvel.at[0])
  qvel = mjx_data.qvel.at[0].set(vel)
  mjx_data = mjx_data.replace(qvel=qvel)
  print(mjx_data.qvel.at[0])
  pos = mjx.step(mjx_model, mjx_data).qpos[0]
  return pos

vel = jax.numpy.arange(0.0, 1.0, 0.1)
print(vel)
# pos = jax.jit(batched_step)(vel)
pos = jax.vmap(batched_step, in_axes=0)(vel)
print(pos)