import mujoco
from mujoco import viewer
import os
from mujoco import mjx
import jax
import numpy as np



model_path = f"{os.path.dirname(__file__)}/ur5e_hande_mjx/scene.xml" 
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

data.qpos[:6] = np.array([1.5, -1.8, 1.75, -1.25, -1.6, 0])

# mjx_model = mjx.put_model(model)
# mjx_data = mjx.put_data(model, data)

# jax.config.update('jax_default_matmul_precision', 'high')
# jax.config.update("jax_enable_x64", True)
# jit_step = jax.jit(mjx.step)
# mjx_data = jit_step(mjx_model, mjx_data)

# x = np.where(mjx_data.contact.geom.tolist() == [1,36])
# print((np.array(mjx_data.contact.geom.tolist()) == (1,36)).all(axis=1).nonzero()[0][0])
viewer.launch(model, data)