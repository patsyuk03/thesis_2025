import mujoco
from mujoco import viewer
from mujoco import mjx
import os

model_path = f"{os.path.dirname(__file__)}/../universal_robots_ur5e/scene_mjx.xml" 

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path(model_path)
model.opt.disableactuator = 0

# Create a data structure for simulation
data = mujoco.MjData(model)
# mx = mjx.put_model(model)
# dx = mjx.put_data(model, data)

mjx_data = mjx.put_data(model, data)

print(mjx_data.qpos[:7])




# Launch the viewer to visualize the model
viewer.launch(model, data)


print(mjx_data.qpos[:7])

