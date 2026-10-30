import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("scene.xml")              # load scene
data = mujoco.MjData(model)                                     # simulation state

# print every gripper body and every actuator, so we know the names
for i in range(model.nbody):
    if model.body(i).name.startswith("gripper"):
        print("body:", model.body(i).name)
print("actuators:", [model.actuator(i).name for i in range(model.nu)])