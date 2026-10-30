import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("scene.xml")              # load scene
data = mujoco.MjData(model)                                     # simulation state

# print every gripper body and every actuator, so we know the names
for i in range(model.nbody):
    if model.body(i).name.startswith("gripper"):
        print("body:", model.body(i).name)
print("actuators:", [model.actuator(i).name for i in range(model.nu)])

grip = model.actuator("gripper_fingers_actuator").id  # gripper motor (0 = open, 255 = closed)
left = model.body("gripper_left_pad").id # left finger pad
right = model.body("gripper_right_pad").id # right finger pad

data.ctrl[grip] = 255    # set gripper command to fully closed (0 = open, 255 = closed)

for _ in range(500):   # repeat 500 times = 1 s of sim time (0.002 s per step)
    data.qfrc_applied[:] = data.qfrc_bias  # push back against gravity on every joint so the arm stays still 
    mujoco.mj_step(model, data)    # advance physics one step so the fingers move

gap = np.linalg.norm(data.xpos[left] - data.xpos[right]) # straight-line distance between the two finger pads (m)
print(f"closed gap: {gap*1000:.1f} mm")  # print the gap in mm to 1 decimal place
print("arm joints:", data.qpos[:6].round(3))  # first 6 joints = arm; should all stay near 0