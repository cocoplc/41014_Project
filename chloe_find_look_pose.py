import time
import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("scene.xml")              # load scene
data = mujoco.MjData(model)                                     # simulation state

pose = [-1.85, -1.57, 0.93, -0.93, -1.57, 0]                    # pan swung toward table centre, forearm tilted up, wrist1 keeps gripper pointing down                # starting guess: typical UR "pointing down" pose (rad)
data.qpos[:6] = pose                                            # put the arm there
mujoco.mj_forward(model, data)                                  # update positions

left = model.body("gripper_left_pad").id                        # left finger pad
right = model.body("gripper_right_pad").id                      # right finger pad
grip_pos = (data.xpos[left] + data.xpos[right]) / 2             # point between the fingers
target = np.array([0.04, -0.29, 0.40])                          # table centre, 28 cm above the tabletop (0.12)
print("gripper:", grip_pos.round(3))
print("target: ", target)
print("error:  ", (target - grip_pos).round(3))                 # how far off in x, y, z (m)

with mujoco.viewer.launch_passive(model, data) as viewer:      # open the viewer to look at it
    while viewer.is_running():
        mujoco.mj_forward(model, data)                          # no physics, so nothing falls
        viewer.sync()
        time.sleep(0.02)