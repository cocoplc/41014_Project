import time
import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("scene.xml")              # load scene
data = mujoco.MjData(model)                                     # simulation state

start = [-1.57, -1.57, 1.57, -1.57, -1.57, 0]                   # starting guess: typical UR "pointing down" pose (rad)
data.qpos[:6] = start                                           # put the arm there
data.ctrl[:6] = start                                           # tell the arm motors to hold these angles
mujoco.mj_forward(model, data)                                  # update positions

with mujoco.viewer.launch_passive(model, data) as viewer:      # open the viewer window
    while viewer.is_running():                                  # loop until you close the window
        for _ in range(10):                                     # 10 physics steps...
            mujoco.mj_step(model, data)
        viewer.sync()                                           # ...then redraw
        time.sleep(0.02)                                        # run at roughly real speed

print("look pose:", data.qpos[:6].round(3).tolist())            # joint angles when you closed the window