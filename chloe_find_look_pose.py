import time
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

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

renderer = mujoco.Renderer(model, 480, 640)             # off-screen renderer, 480 x 640 image
renderer.update_scene(data, camera="wrist_camera")      # look through the wrist camera
rgb = renderer.render()                                 # colour image

renderer.enable_depth_rendering()                       # switch to depth mode
renderer.update_scene(data, camera="wrist_camera")
depth = renderer.render()                               # depth image in metres
renderer.disable_depth_rendering()                      # back to colour mode

print(f"depth range: {depth.min():.3f} to {depth.max():.3f} m")   # table should now be the far point, not 46 m

fig, ax = plt.subplots(1, 2, figsize=(10, 4))                     # 1 row x 2 plots side by side
ax[0].imshow(rgb); ax[0].set_title("wrist RGB at look pose")    # colour image
ax[1].imshow(depth, vmax=1); ax[1].set_title("wrist depth (m)")           # depth, colours capped at 1 m
plt.show()                                                        # close this window to open the viewer