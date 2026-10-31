import mujoco
import numpy as np
import matplotlib.pyplot as plt

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

# --- wrist camera check ---
print("cameras:", [model.camera(i).name for i in range(model.ncam)])  # list camera names

cam = "wrist_camera"  # change to the wrist camera name printed above
renderer = mujoco.Renderer(model, 480, 640) # off-screen renderer, 480 x 640 image

renderer.update_scene(data, camera=cam)  # look through the wrist camera
rgb = renderer.render() # colour image

renderer.enable_depth_rendering()  # switch to depth mode
renderer.update_scene(data, camera=cam)
depth = renderer.render()   # depth image in metres
renderer.disable_depth_rendering()  # back to colour mode

print(f"depth range: {depth.min():.3f} to {depth.max():.3f} m")  # nearest and furthest points

fig, ax = plt.subplots(1, 2, figsize=(10, 4))           # make a window with 1 row x 2 plots side by side, 10 x 4 inches
ax[0].imshow(rgb);   ax[0].set_title("wrist RGB")        # left plot: show the colour image and give it a title
ax[1].imshow(depth); ax[1].set_title("wrist depth (m)")  # right plot: show the depth image (distance in metres) and give it a title
plt.show()                                               # open the window so the images appear

# --- place ball between the open fingers ---
ball_jnt = model.joint("target_ball_freejoint").id   # ball's free joint
ball_q = model.jnt_qposadr[ball_jnt]                 # where the ball's position starts in qpos (x, y, z + 4 quaternion values)
ball_v = model.jnt_dofadr[ball_jnt]                  # where the ball's velocity starts in qvel (6 values)
ball_body = model.body("target_ball_body").id        # ball body (to read its position)

data.ctrl[grip] = 0                       # open the gripper
for _ in range(500):                      # 1 s of sim time for the fingers to open
    data.qfrc_applied[:] = data.qfrc_bias # cancel gravity on everything (arm and ball) so nothing falls
    mujoco.mj_step(model, data)           # advance physics one step

mid = (data.xpos[left] + data.xpos[right]) / 2   # point halfway between the two pads
data.qpos[ball_q:ball_q+3] = mid                 # move the ball to that point
data.qvel[ball_v:ball_v+6] = 0                   # make sure the ball isn't moving
mujoco.mj_forward(model, data)                   # update positions without advancing time

print("pad midpoint:", mid.round(3))                          # where we wanted the ball
print("ball position:", data.xpos[ball_body].round(3))        # where the ball actually is
print(f"open gap: {np.linalg.norm(data.xpos[left] - data.xpos[right])*1000:.1f} mm")  # gap with fingers open

# --- close gripper on the ball and check both fingers touch it ---
ball_geom = model.geom("target_ball").id                                  # ball geom (contacts are between geoms)
left_bodies = [left, model.body("gripper_left_silicone_pad").id]          # left pad + its silicone layer
right_bodies = [right, model.body("gripper_right_silicone_pad").id]       # right pad + its silicone layer

data.ctrl[grip] = 255                     # close the gripper
for _ in range(500):                      # 1 s of sim time for the fingers to close on the ball
    data.qfrc_applied[:] = data.qfrc_bias # still cancelling gravity, so the ball stays put while gripping
    mujoco.mj_step(model, data)           # advance physics one step

left_touch = False                        # assume no contact until we find one
right_touch = False
for i in range(data.ncon):                # go through every contact happening right now
    c = data.contact[i]
    if c.geom1 == ball_geom or c.geom2 == ball_geom:            # is the ball part of this contact?
        other = c.geom2 if c.geom1 == ball_geom else c.geom1    # the geom the ball is touching
        body = model.geom_bodyid[other]                         # which body that geom belongs to
        if body in left_bodies:
            left_touch = True
        if body in right_bodies:
            right_touch = True

print("left finger touching:", left_touch)
print("right finger touching:", right_touch)
print(f"gap on ball: {np.linalg.norm(data.xpos[left] - data.xpos[right])*1000:.1f} mm")  # should stop well above the 14.9 mm closed gap