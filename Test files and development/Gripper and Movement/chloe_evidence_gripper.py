import os
import mujoco
import numpy as np
import matplotlib.pyplot as plt
from gripper import fingers_touching, close_step       # reuse your gripper functions

os.makedirs("evidence", exist_ok=True)                  # evidence folder

model = mujoco.MjModel.from_xml_path("scene.xml")       # load scene
data = mujoco.MjData(model)                             # simulation state
mujoco.mj_forward(model, data)                          # update positions

grip = model.actuator("gripper_fingers_actuator").id    # gripper motor
left = model.body("gripper_left_pad").id                # left finger pad
right = model.body("gripper_right_pad").id              # right finger pad
ball_q = model.jnt_qposadr[model.joint("target_ball_freejoint").id]   # ball position start in qpos
ball_v = model.jnt_dofadr[model.joint("target_ball_freejoint").id]    # ball velocity start in qvel
ball_body = model.body("target_ball_body").id           # ball body

lines = []

data.qpos[ball_q:ball_q+3] = (data.xpos[left] + data.xpos[right]) / 2   # ball between the open pads
mujoco.mj_forward(model, data)

for step in range(2000):                                # gradual close, up to 4 s
    data.qfrc_applied[:] = data.qfrc_bias               # cancel gravity during the close
    if close_step(model, data):                         # True once both fingers touch
        lines.append(f"grasped after {step*0.002:.2f} s, gripper command = {data.ctrl[grip]:.0f}")
        break
    mujoco.mj_step(model, data)

lines.append(f"fingers touching (left, right): {fingers_touching(model, data)}")

for _ in range(250):                                    # 0.5 s for the squeeze to take effect
    data.qfrc_applied[:] = data.qfrc_bias
    mujoco.mj_step(model, data)

offset_start = data.xpos[ball_body] - (data.xpos[left] + data.xpos[right]) / 2   # ball relative to fingers
for _ in range(1000):                                   # 2 s with gravity on the ball
    data.qfrc_applied[:] = data.qfrc_bias
    data.qfrc_applied[ball_v:ball_v+6] = 0
    mujoco.mj_step(model, data)
offset_end = data.xpos[ball_body] - (data.xpos[left] + data.xpos[right]) / 2
slip = np.linalg.norm(offset_end - offset_start)        # movement within the grip (m)

lines.append(f"slip in grip after 2 s: {slip*1000:.1f} mm")
lines.append("PASS: ball held by finger contact" if slip < 0.005 else "FAIL")

cam = mujoco.MjvCamera()                                # a viewing camera we can aim ourselves
cam.lookat[:] = (data.xpos[left] + data.xpos[right]) / 2   # aim at the gripper
cam.distance = 0.3                                      # 30 cm away
cam.azimuth = 90                                        # view from the side
cam.elevation = -20                                     # slightly from above
renderer = mujoco.Renderer(model, 480, 640)
renderer.update_scene(data, camera=cam)
plt.imsave("evidence/gripper_holding_ball.png", renderer.render())    # picture of the grasp

for line in lines:
    print(line)
with open("evidence/gripper_contact.txt", "w") as f:    # save results
    f.write("\n".join(lines))