import mujoco
import numpy as np


def fingers_touching(model, data, obj_geom="target_ball"):
    obj = model.geom(obj_geom).id                                                           # the object's geom
    left_bodies = [model.body("gripper_left_pad").id, model.body("gripper_left_silicone_pad").id]     # left pad + silicone
    right_bodies = [model.body("gripper_right_pad").id, model.body("gripper_right_silicone_pad").id]  # right pad + silicone

    left_touch = False                                          # assume no contact until we find one
    right_touch = False
    for i in range(data.ncon):                                  # go through every contact happening right now
        c = data.contact[i]
        if c.geom1 == obj or c.geom2 == obj:                    # is the object part of this contact?
            other = c.geom2 if c.geom1 == obj else c.geom1      # the geom the object is touching
            body = model.geom_bodyid[other]                     # which body that geom belongs to
            if body in left_bodies:
                left_touch = True
            if body in right_bodies:
                right_touch = True
    return left_touch, right_touch                              # (left touching?, right touching?)

def close_step(model, data, rate=1, squeeze=50, obj_geom="target_ball"):
    grip = model.actuator("gripper_fingers_actuator").id       # gripper motor (0 = open, 255 = closed)
    left_touch, right_touch = fingers_touching(model, data, obj_geom)   # check contacts now

    if left_touch and right_touch:                              # both fingers are on the object
        data.ctrl[grip] = min(data.ctrl[grip] + squeeze, 255)   # close a bit further so it grips firmly (max 255)
        return True                                             # done - grasped

    data.ctrl[grip] = min(data.ctrl[grip] + rate, 255)          # not touching yet, so close a little more
    return False                                                # not grasped yet


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path("scene.xml")           # load scene
    data = mujoco.MjData(model)                                 # simulation state
    mujoco.mj_forward(model, data)                              # work out positions and contacts

    print("touching at start:", fingers_touching(model, data))  # ball is on the table, so expect (False, False)

    grip = model.actuator("gripper_fingers_actuator").id        # gripper motor
    left = model.body("gripper_left_pad").id                    # left finger pad
    right = model.body("gripper_right_pad").id                  # right finger pad
    ball_q = model.jnt_qposadr[model.joint("target_ball_freejoint").id]   # ball position start in qpos

    mid = (data.xpos[left] + data.xpos[right]) / 2              # point between the open pads
    data.qpos[ball_q:ball_q+3] = mid                            # move the ball there
    mujoco.mj_forward(model, data)                              # update positions

    for step in range(2000):                                    # up to 4 s
        data.qfrc_applied[:] = data.qfrc_bias                   # cancel gravity so nothing falls during the test
        if close_step(model, data):                             # close a bit; True once both fingers touch
            print(f"grasped after {step*0.002:.2f} s, gripper command = {data.ctrl[grip]:.0f}")
            break
        mujoco.mj_step(model, data)                             # advance physics one step

    print("touching at end:", fingers_touching(model, data))    # expect (True, True)

    ball_v = model.jnt_dofadr[model.joint("target_ball_freejoint").id]   # ball velocity start in qvel
    ball_body = model.body("target_ball_body").id                         # ball body (to read its position)

    for _ in range(250):                                        # 0.5 s for the extra squeeze to take effect
        data.qfrc_applied[:] = data.qfrc_bias                   # still cancelling gravity on everything
        mujoco.mj_step(model, data)

    offset_start = data.xpos[ball_body] - (data.xpos[left] + data.xpos[right]) / 2   # ball relative to fingers before letting go

    for _ in range(1000):                                       # 2 s with gravity on the ball
        data.qfrc_applied[:] = data.qfrc_bias                   # cancel gravity on the arm and gripper
        data.qfrc_applied[ball_v:ball_v+6] = 0                  # but not on the ball
        mujoco.mj_step(model, data)

    offset_end = data.xpos[ball_body] - (data.xpos[left] + data.xpos[right]) / 2     # ball relative to fingers after 2 s
    slip = np.linalg.norm(offset_end - offset_start)            # how far the ball moved within the grip (m)
    print(f"slip in grip: {slip*1000:.1f} mm")
    print("PASS: gradual grip holds" if slip < 0.005 else "FAIL: increase squeeze")