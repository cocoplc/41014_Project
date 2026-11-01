import mujoco
import numpy as np

BALL_GEOM = "target_ball"                                           # ball geom in scene.xml
GRIPPER = "gripper_fingers_actuator"                                # gripper actuator in scene.xml
LEFT_PADS = ["gripper_left_pad", "gripper_left_silicone_pad"]       # left pad bodies
RIGHT_PADS = ["gripper_right_pad", "gripper_right_silicone_pad"]    # right pad bodies


def fingers_touching(model, data, obj_geom=BALL_GEOM):
    obj = model.geom(obj_geom).id                                   # the object's geom
    left = [model.body(pad).id for pad in LEFT_PADS]                # left pad + silicone
    right = [model.body(pad).id for pad in RIGHT_PADS]              # right pad + silicone

    left_touch = False
    right_touch = False
    for i in range(data.ncon):                                      # every contact right now
        c = data.contact[i]
        if c.geom1 == obj or c.geom2 == obj:                        # is the object in this contact?
            other = c.geom2 if c.geom1 == obj else c.geom1          # the geom it's touching
            body = model.geom_bodyid[other]                         # which body that geom belongs to
            if body in left:
                left_touch = True
            if body in right:
                right_touch = True
    return left_touch, right_touch


def close_step(model, data, rate=1, squeeze=50, obj_geom=BALL_GEOM):
    grip = model.actuator(GRIPPER).id                               # 0 = open, 255 = closed
    left_touch, right_touch = fingers_touching(model, data, obj_geom)

    if left_touch and right_touch:                                  # both fingers on the object
        data.ctrl[grip] = min(data.ctrl[grip] + squeeze, 255)       # final squeeze
        return True

    data.ctrl[grip] = min(data.ctrl[grip] + rate, 255)              # close a little more
    return False


def hold_still(model, data, float_ball=False, obj_geom=BALL_GEOM):
    data.qfrc_applied[:] = data.qfrc_bias                           # cancel gravity on everything
    if not float_ball:                                              # normal use: ball feels gravity
        ball_body = model.geom_bodyid[model.geom(obj_geom).id]
        ball_v = model.jnt_dofadr[model.body_jntadr[ball_body]]     # first DOF of the ball's free joint
        data.qfrc_applied[ball_v:ball_v+6] = 0

def finger_midpoint(model, data):
    left = model.body(LEFT_PADS[0]).id
    right = model.body(RIGHT_PADS[0]).id
    return (data.xpos[left] + data.xpos[right]) / 2.0               # midpoint of the two pads
