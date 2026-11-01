import mujoco
import numpy as np

BALL_GEOM = "target_ball"  # name of the ball geom in the scene.xml file
GRIPPER = "gripper_fingers_actuator"  # name of the gripper actuator in the scene.xml file
LEFT_PADS = ["gripper_left_pad", "gripper_left_silicone_pad"]  # names of the left gripper pads in the scene.xml file
RIGHT_PADS = ["gripper_right_pad", "gripper_right_silicone_pad"]  # names of the right gripper pads in the scene.xml file

def fingers_touching(model, data, obj_geom=BALL_GEOM):
   ball = model.geom(obj_geom).id                                      # ball geom
   left = [model.body(pad).id for pad in LEFT_PADS]                    # left finger pads
   right = [model.body(pad).id for pad in RIGHT_PADS]                  # right finger pads

   left_touch = False
   right_touch = False

   for i in range(data.ncon):                                    # check all contacts
    c = data.contact[i]                                        # get contact info
    if ball not in (c.geom1, c.geom2):
        continue                                               # not touching the ball
    other = c.geom2 if c.geom1 == ball else c.geom1            # the other geom in contact
    body = model.geom_bodyid[other]                            # which body that geom belongs to

    if body in left:                                           # is it a left finger pad?
        left_touch = True
    elif body in right:                                        # is it a right finger pad?
        right_touch = True
   return left_touch, right_touch



def close_step(model, data, rate=1, squeeze=50, obj_geom=BALL_GEOM):
    grip = model.actuator(GRIPPER).id       # gripper motor (0 = open, 255 = closed)
    left_touch, right_touch = fingers_touching(model, data, obj_geom)   # check contacts now

    if left_touch and right_touch:                              # both fingers are on the object
        data.ctrl[grip] = min(data.ctrl[grip] + squeeze, 255)   # close a bit further so it grips firmly (max 255)
        return True                                             # done - grasped

    data.ctrl[grip] = min(data.ctrl[grip] + rate, 255)          # not touching yet, so close a little more
    return False                                                # not grasped yet

def hold_still(data):
    data.qfrc_applied[:] = data.qfrc_bias   # cancel gravity on everything
    
def finger_midpoint(model, data):
    left = model.body(LEFT_PADS[0]).id                    # left finger pad
    right = model.body(RIGHT_PADS[0]).id                  # right finger pad
    mid = (data.xpos[left] + data.xpos[right]) / 2.0   # midpoint of the two finger pads
    return mid