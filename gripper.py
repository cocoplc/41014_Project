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


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path("scene.xml")           # load scene
    data = mujoco.MjData(model)                                 # simulation state
    mujoco.mj_forward(model, data)                              # work out positions and contacts

    print("touching at start:", fingers_touching(model, data))  # ball is on the table, so expect (False, False)