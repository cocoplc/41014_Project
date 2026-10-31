import mujoco
import numpy as np

LOOK_POSE = [-1.85, -1.57, 0.93, -0.93, -1.57, 0]       # arm above the table, gripper pointing down (rad)


def set_look_pose(model, data):
    data.qpos[:6] = LOOK_POSE                           # put the arm in the look pose
    data.qvel[:6] = 0                                   # arm not moving
    mujoco.mj_forward(model, data)                      # update positions and camera


def get_wrist_images(model, data, renderer):
    renderer.update_scene(data, camera="wrist_camera")  # look through the wrist camera
    rgb = renderer.render()                             # colour image (480 x 640 x 3)

    renderer.enable_depth_rendering()                   # switch to depth mode
    renderer.update_scene(data, camera="wrist_camera")
    depth = renderer.render()                           # depth image in metres (480 x 640)
    renderer.disable_depth_rendering()                  # back to colour mode
    return rgb, depth


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path("scene.xml")   # load scene
    data = mujoco.MjData(model)                         # simulation state
    renderer = mujoco.Renderer(model, 480, 640)         # off-screen renderer

    set_look_pose(model, data)                          # arm to look pose
    rgb, depth = get_wrist_images(model, data, renderer)        # grab both images
    print("rgb:", rgb.shape, " depth:", depth.shape)            # expect (480, 640, 3) and (480, 640)
    print(f"depth range: {depth.min():.3f} to {depth.max():.3f} m")   # should match your look-pose check