import mujoco
import numpy as np

LOOK_POSE = [-1.85, -1.57, 0.93, -0.93, -1.57, 0]       # arm above the table, gripper pointing down (rad)
TABLE_CENTRE = np.array([0.04, -0.29])                  # table centre (x, y)
BALL_Z = 0.145                                          # ball centre height: table top 0.12 + radius 0.025

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

def randomise_ball(model, data, seed):
    rng = np.random.default_rng(seed)                   # random generator. Same seed --> same numbers
    x = TABLE_CENTRE[0] + rng.uniform(-0.10, 0.10)      # random x within +- 10 cm of table centre
    y = TABLE_CENTRE[1] + rng.uniform(-0.08, 0.08)      # random y within +- 8 cm of table centre
    adr = model.jnt_qposadr[model.joint("target_ball_freejoint").id]   # where the ball's values start in qpos
    data.qpos[adr:adr+3] = [x, y, BALL_Z]               # ball position
    data.qpos[adr+3:adr+7] = [1, 0, 0, 0]               # ball orientation - no rotation
    mujoco.mj_forward(model, data)                      # update positions without advancing time
    return np.array([x, y, BALL_Z])                     # true ball position


def reset_scene(model, data, seed):
    mujoco.mj_resetData(model, data)                    # back to the starting state
    set_look_pose(model, data)                          # arm above the table, camera looking down
    return randomise_ball(model, data, seed)            # place the ball, return its true position


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path("scene.xml")   # load scene
    data = mujoco.MjData(model)                         # simulation state
    renderer = mujoco.Renderer(model, 480, 640)         # off-screen renderer

    set_look_pose(model, data)                          # arm to look pose
    rgb, depth = get_wrist_images(model, data, renderer)        # grab both images
    print("rgb:", rgb.shape, " depth:", depth.shape)            # expect (480, 640, 3) and (480, 640)
    print(f"depth range: {depth.min():.3f} to {depth.max():.3f} m")   # should match your look-pose check
    
    for seed in range(3):                               # try 3 different seeds
        ball = reset_scene(model, data, seed)           # fresh scene with the ball placed
        rgb, depth = get_wrist_images(model, data, renderer)
        print(f"seed {seed}: ball at {ball.round(3)}, depth range {depth.min():.3f} to {depth.max():.3f} m")