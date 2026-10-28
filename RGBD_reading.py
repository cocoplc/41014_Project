"""
vision.py - RGB-D red-cube localisation in the WORLD frame (Day 3, Hannah)

Chain:  RGB-D image -> pixels -> camera (optical) frame point -> world-frame point

Interface agreement (team plan):
  * camera points: metres, OPTICAL frame (x right, y down, z forward)
  * target / tool poses: metres, WORLD frame

Check the names below match Chloe's scene.xml before running.
"""

import numpy as np
import mujoco
from spatialmath import SE3

CAMERA_NAME = "wrist_camera"   # <camera name=...> on the UR3e wrist
CUBE_BODY = "target_cube"      # <body name=...> of the red cube (ground truth only)
WIDTH, HEIGHT = 640, 480

# MuJoCo cameras look along -z with y up. The optical frame looks along +z
# with y down. This flips y and z to convert between the two.
R_MJ_TO_OPT = np.diag([1.0, -1.0, -1.0])


# ---------------------------------------------------------------- intrinsics
def intrinsics_from_model(model, cam_name=CAMERA_NAME, width=WIDTH, height=HEIGHT):
    """Ideal K from the camera's vertical field of view (cross-check for calibrate.py)."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    fovy = np.deg2rad(model.cam_fovy[cam_id])
    f = (height / 2) / np.tan(fovy / 2)
    return np.array([[f, 0, width / 2],
                     [0, f, height / 2],
                     [0, 0, 1.0]])


# ---------------------------------------------------------------- extrinsics
def camera_pose_world(model, data, cam_name=CAMERA_NAME):
    """T_WC: pose of the camera OPTICAL frame in the world frame, from the
    current joint state. Recomputed every frame because the camera moves with the wrist."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    R_W_mj = data.cam_xmat[cam_id].reshape(3, 3)
    p_W = data.cam_xpos[cam_id].copy()
    R_WC = R_W_mj @ R_MJ_TO_OPT
    return SE3.Rt(R_WC, p_W, check=False)


# ---------------------------------------------------------------- capture
class WristCamera:
    def __init__(self, model, cam_name=CAMERA_NAME, width=WIDTH, height=HEIGHT):
        self.model, self.cam_name = model, cam_name
        self.renderer = mujoco.Renderer(model, height, width)

    def capture(self, data):
        self.renderer.update_scene(data, camera=self.cam_name)
        self.renderer.disable_depth_rendering()
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy()   # metres along optical z
        self.renderer.disable_depth_rendering()
        return rgb, depth


# ---------------------------------------------------------------- detection
def red_mask(rgb):
    """Ratio-based threshold. Rendered red is shaded, so it is rarely (255, 0, 0)."""
    r, g, b = (rgb[..., i].astype(float) for i in range(3))
    return (r > 80) & (r > 1.6 * g) & (r > 1.6 * b)


def detect_cube_camera(rgb, depth, K, min_pixels=30):
    """Return the cube's visible-surface centroid in the optical frame (metres), or None."""
    mask = red_mask(rgb) & np.isfinite(depth) & (depth > 0)
    v, u = np.nonzero(mask)
    if len(u) < min_pixels:
        return None, None

    # Back-project every red pixel, then average in 3D. This is more robust
    # than (mean pixel, median depth) when the cube is seen at an angle.
    z = depth[v, u]
    Kinv = np.linalg.inv(K)
    pix = np.vstack([u, v, np.ones_like(u)]).astype(float)
    pts_C = (Kinv @ pix) * z            # 3 x N
    P_C = pts_C.mean(axis=1)
    return P_C, (u.mean(), v.mean())


def detect_cube_world(rgb, depth, K, T_WC):
    """Main Day 3 function: RGB-D -> world-frame target (metres)."""
    P_C, pixel = detect_cube_camera(rgb, depth, K)
    if P_C is None:
        return None, None
    P_W = (T_WC * P_C).flatten()
    return P_W, pixel


# ---------------------------------------------------------------- Day 3 test
def test_consistency_as_wrist_moves(xml_path="scene.xml", n_poses=6, seed=0):
    """Move the wrist through several poses. The world-frame target should stay put."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    cam = WristCamera(model)
    K = intrinsics_from_model(model)   # swap in your calibrated K from calibrate.py

    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY)
    rng = np.random.default_rng(seed)
    q_home = model.key_qpos[0][:6].copy() if model.nkey else data.qpos[:6].copy()

    estimates = []
    for i in range(n_poses):
        # Perturb only the wrist joints (indices 3-5), small enough to keep the cube in view
        q = q_home.copy()
        q[3:6] += rng.uniform(-0.25, 0.25, size=3)
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)

        rgb, depth = cam.capture(data)
        T_WC = camera_pose_world(model, data)
        P_W, pixel = detect_cube_world(rgb, depth, K, T_WC)
        true_W = data.xpos[cube_id]

        if P_W is None:
            print(f"pose {i}: cube not visible")
            continue
        err = np.linalg.norm(P_W - true_W)
        estimates.append(P_W)
        print(f"pose {i}: pixel=({pixel[0]:.0f},{pixel[1]:.0f})  "
              f"P_W={np.round(P_W, 4)}  true={np.round(true_W, 4)}  err={err*1000:.1f} mm")

    if len(estimates) >= 2:
        E = np.array(estimates)
        spread = np.linalg.norm(E - E.mean(axis=0), axis=1).max()
        print(f"\nMax spread across wrist poses: {spread*1000:.1f} mm")
        # The surface centroid sits about half a cube-width from the true centre,
        # so judge consistency (spread), not absolute error, today.
        assert spread < 0.01, "World target drifts with wrist motion - check T_WC / axes"
        print("PASS: world-frame target is consistent as the wrist moves")


if __name__ == "__main__":
    test_consistency_as_wrist_moves()


