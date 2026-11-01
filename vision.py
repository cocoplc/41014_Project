import numpy as np
import mujoco
from spatialmath import SE3
 
CAMERA_NAME = "wrist_camera"
TARGET_BODY = "target_ball_body"
BALL_RADIUS = 0.025            # m
WIDTH, HEIGHT = 640, 480
 
# MuJoCo cameras look along -z with y up; the optical frame looks along +z with y down.
R_MJ_TO_OPT = np.diag([1.0, -1.0, -1.0])
 
 
# ---------------------------------------------------------------- intrinsics
def intrinsics_from_model(model, width=WIDTH, height=HEIGHT):
    """Ideal pinhole K from the camera's vertical field of view."""
    fovy = np.deg2rad(model.cam_fovy[model.camera(CAMERA_NAME).id])
    f = (height / 2) / np.tan(fovy / 2)
    return np.array([[f, 0, width / 2],
                     [0, f, height / 2],
                     [0, 0, 1.0]])
 
 
# ---------------------------------------------------------------- extrinsics
def camera_pose_world(model, data):
    """T_WC: camera OPTICAL frame in the world, from the current joint state."""
    cid = model.camera(CAMERA_NAME).id
    R_WC = data.cam_xmat[cid].reshape(3, 3) @ R_MJ_TO_OPT
    return SE3.Rt(R_WC, data.cam_xpos[cid].copy(), check=False)
 
 
# ---------------------------------------------------------------- capture
class WristCamera:
    def __init__(self, model, width=WIDTH, height=HEIGHT):
        self.renderer = mujoco.Renderer(model, height, width)
 
    def capture(self, data):
        self.renderer.update_scene(data, camera=CAMERA_NAME)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data, camera=CAMERA_NAME)
        depth = self.renderer.render().copy()      # metres along optical z
        self.renderer.disable_depth_rendering()
        return rgb, depth
 
 
# ---------------------------------------------------------------- detection
def target_mask(rgb):
    """Ball renders ~(255, 0-30, 150); floor pink ~(255, 90, 255); table ~(255, 85-210, 170+)."""
    r, g, b = (rgb[..., i].astype(float) for i in range(3))
    return (r > 60) & (g < 0.2 * r) & (b > 0.3 * r) & (b < 0.85 * r)
 
 
def detect_ball_camera(rgb, depth, K, radius=BALL_RADIUS, min_pixels=30):
    """Ball CENTRE in the optical frame (m) and its pixel centroid, or (None, None).
 
    Every ball pixel is back-projected to a 3D surface point. Pixels sample the
    visible cap evenly in image area, and the mean depth of a sphere's cap over its
    projected disc is 2r/3 in front of the centre, so the centre is pushed back by
    2r/3 along the viewing ray. This works from any wrist angle."""
    mask = target_mask(rgb) & np.isfinite(depth) & (depth > 0)
    v, u = np.nonzero(mask)
    if len(u) < min_pixels:
        return None, None
    z = depth[v, u]
    pix = np.vstack([u, v, np.ones_like(u)]).astype(float)
    P_surf = ((np.linalg.inv(K) @ pix) * z).mean(axis=1)
    P_C = P_surf + (2.0 / 3.0) * radius * P_surf / np.linalg.norm(P_surf)
    return P_C, (u.mean(), v.mean())
 
 
def detect_ball_world(rgb, depth, K, T_WC):
    """RGB-D -> world-frame ball centre (m). This is what Josie's servoing uses."""
    P_C, pixel = detect_ball_camera(rgb, depth, K)
    if P_C is None:
        return None, None
    return (T_WC * P_C).flatten(), pixel
 
 
# ---------------------------------------------------------------- Day 3 check
if __name__ == "__main__":
    from scene_setup import reset_scene, LOOK_POSE
 
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
    cam, K = WristCamera(model), intrinsics_from_model(model)
 
    true_W = reset_scene(model, data, seed=0)
    rng = np.random.default_rng(0)
    estimates = []
    for i in range(6):                                   # wrist moves, ball stays put
        data.qpos[:6] = LOOK_POSE
        data.qpos[3:6] += rng.uniform(-0.15, 0.15, 3)
        mujoco.mj_forward(model, data)
        rgb, depth = cam.capture(data)
        P_W, pixel = detect_ball_world(rgb, depth, K, camera_pose_world(model, data))
        if P_W is None:
            print(f"pose {i}: ball not visible"); continue
        estimates.append(P_W)
        print(f"pose {i}: pixel ({pixel[0]:.0f}, {pixel[1]:.0f})  P_W {P_W.round(4)}  "
              f"err {1000*np.linalg.norm(P_W - true_W):.1f} mm")
    E = np.array(estimates)
    spread = np.linalg.norm(E - E.mean(axis=0), axis=1).max()
    print(f"max spread across wrist poses: {1000*spread:.1f} mm  ->  {'PASS' if spread < 0.01 else 'FAIL'}")
    cam.renderer.close()
 