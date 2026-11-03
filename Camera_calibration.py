import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
 
from scene_setup import reset_scene, LOOK_POSE
from vision import (WristCamera, camera_pose_world, intrinsics_from_model,
                    detect_ball_camera, target_mask, TARGET_BODY)
 
out = "evidence/vision"
test_seeds = range(0, 20)         # ball positions used for testing
calib_seeds = range(100, 140)     # different positions, only used for calibration
wrist_jitter = 0.15               # rad: how much the wrist is moved in some trials
noise_levels = [0.0, 0.001, 0.002, 0.005, 0.01]   # depth noise (m)
 
 
def set_up(model, data, seed, move_wrist):
    """Place the ball (Chloe's reset), optionally move the wrist, return the true ball centre."""
    reset_scene(model, data, seed)
    if move_wrist:
        rng = np.random.default_rng(1000 + seed)
        data.qpos[:6] = LOOK_POSE
        data.qpos[3:6] += rng.uniform(-wrist_jitter, wrist_jitter, 3)
        mujoco.mj_forward(model, data)
    return data.body(TARGET_BODY).xpos.copy()
 
 
def rmse(errors):
    return float(np.sqrt(np.mean(np.square(errors))))
 
 
# ---------------------------------------------------------------- 1. calibration
def calibrate(model, data, cam):
    """Pinhole model: u = fx * X/Z + cx,  v = fy * Y/Z + cy.
    (X, Y, Z) = true ball centre in the camera frame, (u, v) = detected ball pixel.
    Each equation is a straight line, so fx, cx and fy, cy come from a line fit."""
    xz, yz, us, vs = [], [], [], []
    for seed in calib_seeds:
        for move_wrist in (False, True):
            true_ball = set_up(model, data, seed, move_wrist)
            rgb, _ = cam.capture(data)
            v, u = np.nonzero(target_mask(rgb))
            if len(u) < 30:                      # ball not visible enough
                continue
            X, Y, Z = (camera_pose_world(model, data).inv() * true_ball).flatten()
            xz.append(X / Z); yz.append(Y / Z); us.append(u.mean()); vs.append(v.mean())
 
    fx, cx = np.polyfit(xz, us, 1)               # straight-line fit: slope and intercept
    fy, cy = np.polyfit(yz, vs, 1)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
 
    u_pred = fx * np.array(xz) + cx
    v_pred = fy * np.array(yz) + cy
    reproj = np.hypot(u_pred - us, v_pred - vs)  # pixel error of the fitted model
    return K, reproj, (us, vs, u_pred, v_pred)
 
 
def calibrate_depth(model, data, cam, K):
    """Check the depth readings against known distances: fit measured = scale * true.
    For each view, look at the pixel where the ball's centre appears. Along that ray the
    camera sees the front of the ball, so the true depth there is Z * (1 - r / distance)."""
    radius = model.geom("target_ball").size[0]
    true_depths, measured_depths = [], []
    for seed in calib_seeds:
        for move_wrist in (False, True):
            true_ball = set_up(model, data, seed, move_wrist)
            _, depth = cam.capture(data)
            X, Y, Z = (camera_pose_world(model, data).inv() * true_ball).flatten()
            u = int(round(K[0, 0] * X / Z + K[0, 2]))       # pixel where the centre appears
            v = int(round(K[1, 1] * Y / Z + K[1, 2]))
            if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
                continue                                     # ball outside the image
            distance = np.sqrt(X**2 + Y**2 + Z**2)
            true_depths.append(Z * (1 - radius / distance))  # depth of the ball's front surface
            measured_depths.append(depth[v, u])
 
    true_depths = np.array(true_depths)
    measured_depths = np.array(measured_depths)
    scale = np.sum(measured_depths * true_depths) / np.sum(true_depths**2)   # best-fit scale
    depth_err = measured_depths - true_depths
    return scale, depth_err
 
 
# ---------------------------------------------------------------- 2. accuracy
def localisation_errors(model, data, cam, K, noise=0.0):
    """Error (m) for every test position, with and without moving the wrist."""
    rng = np.random.default_rng(42)
    errors = []
    for move_wrist in (False, True):
        for seed in test_seeds:
            true_ball = set_up(model, data, seed, move_wrist)
            rgb, depth = cam.capture(data)
            depth = depth + rng.normal(0, noise, depth.shape) if noise > 0 else depth
            P_C, _ = detect_ball_camera(rgb, depth, K)
            if P_C is None:
                continue
            estimate = (camera_pose_world(model, data) * P_C).flatten()
            errors.append(np.linalg.norm(estimate - true_ball))
    return np.array(errors)
 
 
def main():
    os.makedirs(out, exist_ok=True)
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
    cam = WristCamera(model)
 
    # 1. Calibration
    K_ideal = intrinsics_from_model(model)
    K_cal, reproj, (us, vs, u_pred, v_pred) = calibrate(model, data, cam)
    np.save(f"{out}/K_calibrated.npy", K_cal)
    depth_scale, depth_err = calibrate_depth(model, data, cam, K_cal)
 
    # 2. Accuracy
    err_ideal = localisation_errors(model, data, cam, K_ideal)
    err_cal = localisation_errors(model, data, cam, K_cal)
 
    # 3. Depth noise
    noise_rmse = [rmse(localisation_errors(model, data, cam, K_cal, s)) for s in noise_levels]
 
    total = 2 * len(test_seeds)
    summary = "\n".join([
        "VISION EVIDENCE",
        "",
        "1. Camera calibration",
        f"   ideal K (fovy 75):  fx {K_ideal[0,0]:.2f}  fy {K_ideal[1,1]:.2f}  cx {K_ideal[0,2]:.2f}  cy {K_ideal[1,2]:.2f}",
        f"   calibrated K:       fx {K_cal[0,0]:.2f}  fy {K_cal[1,1]:.2f}  cx {K_cal[0,2]:.2f}  cy {K_cal[1,2]:.2f}",
        f"   reprojection error: RMSE {rmse(reproj):.2f} px, max {reproj.max():.2f} px ({len(us)} points)",
        "",
        "   Depth calibration (measured = scale x true)",
        f"   scale {depth_scale:.4f}, mean error {1000*np.mean(depth_err):+.2f} mm",
        f"   depth error: RMSE {1000*rmse(depth_err):.2f} mm, max {1000*np.abs(depth_err).max():.2f} mm ({len(depth_err)} points)",
        "",
        "2. 3D localisation error",
        f"   ideal K:      detected {len(err_ideal)}/{total}, RMSE {1000*rmse(err_ideal):.2f} mm, max {1000*err_ideal.max():.2f} mm",
        f"   calibrated K: detected {len(err_cal)}/{total}, RMSE {1000*rmse(err_cal):.2f} mm, max {1000*err_cal.max():.2f} mm",
        "",
        "3. Depth noise (calibrated K)",
        *[f"   noise {1000*s:4.1f} mm -> RMSE {1000*e:.2f} mm" for s, e in zip(noise_levels, noise_rmse)],
    ])
    with open(f"{out}/summary.txt", "w") as f:
        f.write(summary)
    print(summary)
 
    # Plots
    plt.figure(figsize=(7, 4))
    plt.hist(1000 * err_cal, bins=12)
    plt.xlabel("3D position error (mm)"); plt.ylabel("trials")
    plt.title("Ball localisation error (calibrated K)")
    plt.tight_layout(); plt.savefig(f"{out}/error_hist.png", dpi=150); plt.close()
 
    plt.figure(figsize=(7, 4))
    plt.plot(1000 * np.array(noise_levels), 1000 * np.array(noise_rmse), "o-")
    plt.xlabel("depth noise (mm)"); plt.ylabel("RMSE (mm)"); plt.grid(alpha=0.3)
    plt.title("Localisation error vs depth noise")
    plt.tight_layout(); plt.savefig(f"{out}/error_vs_noise.png", dpi=150); plt.close()
 
    plt.figure(figsize=(7, 5))
    plt.scatter(us, vs, s=18, label="detected")
    plt.scatter(u_pred, v_pred, s=18, marker="x", label="predicted by calibrated K")
    plt.xlim(0, 640); plt.ylim(480, 0); plt.gca().set_aspect("equal")
    plt.xlabel("u (px)"); plt.ylabel("v (px)"); plt.legend(fontsize=8)
    plt.title(f"Calibration: reprojection RMSE {rmse(reproj):.2f} px")
    plt.tight_layout(); plt.savefig(f"{out}/calibration_reprojection.png", dpi=150); plt.close()
 
    # Example frame: RGB, mask, depth
    set_up(model, data, 0, False)
    rgb, depth = cam.capture(data)
    _, pixel = detect_ball_camera(rgb, depth, K_cal)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].imshow(rgb); ax[0].set_title("Wrist RGB")
    ax[1].imshow(target_mask(rgb), cmap="gray"); ax[1].set_title("Ball mask")
    ax[2].imshow(depth, vmax=1); ax[2].set_title("Wrist depth (m)")
    for a in ax:
        a.scatter(*pixel, c="yellow", marker="x", s=120)
        a.axis("off")
    plt.tight_layout(); plt.savefig(f"{out}/example_detection.png", dpi=150); plt.close()
 
    cam.renderer.close()
    print(f"\nSaved to {out}/")
 
 
if __name__ == "__main__":
    main()
 