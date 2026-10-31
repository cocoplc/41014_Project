import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
 
from scene_setup import reset_scene, LOOK_POSE
from vision import (WristCamera, camera_pose_world, intrinsics_from_model,
                    detect_ball_camera, target_mask, TARGET_BODY, BALL_RADIUS)
 
OUT = "evidence/vision"
TEST_SEEDS = range(0, 20)          # localisation test positions
CALIB_SEEDS = range(100, 140)      # calibration positions: never used for testing
WRIST_JITTER = 0.15                # rad on wrist joints for the "wrist moved" condition
NOISE_LEVELS = [0.0, 0.001, 0.002, 0.005, 0.01]   # depth noise std (m)
 
 
# ---------------------------------------------------------------- scene helpers
def setup_trial(model, data, seed, jitter=0.0):
    """Chloe's reset: look pose + random ball. Optionally move the wrist too."""
    reset_scene(model, data, seed)
    if jitter > 0:
        rng = np.random.default_rng(1000 + seed)
        data.qpos[:6] = LOOK_POSE
        data.qpos[3:6] += rng.uniform(-jitter, jitter, 3)
        mujoco.mj_forward(model, data)
    return data.body(TARGET_BODY).xpos.copy()        # ground-truth centre (world)
 
 
def localise(rgb, depth, K, T_WC, noise_std=0.0, rng=None):
    if noise_std > 0:
        depth = depth + rng.normal(0.0, noise_std, depth.shape)
    P_C, pixel = detect_ball_camera(rgb, depth, K)
    if P_C is None:
        return None, None, None
    return (T_WC * P_C).flatten(), P_C, pixel
 
 
def rmse(errors):
    e = np.asarray(errors, float)
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan")
 
 
# ---------------------------------------------------------------- 1. calibration
def calibrate(model, data, cam):
    """Fit u = fx*X/Z + cx and v = fy*Y/Z + cy by least squares, where (X, Y, Z) is the
    TRUE ball centre in the camera frame and (u, v) the detected ball pixel centroid."""
    rows = []
    for seed in CALIB_SEEDS:
        for jitter in (0.0, WRIST_JITTER):            # wrist moves spread the ball across the image
            true_W = setup_trial(model, data, seed, jitter)
            rgb, _ = cam.capture(data)
            v, u = np.nonzero(target_mask(rgb))
            if len(u) < 30:
                continue
            X, Y, Z = (camera_pose_world(model, data).inv() * true_W).flatten()
            rows.append((X / Z, Y / Z, u.mean(), v.mean()))
    a = np.array(rows)
    (fx, cx), *_ = np.linalg.lstsq(np.c_[a[:, 0], np.ones(len(a))], a[:, 2], rcond=None)
    (fy, cy), *_ = np.linalg.lstsq(np.c_[a[:, 1], np.ones(len(a))], a[:, 3], rcond=None)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    reproj = np.hypot(fx * a[:, 0] + cx - a[:, 2], fy * a[:, 1] + cy - a[:, 3])
    return K, reproj, a
 
 
# ---------------------------------------------------------------- 2/3. trials
def run_trials(model, data, cam, K, noise_std=0.0):
    rows = []
    rng = np.random.default_rng(42)
    for condition, jitter in (("look_pose", 0.0), ("wrist_moved", WRIST_JITTER)):
        for seed in TEST_SEEDS:
            true_W = setup_trial(model, data, seed, jitter)
            rgb, depth = cam.capture(data)
            T_WC = camera_pose_world(model, data)
            P_W, P_C, pixel = localise(rgb, depth, K, T_WC, noise_std, rng)
            row = dict(condition=condition, seed=seed, noise_mm=1000 * noise_std,
                       true_x=true_W[0], true_y=true_W[1], true_z=true_W[2],
                       n_pixels=int(target_mask(rgb).sum()), detected=P_W is not None)
            if P_W is not None:
                # Same estimate WITHOUT the sphere-centre correction, to show what it buys
                ray = P_C / np.linalg.norm(P_C)
                P_surf_W = (T_WC * (P_C - (2 / 3) * BALL_RADIUS * ray)).flatten()
                row.update(est_x=P_W[0], est_y=P_W[1], est_z=P_W[2],
                           u=pixel[0], v=pixel[1],
                           err=np.linalg.norm(P_W - true_W),
                           err_surface=np.linalg.norm(P_surf_W - true_W),
                           ex=P_W[0] - true_W[0], ey=P_W[1] - true_W[1], ez=P_W[2] - true_W[2])
            rows.append(row)
    return rows
 
 
# ---------------------------------------------------------------- plots
def save_example(model, data, cam, K):
    true_W = setup_trial(model, data, 0)
    rgb, depth = cam.capture(data)
    P_C, pixel = detect_ball_camera(rgb, depth, K)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].imshow(rgb); ax[0].set_title("Wrist RGB")
    ax[1].imshow(target_mask(rgb), cmap="gray"); ax[1].set_title("Ball mask")
    im = ax[2].imshow(depth, vmax=1); ax[2].set_title("Wrist depth (m)")
    plt.colorbar(im, ax=ax[2], fraction=0.035)
    for a in ax:
        if pixel:
            a.scatter(*pixel, c="yellow", marker="x", s=120)
        a.axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/example_detection.png", dpi=150); plt.close()
 
 
def main():
    os.makedirs(OUT, exist_ok=True)
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
    cam = WristCamera(model)
    K_ideal = intrinsics_from_model(model)
 
    # 1. Calibration ------------------------------------------------------
    K_cal, reproj, calib_pts = calibrate(model, data, cam)
    np.save(f"{OUT}/K_calibrated.npy", K_cal)
    d = K_cal - K_ideal
 
    # 2. Accuracy with the ideal and the calibrated K (no noise) -----------
    results = {}
    for name, K in (("ideal K", K_ideal), ("calibrated K", K_cal)):
        rows = run_trials(model, data, cam, K)
        ok = [r for r in rows if r["detected"]]
        results[name] = (rows, ok)
    rows, ok = results["calibrated K"]
 
    # 3. Depth-noise sweep (calibrated K) ----------------------------------
    sweep, all_rows = [], list(rows)
    for s in NOISE_LEVELS:
        nr = rows if s == 0 else run_trials(model, data, cam, K_cal, s)
        if s > 0:
            all_rows += nr
        sweep.append(rmse([r["err"] for r in nr if r["detected"]]))
 
    # CSV ------------------------------------------------------------------
    keys = list(dict.fromkeys(k for r in all_rows for k in r))
    with open(f"{OUT}/trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(all_rows)
 
    # Summary ----------------------------------------------------------------
    def block(name):
        rws, okk = results[name]
        out = [f"{name}:  detected {len(okk)}/{len(rws)}"]
        for cond in ("look_pose", "wrist_moved"):
            e = [r["err"] for r in okk if r["condition"] == cond]
            if e:
                out.append(f"    {cond:12s} RMSE {1000*rmse(e):6.2f} mm   mean {1000*np.mean(e):6.2f}   max {1000*np.max(e):6.2f} mm")
        return out
 
    bias = 1000 * np.mean([[r["ex"], r["ey"], r["ez"]] for r in ok], axis=0)
    summary = "\n".join([
        "HANNAH - DAY 4 VISION EVIDENCE",
        "",
        "1. Camera calibration (fit from known ball centres)",
        f"   points used: {len(calib_pts)} ({len(CALIB_SEEDS)} seeds x 2 wrist poses; separate from test seeds)",
        f"   ideal K (fovy 75):      fx {K_ideal[0,0]:.2f}  fy {K_ideal[1,1]:.2f}  cx {K_ideal[0,2]:.2f}  cy {K_ideal[1,2]:.2f}",
        f"   calibrated K:           fx {K_cal[0,0]:.2f}  fy {K_cal[1,1]:.2f}  cx {K_cal[0,2]:.2f}  cy {K_cal[1,2]:.2f}",
        f"   difference:             fx {d[0,0]:+.2f} ({100*d[0,0]/K_ideal[0,0]:+.2f}%)  fy {d[1,1]:+.2f} ({100*d[1,1]/K_ideal[1,1]:+.2f}%)"
        f"  cx {d[0,2]:+.2f}  cy {d[1,2]:+.2f} px",
        f"   reprojection error:     RMSE {rmse(reproj):.2f} px   max {reproj.max():.2f} px",
        "",
        f"2. 3D localisation accuracy ({len(TEST_SEEDS)} random ball positions per condition)",
        *["   " + line for line in block("ideal K")],
        *["   " + line for line in block("calibrated K")],
        f"   without sphere-centre correction (calibrated K): RMSE {1000*rmse([r['err_surface'] for r in ok]):.2f} mm",
        f"   mean bias (x, y, z): {bias.round(2)} mm",
        "",
        "3. Robustness to depth noise (calibrated K, both conditions)",
        *[f"   sigma {1000*s:5.1f} mm  ->  RMSE {1000*e:6.2f} mm" for s, e in zip(NOISE_LEVELS, sweep)],
        "",
        "Notes: ground truth is the ball body position from MuJoCo. Error is the straight-line",
        "distance between the estimated and true ball centre.",
    ])
    with open(f"{OUT}/summary.txt", "w") as f:
        f.write(summary)
    print(summary)
 
    # Plots ------------------------------------------------------------------
    err_lp = [1000 * r["err"] for r in ok if r["condition"] == "look_pose"]
    err_wm = [1000 * r["err"] for r in ok if r["condition"] == "wrist_moved"]
    plt.figure(figsize=(7, 4))
    plt.hist([err_lp, err_wm], bins=12, label=["look pose", "wrist moved"])
    plt.xlabel("3D position error (mm)"); plt.ylabel("trials"); plt.legend()
    plt.title("Ball localisation error (calibrated K)")
    plt.tight_layout(); plt.savefig(f"{OUT}/error_hist.png", dpi=150); plt.close()
 
    plt.figure(figsize=(7, 4))
    plt.boxplot([[1000 * r[k] for r in ok] for k in ("ex", "ey", "ez")])
    plt.xticks([1, 2, 3], ["x", "y", "z"])
    plt.axhline(0, color="k", lw=0.8)
    plt.ylabel("error (mm)"); plt.title("Per-axis error, world frame")
    plt.tight_layout(); plt.savefig(f"{OUT}/error_xyz.png", dpi=150); plt.close()
 
    plt.figure(figsize=(7, 4))
    plt.plot(1000 * np.array(NOISE_LEVELS), 1000 * np.array(sweep), "o-")
    plt.xlabel("depth noise std (mm)"); plt.ylabel("RMSE (mm)"); plt.grid(alpha=0.3)
    plt.title("Localisation robustness to depth noise")
    plt.tight_layout(); plt.savefig(f"{OUT}/error_vs_noise.png", dpi=150); plt.close()
 
    plt.figure(figsize=(7, 4))
    a = calib_pts
    plt.scatter(a[:, 2], a[:, 3], s=18, label="detected")
    plt.scatter(K_cal[0, 0] * a[:, 0] + K_cal[0, 2], K_cal[1, 1] * a[:, 1] + K_cal[1, 2],
                s=18, marker="x", label="reprojected (calibrated K)")
    plt.xlim(0, 640); plt.ylim(480, 0); plt.gca().set_aspect("equal")
    plt.xlabel("u (px)"); plt.ylabel("v (px)"); plt.legend(fontsize=8)
    plt.title(f"Calibration: reprojection RMSE {rmse(reproj):.2f} px")
    plt.tight_layout(); plt.savefig(f"{OUT}/calibration_reprojection.png", dpi=150); plt.close()
 
    plt.figure(figsize=(6, 5))
    tx = [r["true_x"] for r in ok if r["condition"] == "look_pose"]
    ty = [r["true_y"] for r in ok if r["condition"] == "look_pose"]
    plt.scatter(tx, ty, c=err_lp, cmap="viridis", s=60)
    plt.colorbar(label="error (mm)")
    plt.xlabel("x (m)"); plt.ylabel("y (m)"); plt.gca().set_aspect("equal")
    plt.title("Error by ball position on the table (look pose)")
    plt.tight_layout(); plt.savefig(f"{OUT}/error_by_position.png", dpi=150); plt.close()
 
    save_example(model, data, cam, K_cal)
    print(f"\nSaved evidence to {OUT}/")
    cam.renderer.close()          # close before Python exits (avoids shutdown warnings)
 
 
if __name__ == "__main__":
    main()
 