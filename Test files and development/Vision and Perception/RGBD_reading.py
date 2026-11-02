"""
evaluate_vision.py - Day 4 localisation evidence (Hannah)
Commit: test: quantify localisation accuracy

Produces (in results/vision/):
  * trials.csv             every trial: true pos, estimate, error, pixel count
  * summary.txt            RMSE / mean / max, success rate, calibration evidence
  * error_hist.png         error distribution (raw vs offset-corrected)
  * error_vs_noise.png     RMSE as depth noise increases
  * error_xyz.png          per-axis error (shows any systematic bias)
  * example_detection.png  one RGB + depth frame with the detection marked

Uses vision.py from Day 3. Set CUBE_BODY / CAMERA_NAME there.
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

from scene.xml import (WristCamera, camera_pose_world, intrinsics_from_model,
                    detect_cube_camera, red_mask, CUBE_BODY)

OUT = "results/vision"
XML = "scene.xml"
SEEDS = [0, 1, 2, 3, 4]          # matches the reference's five seeds
TRIALS_PER_SEED = 10
WRIST_JITTER = 0.2               # rad, wrist joints 3-5
CUBE_XY_RANGE = 0.06             # m, +/- around the cube's default position
NOISE_LEVELS = [0.0, 0.002, 0.005, 0.01, 0.02]   # depth noise std (m)


# ---------------------------------------------------------------- helpers
def load_K(model):
    """Calibrated K from calibrate.py if saved, else the ideal model K."""
    K_model = intrinsics_from_model(model)
    if os.path.exists("calibration/K.npy"):
        return np.load("calibration/K.npy"), K_model, True
    print("calibration/K.npy not found - using ideal K from fovy")
    return K_model, K_model, False


def cube_handles(model):
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY)
    jnt = model.body_jntadr[body]
    qadr = model.jnt_qposadr[jnt] if jnt >= 0 else None   # free joint -> 7 qpos
    half = model.geom_size[model.body_geomadr[body]][2]    # box half-height
    return body, qadr, half


def place_cube(data, qadr, base_xyz, rng):
    if qadr is None:
        return
    data.qpos[qadr:qadr + 3] = base_xyz + np.array(
        [*rng.uniform(-CUBE_XY_RANGE, CUBE_XY_RANGE, 2), 0.0])
    data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]


def localise(rgb, depth, K, T_WC, noise_std, rng):
    if noise_std > 0:
        depth = depth + rng.normal(0, noise_std, depth.shape)
    P_C, pixel = detect_cube_camera(rgb, depth, K)
    if P_C is None:
        return None, None
    return (T_WC * P_C).flatten(), pixel


# ---------------------------------------------------------------- trials
def run_trials(model, data, cam, K, noise_std=0.0):
    body, qadr, half = cube_handles(model)
    mujoco.mj_forward(model, data)
    q_home = model.key_qpos[0][:6].copy() if model.nkey else data.qpos[:6].copy()
    base_xyz = data.qpos[qadr:qadr + 3].copy() if qadr is not None else None

    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for t in range(TRIALS_PER_SEED):
            mujoco.mj_resetData(model, data)
            if model.nkey:
                mujoco.mj_resetDataKeyframe(model, data, 0)
            place_cube(data, qadr, base_xyz, rng)
            data.qpos[:6] = q_home
            data.qpos[3:6] += rng.uniform(-WRIST_JITTER, WRIST_JITTER, 3)
            mujoco.mj_forward(model, data)

            rgb, depth = cam.capture(data)
            T_WC = camera_pose_world(model, data)
            P_W, pixel = localise(rgb, depth, K, T_WC, noise_std, rng)
            true = data.xpos[body].copy()

            row = dict(seed=seed, trial=t, noise=noise_std,
                       true_x=true[0], true_y=true[1], true_z=true[2],
                       detected=P_W is not None,
                       n_pixels=int(red_mask(rgb).sum()))
            if P_W is not None:
                # Visible-top centroid is ~one half-height above the true centre
                P_corr = P_W - np.array([0, 0, half])
                row.update(est_x=P_W[0], est_y=P_W[1], est_z=P_W[2],
                           err_raw=np.linalg.norm(P_W - true),
                           err_corr=np.linalg.norm(P_corr - true),
                           ex=P_corr[0] - true[0], ey=P_corr[1] - true[1],
                           ez=P_corr[2] - true[2])
            rows.append(row)
    return rows


def rmse(e):
    e = np.asarray(e)
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan")


# ---------------------------------------------------------------- calibration evidence
def calibration_evidence(K_used, K_model, calibrated):
    lines = ["Camera calibration evidence",
             f"  source: {'calibrate.py (calibration/K.npy)' if calibrated else 'ideal model K'}",
             f"  K used:\n{np.array2string(K_used, precision=2)}",
             f"  K from model fovy:\n{np.array2string(K_model, precision=2)}"]
    if calibrated:
        d = K_used - K_model
        lines += [f"  fx error: {d[0,0]:+.2f} px ({100*d[0,0]/K_model[0,0]:+.2f}%)",
                  f"  fy error: {d[1,1]:+.2f} px ({100*d[1,1]/K_model[1,1]:+.2f}%)",
                  f"  principal point error: ({d[0,2]:+.2f}, {d[1,2]:+.2f}) px"]
    if os.path.exists("calibration/reprojection_error.txt"):
        lines.append("  reprojection error: " +
                     open("calibration/reprojection_error.txt").read().strip())
    return "\n".join(lines)


# ---------------------------------------------------------------- plots
def save_example(cam, model, data, K):
    mujoco.mj_forward(model, data)
    rgb, depth = cam.capture(data)
    P_C, pixel = detect_cube_camera(rgb, depth, K)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].imshow(rgb); ax[0].set_title("Wrist RGB")
    ax[1].imshow(red_mask(rgb), cmap="gray"); ax[1].set_title("Red mask")
    im = ax[2].imshow(depth, cmap="viridis"); ax[2].set_title("Wrist depth (m)")
    plt.colorbar(im, ax=ax[2])
    if pixel:
        for a in ax:
            a.scatter(*pixel, c="yellow", marker="x", s=120)
    for a in ax:
        a.axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/example_detection.png", dpi=150); plt.close()


def main():
    os.makedirs(OUT, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    cam = WristCamera(model)
    K, K_model, calibrated = load_K(model)

    # 1. Main accuracy run (no added noise)
    rows = run_trials(model, data, cam, K, 0.0)
    ok = [r for r in rows if r["detected"]]
    raw = [r["err_raw"] for r in ok]
    corr = [r["err_corr"] for r in ok]

    # 2. Noise sweep
    noise_rmse = []
    all_rows = list(rows)
    for s in NOISE_LEVELS[1:]:
        nr = run_trials(model, data, cam, K, s)
        all_rows += nr
        noise_rmse.append(rmse([r["err_corr"] for r in nr if r["detected"]]))
    noise_rmse = [rmse(corr)] + noise_rmse

    # CSV
    keys = sorted({k for r in all_rows for k in r})
    with open(f"{OUT}/trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(all_rows)

    # Summary
    bias = np.mean([[r["ex"], r["ey"], r["ez"]] for r in ok], axis=0) if ok else [np.nan]*3
    summary = "\n".join([
        "3D localisation accuracy (no added noise)",
        f"  trials: {len(rows)}  ({len(SEEDS)} seeds x {TRIALS_PER_SEED})",
        f"  detection rate: {len(ok)}/{len(rows)}",
        f"  RMSE raw (surface centroid):  {1000*rmse(raw):.2f} mm",
        f"  RMSE offset-corrected:        {1000*rmse(corr):.2f} mm",
        f"  mean / max corrected error:   {1000*np.mean(corr):.2f} / {1000*np.max(corr):.2f} mm" if corr else "",
        f"  mean bias (x, y, z):          {np.round(1000*np.array(bias), 2)} mm",
        "",
        "Depth-noise sweep (corrected RMSE)",
        *[f"  sigma={1000*s:5.1f} mm -> RMSE {1000*e:.2f} mm" for s, e in zip(NOISE_LEVELS, noise_rmse)],
        "",
        calibration_evidence(K, K_model, calibrated),
    ])
    open(f"{OUT}/summary.txt", "w").write(summary)
    print(summary)

    # Plots
    plt.figure(figsize=(7, 4))
    plt.hist(1000*np.array(raw), bins=20, alpha=0.6, label="raw (surface)")
    plt.hist(1000*np.array(corr), bins=20, alpha=0.6, label="offset-corrected")
    plt.xlabel("3D position error (mm)"); plt.ylabel("trials"); plt.legend()
    plt.title("Localisation error distribution")
    plt.tight_layout(); plt.savefig(f"{OUT}/error_hist.png", dpi=150); plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(1000*np.array(NOISE_LEVELS), 1000*np.array(noise_rmse), "o-")
    plt.xlabel("Depth noise std (mm)"); plt.ylabel("RMSE (mm)")
    plt.title("Localisation robustness to depth noise"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/error_vs_noise.png", dpi=150); plt.close()

    plt.figure(figsize=(7, 4))
    plt.boxplot([[1000*r[k] for r in ok] for k in ("ex", "ey", "ez")],
                labels=["x", "y", "z"])
    plt.axhline(0, color="k", lw=0.8)
    plt.ylabel("error (mm)"); plt.title("Per-axis error (world frame)")
    plt.tight_layout(); plt.savefig(f"{OUT}/error_xyz.png", dpi=150); plt.close()

    save_example(cam, model, data, K)
    print(f"\nSaved evidence to {OUT}/")


if __name__ == "__main__":
    main()