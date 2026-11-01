import time
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
from scipy.linalg import solve_discrete_are
from scene_setup import reset_scene
from vision import (WristCamera, detect_ball_camera, R_MJ_TO_OPT)

# Intermediate milestone: detect the ball and move to the approach position.
DT = 0.01
MAX_TIME = 35.0
CAMERA_EVERY = 25
APPROACH_HEIGHT = 0.12
MAX_SPEED = 0.25
MAX_LAG = np.deg2rad(3)
FOLDER = Path(__file__).resolve().parent


def run_trial(seed=0, encoder_std=0.005, show_viewer=True):
    model = mujoco.MjModel.from_xml_path(str(FOLDER / "scene.xml"))
    data = mujoco.MjData(model)
    reset_scene(model, data, seed)

    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint"
    ]

    motor_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3"
    ]

    # Find where MuJoCo stores the six joint positions.
    q_id = np.array([model.joint(name).qposadr[0] for name in joint_names])

    # Find where MuJoCo stores the six joint velocities.
    v_id = np.array([model.joint(name).dofadr[0] for name in joint_names])

    # Map motor names onto the actuator indices used for torque commands.
    motor_id = np.array([model.actuator(name).id for name in motor_names])

    # Map each encoder name onto its sensor-data index.
    encoder_id = np.array([model.sensor("encoder_" + name).adr[0] for name in joint_names])

    gripper_id = model.actuator("gripper_fingers_actuator").id
    site_id = model.site("attachment_site").id
    camera_id = model.camera("wrist_camera").id
    ball_id = model.body("target_ball_body").id

    # Calculate the number of physics updates per controller step.
    physics_steps = round(DT / model.opt.timestep)

    if not np.isclose(
        physics_steps * model.opt.timestep,
        DT
    ):
        raise ValueError(
            "Controller timestep must match physics timestep"
        )

    """STATE-SPACE MODEL AND LQR"""

    # State = [joint angle, joint angular velocity].
    # Input = joint angular acceleration.
    # Measurement = encoder joint angle.

    A = np.array([
        [1, DT],
        [0, 1]
    ])

    B = np.array([
        [0.5 * DT**2],
        [DT]
    ])

    C = np.array([[1, 0]])

    # Test controllability and observability.
    controllability = np.hstack((B,A @ B))

    observability = np.vstack((C, C @ A))

    print("Controllability rank:", np.linalg.matrix_rank(controllability))

    print("Observability rank:", np.linalg.matrix_rank(observability))

    # LQR weighting matrices.
    Q = np.diag([100.0, 10.0])    # Penalise angle error and velocity error
    R = np.array([[1.0]])         # Penalise large acceleration commands

    # Solve the discrete Riccati equation for the LQR feedback law.
    P_lqr = solve_discrete_are(A, B, Q, R)

    # Calculate the two feedback gains: angle error and velocity error.
    K = np.linalg.solve(R + B.T @ P_lqr @ B, B.T @ P_lqr @ A).ravel()

    print("LQR gain:", K)

    """Kalman Filter"""
    # encoder_std comes from the selected experiment.
    # The Kalman measurement covariance changes with it.
    acceleration_std = 0.5

    # Encoder measurement covariance is variance = standard deviation squared.
    R_noise = np.array([[encoder_std**2]])

    # Process covariance describes uncertainty in the acceleration model.
    Q_noise = (acceleration_std**2 * (B @ B.T) + 1e-10 * np.eye(2))

    # One estimated angle and angular velocity for each of the six joints.
    estimate = np.zeros((6, 2))

    # Initialise the angle estimates from encoders; velocity starts at zero.
    estimate[:, 0] = data.sensordata[encoder_id]

    # Covariance matrices for all six Kalman filters.
    P = np.repeat(np.diag([0.1**2, 0.5**2])[None], 6, axis=0)

    previous_acceleration = np.zeros(6)

    # Keep the random samples repeatable between trials.
    rng = np.random.default_rng(42)

    # Separate data object for computing kinematics and dynamics from estimates.
    estimated_data = mujoco.MjData(model)

    estimated_data.qpos[:] = model.qpos0
    estimated_data.qpos[q_id] = estimate[:, 0]

    mujoco.mj_forward(model, estimated_data)

    # Prepare the dense inertia matrix for computed torque control.
    mass_matrix = np.zeros((model.nv, model.nv))

    """Camera and Gripper"""

    calibration_file = (FOLDER / "evidence" / "vision" / "K_calibrated.npy")

    camera_matrix = np.load(calibration_file)

    camera = WristCamera(model)

    # Calculate the midpoint between the open gripper pads.
    site_R = estimated_data.site_xmat[site_id].reshape(3, 3).copy()
    site_position = estimated_data.site_xpos[site_id].copy()
    left_pad = estimated_data.body("gripper_left_pad").xpos
    right_pad = estimated_data.body("gripper_right_pad").xpos
    finger_midpoint = (left_pad + right_pad) / 2

    # Express the finger midpoint offset relative to the tool frame.
    finger_offset = (site_R.T @ (finger_midpoint - site_position))

    # Maintain the original gripper orientation.
    start_orientation = site_R.copy()

    stage = "SEARCH"
    stage_start = data.time
    last_seen = -100.0
    ball_estimate = None
    q_target = estimate[:, 0].copy()
    qdot_target = np.zeros(6)
    gripper_command = 0.0
    stop_reason = "Maximum simulation time"
    camera_errors = []
    ball_start = data.xpos[ball_id].copy()
    save_evidence = False          # Just test detection and approach in this milestone.
    viewer = mujoco.viewer.launch_passive(model, data) if show_viewer else None
    try:
        for step in range(int(MAX_TIME / DT)):
            old_stage = stage
            mujoco.mj_forward(model, data)
            # Read the encoders and add noise; never use true qpos for control.
            measurements = (data.sensordata[encoder_id].copy() + rng.normal(0, encoder_std, 6))

            for j in range(6):

                # Predict the angle and angular velocity.
                x_previous = estimate[j].reshape(2, 1)

                x_pred = (A @ x_previous + B * previous_acceleration[j])

                # Predict how uncertain the state estimate has become.
                P_pred = (A @ P[j] @ A.T + Q_noise)

                # Innovation is the difference between measured and predicted angle.
                innovation = (measurements[j] - (C @ x_pred).item())

                S = (C @ P_pred @ C.T + R_noise)

                # Calculate how much the measurement should correct the prediction.
                kalman_gain = np.linalg.solve(S, C @ P_pred).T

                # Correct the predicted angle and angular velocity.
                x_corrected = (x_pred + kalman_gain * innovation)

                estimate[j] = x_corrected.flatten()

                # Update uncertainty.
                I = np.eye(2)

                # Joseph covariance update keeps the uncertainty matrix stable.
                P[j] = (
                    (I - kalman_gain @ C)
                    @ P_pred
                    @ (I - kalman_gain @ C).T
                    + kalman_gain
                    @ R_noise
                    @ kalman_gain.T
                )

            # Calculate estimated robot kinematics.
            estimated_data.qpos[q_id] = estimate[:, 0]

            estimated_data.qvel[:] = 0

            estimated_data.qvel[v_id] = estimate[:, 1]

            mujoco.mj_forward(model, estimated_data)

            current_R = estimated_data.site_xmat[site_id].reshape(3, 3)

            # Estimated world position of the midpoint between the fingers.
            finger_position = (estimated_data.site_xpos[site_id] + current_R @ finger_offset)

            """RGB-D Camera"""

            if (stage in ("SEARCH", "APPROACH") and step % CAMERA_EVERY == 0):

                rgb, depth = camera.capture(data)

                point_camera, pixel = detect_ball_camera(rgb, depth, camera_matrix)

                if point_camera is not None:
                    # Camera-to-world transformation.
                    # Uses estimated camera pose.
                    R_wc = (estimated_data.cam_xmat[camera_id].reshape(3, 3) @ R_MJ_TO_OPT)

                    camera_position = (estimated_data.cam_xpos[camera_id])

                    # Transform the ball from optical camera coordinates into world position.
                    detected_ball = (camera_position + R_wc @ point_camera)

                    # Camera localisation error for EVALUATION.
                    # This value is never used for planning.
                    true_ball = data.xpos[ball_id].copy()

                    camera_errors.append(np.linalg.norm(detected_ball - true_ball) * 1000)

                    # Smooth the camera position measurement.
                    if ball_estimate is None:
                        ball_estimate = detected_ball
                    else:
                        ball_estimate = (0.65 * ball_estimate + 0.35 * detected_ball)

                    last_seen = data.time

                    if stage == "SEARCH":
                        stage = "APPROACH"

            """PBVS Trajectory Planning"""

            goal = None

            if stage == "APPROACH":
                goal = (ball_estimate + np.array([0, 0, APPROACH_HEIGHT]))

            if goal is not None:
                # Difference between desired and estimated finger position.
                position_error = (goal - finger_position)

                distance = np.linalg.norm(position_error)

                # Small-angle orientation error keeps the gripper pointing down.
                rotation_error = 0.5 * sum((np.cross(current_R[:, i], start_orientation[:, i]) for i in range(3)), np.zeros(3))

                # Desired Cartesian velocity.
                desired_velocity = np.concatenate((0.65 * position_error, 0.55 * rotation_error))

                # Jacobian at the finger midpoint.
                jacp = np.zeros((3, model.nv))
                jacr = np.zeros((3, model.nv))

                mujoco.mj_jac(model, estimated_data, jacp, jacr, finger_position, model.site_bodyid[site_id])

                J = np.vstack((jacp[:, v_id], jacr[:, v_id]))

                # Damped least-squares inverse Jacobian.
                damping = 0.06

                # Damped least squares converts Cartesian velocity to joint velocity.
                qdot_target = (J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), desired_velocity))

                # Limit joint velocities.
                peak_speed = np.max(np.abs(qdot_target))

                if peak_speed > MAX_SPEED:

                    qdot_target *= (MAX_SPEED / peak_speed)

                # Keep the desired position close to the estimated joint position.
                q_target = np.clip(q_target, estimate[:, 0] - MAX_LAG, estimate[:, 0] + MAX_LAG)

                # Integrate desired joint velocity to update the joint position reference.
                q_target += qdot_target * DT

                # Check whether the stage is complete.
                if (stage == "APPROACH" and distance < 0.008):
                    stage = "DONE"  # Intermediate milestone ends above the ball.

            else:
                # Hold current position when not moving.
                qdot_target[:] = 0

                q_target = np.clip(q_target, estimate[:, 0] - MAX_LAG, estimate[:, 0] + MAX_LAG)

            if stage != old_stage:
                stage_start = data.time
                print(f"{data.time:.2f}s: {old_stage} -> {stage}")

            """LGR and Calculated Torque"""

            desired_states = np.column_stack((q_target, qdot_target))

            error = estimate - desired_states

            # LQR turns the estimated state error into desired acceleration.
            acceleration = -(error @ K)

            # Robot inertia matrix.
            mujoco.mj_fullM(model, estimated_data, mass_matrix)

            arm_mass = mass_matrix[np.ix_(v_id, v_id)]

            # Computed torque compensates for robot inertia and bias forces.
            torque = (arm_mass @ acceleration + estimated_data.qfrc_bias[v_id])

            # Apply motor torque limits.
            clipped = False

            for j, motor in enumerate(motor_id):
                if model.actuator_ctrllimited[motor]:

                    low, high = (model.actuator_ctrlrange[motor])

                    original = torque[j]

                    torque[j] = np.clip(original, low, high)

                    if torque[j] != original:
                        clipped = True

            if clipped:
                saturated_steps += 1

            # Send the control inputs to the actual simulated arm and gripper.
            data.ctrl[motor_id] = torque
            data.ctrl[gripper_id] = gripper_command

            previous_acceleration = acceleration.copy()

            for _ in range(physics_steps):
                mujoco.mj_step(model, data)
            if viewer is not None and viewer.is_running():
                viewer.sync()
                time.sleep(DT)
            if stage == "DONE":
                stop_reason = "Approach target reached"
                break
            if stage == "SEARCH" and data.time > 4:
                stop_reason = "Camera could not find the ball"
                break
            if stage == "APPROACH" and data.time - last_seen > 2.0:
                stop_reason = "Camera lost the ball"
                break
            if stage == "APPROACH" and data.time - stage_start > 24:
                stop_reason = "Approach timed out"
                break
    finally:
        camera.renderer.close()
        if viewer is not None:
            viewer.close()
    print(f"Stage: {stage}, reason: {stop_reason}")
    return stage == "DONE"


if __name__ == "__main__":
    run_trial(seed=0, show_viewer=True)
