
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.linalg import solve_discrete_are


# ============================================================
# 1. LOAD THE ROBOT
# ============================================================

folder = Path(__file__).resolve().parent
model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
data = mujoco.MjData(model)

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

# Find the actual array locations using names rather
# than assuming the joints will always be in order.
q_indices = np.array([
    model.joint(name).qposadr[0] for name in joint_names
])

v_indices = np.array([
    model.joint(name).dofadr[0] for name in joint_names
])

motor_indices = np.array([
    model.actuator(name).id for name in motor_names
])

encoder_indices = np.array([
    model.sensor("encoder_" + name).adr[0]
    for name in joint_names
])

# Confirm that our motor mapping matches the joints.
for i, motor_id in enumerate(motor_indices):
    joint_id = model.joint(joint_names[i]).id
    if model.actuator_trnid[motor_id, 0] != joint_id:
        raise ValueError(f"Motor mapping mismatch: {motor_names[i]}")

    if not np.isclose(model.actuator_gear[motor_id, 0], 1.0):
        raise ValueError("This test expects unit actuator gear ratios.")


# ============================================================
# 2. SET THE STARTING POSITION
# ============================================================

q_start = np.array([0, -1.57, 1.57, -1.57, -1.57, 0])

# Reset the entire simulation, then set the arm pose.
mujoco.mj_resetData(model, data)
data.qpos[q_indices] = q_start
data.qvel[v_indices] = 0.0
mujoco.mj_forward(model, data)

# Hold the other five joints where they started,
# but rotate shoulder pan by 30 degrees.
q_target = q_start.copy()
q_target[0] += np.deg2rad(30.0)


# ============================================================
# 3. BUILD THE LQR CONTROLLER
# ============================================================

h = 0.01

physics_dt = model.opt.timestep
steps_per_control = round(h / physics_dt)

if not np.isclose(steps_per_control * physics_dt, h):
    raise ValueError("Control timestep must match physics steps.")

A = np.array([
    [1, h],
    [0, 1]
])

B = np.array([
    [0.5 * h**2],
    [h]
])

C = np.array([[1.0, 0.0]])

Q = np.diag([100.0, 10.0])
R = np.array([[1.0]])

riccati = solve_discrete_are(A, B, Q, R)

K = np.linalg.solve(
    R + B.T @ riccati @ B,
    B.T @ riccati @ A
)

print("LQR gain:", K)


# ============================================================
# 4. SET UP SIX KALMAN FILTERS
# ============================================================

encoder_std = 0.01
acceleration_std = 0.5

R_n = np.array([[encoder_std**2]])
Q_d = acceleration_std**2 * (B @ B.T)
Q_d += 1e-10 * np.eye(2)

# Each joint has its own angle and velocity estimate.
estimated_states = np.zeros((6, 2))

# Start the angle estimates at the encoder readings.
estimated_states[:, 0] = data.sensordata[encoder_indices]

# Each joint also has its own uncertainty matrix.
P = np.repeat(
    np.diag([0.1**2, 0.5**2])[None, :, :],
    6,
    axis=0
)

# We will use the previous acceleration commands
# to predict the next states.
previous_accelerations = np.zeros(6)

rng = np.random.default_rng(42)

# Add simulated encoder noise so we can test
# the filter under the same conditions as Day 4.
ADD_ENCODER_NOISE = True


# ============================================================
# 5. PREPARE THE TEST
# ============================================================

duration = 5.0
control_steps = round(duration / h)

# Full inertia matrix, including the gripper and object.
M_full = np.zeros((model.nv, model.nv))

log = []

print("\nStarting MuJoCo LQG test...")
print("Shoulder pan target: 30 degrees relative to start")


# ============================================================
# 6. RUN THE CONTROLLER
# ============================================================

with mujoco.viewer.launch_passive(model, data) as viewer:

    wall_start = time.perf_counter()

    for step in range(control_steps):

        if not viewer.is_running():
            break

        # Update MuJoCo's dynamics and encoder values.
        mujoco.mj_forward(model, data)

        measurements = data.sensordata[encoder_indices].copy()

        if ADD_ENCODER_NOISE:
            measurements += rng.normal(
                0.0, encoder_std, size=6
            )

        # ----------------------------------------------------
        # A. UPDATE EACH JOINT'S KALMAN FILTER
        # ----------------------------------------------------

        for j in range(6):

            x_previous = estimated_states[j].reshape(2, 1)

            # Predict angle and velocity.
            x_pred = (
                A @ x_previous
                + B * previous_accelerations[j]
            )

            # Predict uncertainty.
            P_pred = A @ P[j] @ A.T + Q_d

            # Compare encoder angle with prediction.
            innovation = (
                measurements[j] - (C @ x_pred).item()
            )

            S = C @ P_pred @ C.T + R_n

            # Calculate Kalman gain.
            L = np.linalg.solve(
                S,
                C @ P_pred
            ).T

            # Correct our state estimate.
            x_corrected = x_pred + L * innovation

            estimated_states[j] = x_corrected.flatten()

            I = np.eye(2)

            P[j] = (
                (I - L @ C) @ P_pred @ (I - L @ C).T
                + L @ R_n @ L.T
            )

        # ----------------------------------------------------
        # B. CALCULATE LQR ACCELERATION COMMANDS
        # ----------------------------------------------------

        desired_accelerations = np.zeros(6)

        for j in range(6):

            target_state = np.array([
                q_target[j],
                0.0
            ])

            error = estimated_states[j] - target_state

            desired_accelerations[j] = -(
                K @ error
            ).item()

        # ----------------------------------------------------
        # C. CONVERT ACCELERATION INTO MOTOR TORQUE
        # ----------------------------------------------------

        # Convert MuJoCo's compressed inertia matrix into a full matrix.
        # The matrix describes how the robot's joints influence
        # one another when they accelerate.
        mujoco.mj_fullM(model, data, M_full)

        # Take the inertia terms corresponding to the arm.
        M_arm = M_full[np.ix_(v_indices, v_indices)]

        # Include the robot's gravity and velocity-dependent
        # bias forces.
        bias_arm = data.qfrc_bias[v_indices].copy()

        # Computed torque:
        # torque = inertia * desired acceleration + bias
        tau = M_arm @ desired_accelerations + bias_arm

        # Respect the actuator limits in our UR3e model.
        for j, motor_id in enumerate(motor_indices):

            if model.actuator_ctrllimited[motor_id]:
                low, high = model.actuator_ctrlrange[motor_id]
                tau[j] = np.clip(tau[j], low, high)

        # Apply the calculated torque to the six arm motors.
        data.ctrl[motor_indices] = tau

        # Keep the gripper open.
        data.ctrl[
            model.actuator("gripper_fingers_actuator").id
        ] = 0.0

        # ----------------------------------------------------
        # D. ADVANCE THE PHYSICS SIMULATION
        # ----------------------------------------------------

        for _ in range(steps_per_control):
            mujoco.mj_step(model, data)

        previous_accelerations = desired_accelerations.copy()

        # Record the true angle for evaluating the controller.
        true_angle = data.qpos[q_indices[0]]
        estimated_angle = estimated_states[0, 0]

        log.append([
            data.time,
            true_angle,
            estimated_angle,
            tau[0]
        ])

        if (step + 1) % 100 == 0:

            angle_deg = np.rad2deg(true_angle)
            error_deg = np.rad2deg(q_target[0] - true_angle)

            print(
                f"Time: {data.time:.1f} s | "
                f"Shoulder: {angle_deg:.2f} deg | "
                f"Error: {error_deg:.2f} deg | "
                f"Torque: {tau[0]:.2f} Nm"
            )

        viewer.sync()

        # Keep the viewer approximately in real time.
        remaining = data.time - (
            time.perf_counter() - wall_start
        )

        if remaining > 0:
            time.sleep(remaining)


# ============================================================
# 7. REPORT THE FINAL RESULTS
# ============================================================

final_angles = data.qpos[q_indices].copy()
errors = np.rad2deg(q_target - final_angles)

print("\nFinal joint errors (degrees):")
for name, error in zip(joint_names, errors):
    print(f"{name}: {error:.3f}")

print("\nMuJoCo LQG integration test complete.")
