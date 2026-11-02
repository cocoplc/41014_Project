
import numpy as np
import matplotlib.pyplot as plt

# 
h = 0.01

A = np.array([
    [1, h],
    [0, 1]
])

B = np.array([
    [0.5 * h**2],
    [h]
])

C = np.array([[1, 0]])


encoder_std = 0.01

R_n = np.array([[encoder_std**2]])

acceleration_std = 0.5


Q_d = acceleration_std**2 * (B @ B.T)
Q_d += 1e-10 * np.eye(2)

true_state = np.array([
    [0.0],
    [0.0]
])

estimated_state = np.array([
    [0.0],
    [0.0]
])


P = np.diag([0.1**2, 0.5**2])

duration = 5.0
steps = round(duration / h)

rng = np.random.default_rng(42)

times = []
true_angles = []
measured_angles = []
estimated_angles = []

true_velocities = []
estimated_velocities = []

for step in range(steps):

    current_time = (step + 1) * h

    if current_time < 1.0:
        u = 1.0

    elif current_time < 2.0:
        u = -1.0

    else:
        u = 0.0

    actual_acceleration = u + rng.normal(0.0, acceleration_std)

    true_state = (A @ true_state + B * actual_acceleration)

    measurement = ((C @ true_state).item() + rng.normal(0.0, encoder_std))

    predicted_state = A @ estimated_state + B * u

    predicted_P = A @ P @ A.T + Q_d

    innovation = measurement - (C @ predicted_state).item()

    S = C @ predicted_P @ C.T + R_n

    L = np.linalg.solve(S, C @ predicted_P).T

    estimated_state = (predicted_state + L * innovation)

    I = np.eye(2)


    P = ((I - L @ C) @ predicted_P @ (I - L @ C).T + L @ R_n @ L.T)

    times.append(current_time)

    true_angles.append(true_state[0, 0])
    measured_angles.append(measurement)
    estimated_angles.append(estimated_state[0, 0])

    true_velocities.append(true_state[1, 0])
    estimated_velocities.append(estimated_state[1, 0])


    if (step + 1) % 100 == 0:
        print(
            f"Time: {current_time:.1f} s | "
            f"True angle: {true_state[0,0]:.3f} rad | "
            f"Estimated angle: {estimated_state[0,0]:.3f} rad | "
            f"Estimated velocity: {estimated_state[1,0]:.3f} rad/s"
        )

plt.figure()

plt.plot(times, measured_angles, label="Noisy encoder", alpha=0.35)

plt.plot(times, true_angles, label="True angle")

plt.plot(times, estimated_angles, label="Kalman estimate", linestyle="--")

plt.xlabel("Time (s)")
plt.ylabel("Joint angle (rad)")
plt.title("Kalman Filter - Angle Estimation")
plt.legend()
plt.grid(True)

plt.figure()

plt.plot(times, true_velocities, label="True velocity")

plt.plot(times, estimated_velocities, label="Estimated velocity", linestyle="--")

plt.xlabel("Time (s)")
plt.ylabel("Joint velocity (rad/s)")
plt.title("Kalman Filter - Velocity Estimation")
plt.legend()
plt.grid(True)

plt.show()

angle_error = (np.array(estimated_angles) - np.array(true_angles))

velocity_error = (np.array(estimated_velocities) - np.array(true_velocities))

angle_rmse = np.sqrt(np.mean(angle_error**2))
velocity_rmse = np.sqrt(np.mean(velocity_error**2))

print("\nAngle RMSE:", angle_rmse, "rad")
print("Velocity RMSE:", velocity_rmse, "rad/s")

print("\nKalman filter test complete.")
