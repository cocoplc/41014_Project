"""
Making a model of a single joint and designing an LQR contoller to move it

This controller should calculate the angular acceleration needed to move the joint
    to the desired angle based on it's current angle and velocity
"""

import numpy as np
import matplotlib.pyplot as plt
# To solve the Riccati equation
from scipy.linalg import solve_discrete_are

# Controller timesteps (i.e., updates 100 times a second)
    # That makes this a discrete-time model
h = 0.01

# State transition matrix:
A = np.array([
    [1, h],
    [0, 1]
])

# Input matrix
B = np.array([
    [0.5 * h**2],
    [h]
])

# Output matrix
C = np.array([[1, 0]])

# LQR Weights

# State error pentalty matrix
Q = np.diag([100.0, 10.0])

# Control input penalty matrix
R = np.array([[1.0]])


# Solves the Riccati equation
P = solve_discrete_are(A, B, Q, R)

# Calculates the LQR feedback gain
K = np.linalg.solve(
    R + B.T @ P @ B,
    B.T @ P @ A
)

print("LQR gain K:")
print(K)

# Current state vector
x = np.array([
    [0.0],
    [0.0]
])

# Target state vector
target = np.array([
    [np.deg2rad(30.0)],
    [0.0]
])

# Run simulation for 5 seconds
duration = 5.0
steps = round(duration / h)

# Stores the info
times = []
angles = []
velocities = []
accelerations = []

# Running the controller (closed-loop)
for step in range(steps):

    current_time = step * h
    
    # Difference between current and target state
    error = x - target

    # Angular acceleration
    u = float((-K @ error).item())

    times.append(current_time)
    angles.append(np.rad2deg(x[0, 0]))
    velocities.append(x[1, 0])
    accelerations.append(u)

    # Updates the joint state using the state-space model
        # i.e., x(k+1) = A*x(k) + B*u(k)
    x = A @ x + B * u

    # Only prints the results every second (100 steps)
    if step % 100 == 0:
        print(
            f"Time: {current_time:.1f} s | "
            f"Angle: {np.rad2deg(x[0, 0]):.2f} deg | "
            f"Velocity: {x[1, 0]:.3f} rad/s | "
            f"Acceleration: {u:.3f} rad/s^2"
        )

# Creating a graph to plot the results
plt.figure()
plt.plot(times, angles, label="Actual angle")
plt.axhline(30.0, linestyle="--", label="Target angle (30 degrees)")
plt.xlabel("Time (s)")
plt.ylabel("Joint angle (degrees)")
plt.title("LQR Joint Position Control")
plt.legend()
plt.grid(True)
plt.show()

# Prints the final angle achieved (in degrees)
print("\nFinal angle:", np.rad2deg(x[0, 0]), "degrees")

# Prints the final angular velocity
print("Final velocity:", x[1, 0], "rad/s")


print("\nLQR simulation complete.")