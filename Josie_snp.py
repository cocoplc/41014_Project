import time
import numpy as np
from pathlib import Path
import mujoco
import mujoco.viewer

# Load Chloe's environment
FOLDER = Path(__file__).resolve().parent

model = mujoco.MjModel.from_xml_path(str(FOLDER / "scene.xml"))
data = mujoco.MjData(model)

# Set the UR3e starting pose
q_start = np.array([0, -1.57, 1.57, -1.57, -1.57, 0])

data.qpos[0:6] = q_start
data.qvel[0:6] = 0

# Update the robot's kinematics
mujoco.mj_forward(model, data)

# Temporary joint controller - Will need to be replaced by the propper one
Kp = 100
Kd = 20

q_desired = q_start.copy()

# Open ONE viewer with the controller running
with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        # Read current joint positions and velocities
        q = data.qpos[:6].copy()
        qdot = data.qvel[:6].copy()

        # Calculate position error and damping
        error = q_desired - q
        tau = Kp * error - Kd * qdot

        # Add gravity compensation
        mujoco.mj_forward(model, data)
        tau += data.qfrc_bias[:6]

        # Apply actuator torque limits
        limits = model.actuator_ctrlrange[:6]
        tau = np.clip(tau, limits[:, 0], limits[:, 1])

        # Send torque commands
        data.ctrl[:6] = tau

        # Advance the physics simulation
        mujoco.mj_step(model, data)

        # Update viewer
        viewer.sync()

        # Run approximately in real time
        time.sleep(model.opt.timestep)


