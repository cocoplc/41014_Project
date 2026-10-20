import time
import numpy as np
from pathlib import Path
import mujoco
import mujoco.viewer

# Load Chloe's environment
FOLDER = Path(__file__).resolve().parent

model = mujoco.MjModel.from_xml_path(str(FOLDER / "scene.xml"))
data = mujoco.MjData(model)

#----------------------- Identify Robot Joints -----------------------

# Assumes joints match up to the 
UR3e_joints = np.arrange(6)

# Finding Joint Positions 
Find_joint_pos = np.array([model.jnt_qposadr[i] for i in UR3e_joints])
#                          ^ finds where joint i stores its position in data.qpos
#                                                   ^ repeat for all six joints 
#                ^ convert to numpy array 

# Finding Joint Velocities 
Find_joint_vel = np.array([model.jnt_dofadr[i] for i in UR3e_joints])
#                          ^ find degrees of freedom (DOFs) associated with the joints

# This assumes the first six actuators are torque actuators controlling these six joints.
Joint_actuator = np.arange(6)

#----------------------- Set the UR3e starting pose -----------------------

q_start = np.array([0, -1.57, 1.57, -1.57, -1.57, 0])

data.qpos[Find_joint_pos] = q_start
data.qvel[Find_joint_vel] = 0

# Update the robot's kinematics
mujoco.mj_forward(model, data)

# ----------------------- Controller Settings -----------------------

# Joint controller - Will need to be replaced by the propper one
Kp = 100
Kd = 20

# Cartesian Position Controller - NEW
Cart_pos_con = 0.5 # Controls how strongly it stays to end effector 
dt = model.opt.timstep # simulation time intival for calculations 
max_speed = 0.3 # Limits commanded cartesian movement speed 
tolerance = 0.02 # defines acceptable position error

q_desired = q_start.copy()

# ----------------------- Find ball target position -----------------------

ball_position = data.body("ball").xpos.copy()

# Aim 15 cm above the ball body's origin
target_position = ball_position + np.array([0, 0, 0.15])

print("Ball position:", ball_position)
print("Target position:", target_position)

# Find end effector site
site_id = model.site("attachment_site").id

# ----------------------- Start Simulation -----------------------

# Open ONE viewer with the controller running
with mujoco.viewer.launch_passive(model, data) as viewer:

    viewer.sync()
    time.sleep(2)

    reached = False

    while viewer.is_running():

        # Read current joint positions and velocities
        q = data.qpos[Find_joint_pos].copy()
        qdot = data.qvel[Find_joint_vel].copy()

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

# Now we add the movement to the object

# Find end effector position 
ee_position = data.site("attachment_site").xpos # Get the end-effector position in world coordinates
print("End-effector position (X, Y, Z):", ee_position) # Display the X, Y and Z coordinates

# Motion planning 
# We now know where the end effector is
# Lets see how far it is from the ball (off set by z = 1)

# Get the cube's position in world coordinates
ball_position = data.body("ball").xpos.copy()

# Set a target position 10 cm above the cube's top
# ball half-height is 0.05 m
target_position = ball_position + np.array([0, 0, 0.15])

# Calculate the difference between target and end effector
position_error = target_position - ee_position

# Calculate the straight-line distance to the target
distance = np.linalg.norm(position_error)

# End effector Jacobian 
# Create an empty matrix for the position Jacobian
jacp = np.zeros((3, model.nv))

# Create an empty matrix for the orientation Jacobian
jacr = np.zeros((3, model.nv))

# Find the attachment site at the robot's end effector
site_id = model.site("attachment_site").id

# Calculate the Jacobian at the current robot pose
mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

# Display the position Jacobian
print("Position Jacobian:")
print(jacp)

# Extract the UR3e's six joint columns
J_robot = jacp[:, 6:12]

print("Robot Jacobian:")
print(J_robot)

Kp = 0.5  # Proportional gain (1/s)

# Desired velocity towards the target (m/s)
v_desired = Kp * position_error

print("Desired end-effector velocity:", v_desired)

# Pseudoinverse handles the non-square 3x6 Jacobian
J_inverse = np.linalg.pinv(J_robot)

# Calculate the six joint velocities (rad/s)
qdot = J_inverse @ v_desired

print("Required joint velocities:", qdot)

import time

# Step 11: Move the robot towards the target

Kp = 0.5             # Position error gain
dt = model.opt.timestep
max_speed = 0.3      # Maximum joint speed in rad/s
tolerance = 0.02     # Stop within 2 cm of target

# Start with the current joint position targets
q_target = data.qpos[7:13].copy()

with mujoco.viewer.launch_passive(model, data) as viewer:

    # Show the initial robot pose before moving
    viewer.sync()
    time.sleep(2)

    while viewer.is_running():

        # Get current end-effector position
        ee_position = data.site("attachment_site").xpos.copy()

        # Calculate the remaining position error
        error = target_position - ee_position
        distance = np.linalg.norm(error)

        # Stop commanding motion when close enough
        if distance < tolerance:
            print("Target reached!")
            break

        # Calculate desired end-effector velocity
        v_desired = Kp * error

        # Recalculate Jacobian at current pose
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        mujoco.mj_jacSite(
            model, data, jacp, jacr,
            model.site("attachment_site").id
        )

        # Extract robot joint Jacobian
        J_robot = jacp[:, 6:12]

        # Calculate joint velocities
        qdot = np.linalg.pinv(J_robot) @ v_desired

        # Limit joint velocities for safer testing
        qdot = np.clip(qdot, -max_speed, max_speed)

        # Convert joint velocities into position targets
        q_target += qdot * dt

        # Send position targets to the actuators
        data.ctrl[:] = q_target

        # Advance simulation
        mujoco.mj_step(model, data)

        # Update viewer
        viewer.sync()

        # Run approximately in real time
        time.sleep(dt)

    # Keep viewer open to inspect the final position
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.02)
