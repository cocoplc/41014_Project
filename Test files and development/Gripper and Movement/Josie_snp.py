import time
import numpy as np
from pathlib import Path
import mujoco
import mujoco.viewer

# Load Chloe's environment
FOLDER = Path(__file__).resolve().parent

model = mujoco.MjModel.from_xml_path(str(FOLDER / "scene.xml"))
data = mujoco.MjData(model)

# Display all joints in the environment
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)

    print(
        f"Joint {i}: {name}, "
        f"qpos index: {model.jnt_qposadr[i]}, "
        f"DOF index: {model.jnt_dofadr[i]}"
    )

# Display all actuators
for i in range(model.nu):
    print(
        f"Actuator {i}: "
        f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)}"
    )

#----------------------- Identify Robot Joints -----------------------

# Assumes joints match up to the 
UR3e_joints = np.arange(6)

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

# ----------------------- Gripper Actuator -----------------------

# Actuator 6 controls the gripper fingers
Gripper_actuator = 6

# Gripper control settings
gripper_open = 0
gripper_closed = 255

# Start with the gripper open
gripper_command = gripper_open

# Find the allowed gripper control range
gripper_limits = model.actuator_ctrlrange[Gripper_actuator]

print("Gripper control range:", gripper_limits)

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
Cart_pos_con = 0.5 # # Gain converting position error into desired velocity (1/s)
dt = model.opt.timestep # Physics simulation time step (s)
max_speed = 0.3 # Maximum commanded joint velocity (rad/s)
tolerance = 0.005 # Acceptable end-effector position error (m)

q_desired = q_start.copy()

# ----------------------- Find ball target position -----------------------

ball_position = data.body("target_ball_body").xpos.copy()

# Aim 15 cm above the ball body's origin
# target_position = ball_position + np.array([0, 0, 0.25])

# Day 3: Pick-place targets 
# Initial position above the ball
approach_position = ball_position + np.array([0, 0, 0.25])

# Lower position for grasping (temporary estimate)
descend_position = ball_position + np.array([0, 0, 0.15])

# Rise with the ball
rise_position = ball_position + np.array([0, 0, 0.25])

# Start with the approach target
target_position = approach_position.copy()

# Current movement stage
stage = "APPROACH"

print("Approach position:", approach_position)
print("Descend position:", descend_position)

print("Ball position:", ball_position)
print("Target position:", target_position)

# Find end effector site
site_id = model.site("attachment_site").id

# ----------------------- Desired Gripper Orientation -----------------------

# Read the starting orientation of the gripper
R_desired = data.site("attachment_site").xmat.reshape(3, 3).copy()

# Orientation controller gain
K_orientation = 0.5

# ----------------------- Start Simulation -----------------------

last_print = data.time

# Open ONE viewer with the controller running
with mujoco.viewer.launch_passive(model, data) as viewer:

    viewer.sync()
    time.sleep(2)

    reached = False

    while viewer.is_running():

        # Read current joint positions and velocities
        q = data.qpos[Find_joint_pos].copy()
        qdot = data.qvel[Find_joint_vel].copy()

        #---------- Cartesian Controller ----------
        # Read the end-effector position
        ee_position = data.site("attachment_site").xpos.copy()

        # Calculate the position error
        position_error = target_position - ee_position

        # Calculate distance to the ball
        distance = np.linalg.norm(position_error)

        # Display progress approximately every 100 simulation steps
        if data.time - last_print >= 100 * dt:
            print(f"Distance to target: {distance:.4f} m")
            last_print = data.time

        if stage in ("APPROACH", "DESCEND", "RISE") and distance > tolerance:

            reached = False

            # Desired end-effector velocity
            v_desired = Cart_pos_con * position_error

            # Read current end-effector orientation
            R_current = data.site("attachment_site").xmat.reshape(3, 3).copy()

            # Calculate orientation error in world coordinates
            R_error = R_desired @ R_current.T

            # Convert rotation error into a rotation vector
            rotation_error = np.zeros(3)
            mujoco.mju_mat2Quat(quat_error := np.zeros(4), R_error.flatten())

            # Shortest rotation represented as an axis-angle vector
            if quat_error[0] < 0:
                quat_error = -quat_error

            angle = 2 * np.arctan2(np.linalg.norm(quat_error[1:]), quat_error[0])

            axis_length = np.linalg.norm(quat_error[1:])

            if axis_length > 1e-8:
                rotation_error = angle * quat_error[1:] / axis_length

            # Desired angular velocity
            omega_desired = K_orientation * rotation_error

            # Combine linear and angular velocities
            velocity_desired = np.concatenate((v_desired, omega_desired))

            # Create Jacobian matrices
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))

            # Calculate Jacobian at current pose
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

            # Combine position and rotation Jacobians
            J_robot = np.vstack((
                jacp[:, Find_joint_vel],
                jacr[:, Find_joint_vel]
            ))

            # Calculate desired joint velocities using Damped Least Squares
            damping = 0.05

            J_damped = J_robot.T @ np.linalg.inv(
                J_robot @ J_robot.T + damping**2 * np.eye(6)
            )

            qdot_desired = J_damped @ velocity_desired

            # Scale all joint velocities together to preserve their ratios
            peak_speed = np.max(np.abs(qdot_desired))

            if peak_speed > max_speed:
                qdot_desired *= max_speed / peak_speed

            # Convert joint velocities into joint position targets
            q_desired += qdot_desired * dt

        elif stage in ("APPROACH", "DESCEND", "RISE") and not reached:

            print("Target reached!")
            reached = True

            # Hold the reached joint position
            q_desired = q.copy()

            if stage == "APPROACH":

                print("Approach complete! Beginning descent.")

                stage = "DESCEND"
                target_position = descend_position.copy()

            elif stage == "DESCEND":

                print("Descent complete!")

                stage = "READY_TO_GRASP"

            if stage == "READY_TO_GRASP":

                print("gripping")
                stage = "RISE"
                target_position = rise_position.copy()

            elif stage == "RISE":
                print("object in gripper, begin rising.")

                stage = "READY_TO_MOVE"

        #---------- Joint Torque Controller ----------
        # Calculate joint position error
        error = q_desired - q

        # Proportional and derivative torque
        tau = Kp * error - Kd * qdot

        # Add gravity and bias compensation
        mujoco.mj_forward(model, data)
        tau += data.qfrc_bias[Find_joint_vel]

        # Apply torque limits if enabled
        for i, actuator_id in enumerate(Joint_actuator):
            if model.actuator_ctrllimited[actuator_id]:
                low, high = model.actuator_ctrlrange[actuator_id]
                tau[i] = np.clip(tau[i], low, high)

        # Apply torque to the robot
        data.ctrl[Joint_actuator] = tau

        # ----------------------- Gripper Control -----------------------

        # Keep gripper open during approach and descent
        if stage in ("APPROACH", "DESCEND"):
            gripper_command = gripper_open

        # Close gripper once descent is complete
        elif stage in ("READY_TO_GRASP", "RISE", "READY_TO_MOVE"):
            gripper_command = gripper_closed

        # elif stage == "READY_TO_MOVE":
        #     gripper_command = gripper_closed

        # Send command to gripper actuator
        data.ctrl[Gripper_actuator] = gripper_command

        # Advance simulation
        mujoco.mj_step(model, data)

        # Update viewer
        viewer.sync()

        # Approximately real-time simulation
        time.sleep(dt)
