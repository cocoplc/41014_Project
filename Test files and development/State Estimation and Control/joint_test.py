"""
I'm making this file to test how the UR3 moves when a small torque is applied to one of the joints.
If I'm going to make a state-space controller, I need to design it so that the controller can calculate
the amount of torque each motor needs to apply to move the robot to the desired pose.
This is just a basic test because it doesn't use the measured angle to adjust the torque. I'll do that later.
"""

# This lets me save my measurements to a CSV file to analysis
import csv            
# This lets me run the simulation at roughly real life speed when viewing it           
import time            
# There's stuff in different folders so this just helps to direct everything to the right place          
from pathlib import Path       
# Importing mujoco simulator and the viewer
import mujoco                    
import mujoco.viewer             

# When set to false, the viewer window doesn't open (faster)
run_viewer = True               

# Opens the folder containing this file
folder = Path(__file__).resolve().parent
# Reads the file that contains the robnot and the environment
scene = mujoco.MjSpec.from_file(str(folder / "scene.xml"))
# Reads the file that contains the gripper
gripper = mujoco.MjSpec.from_file(str(folder / "robot_models/robotiq/2f85.xml"))
# Mounts the gripper to the robot's attachment point
    # gripper/ lets us use gripper object names without conflicts later on
scene.attach(gripper, site="attachment_site", prefix="gripper/")
# builds the mujoco model
model = scene.compile()    

# For every joint except the one I want to move:
for name in ["shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
             "wrist_2_joint", "wrist_3_joint"]:
    # Find where the joint's angle is stored in the q position array
    index = int(model.joint(name).qposadr[0])  
    # Find what angle the joint should be at when in the "home position" (as defined in the model)
    home_angle = float(model.key("home").qpos[index])
    # This line holds the joint at its home angle (this is a constraint, not a motor command [that's important])
    scene.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=name, data=[home_angle] + [0] * 10)

# Rebuilds the model with the joint constraints applied
model = scene.compile()
# Creates a data object that contains the current state
data = mujoco.MjData(model)      

"""
Useful:
data.qpos = joint positions
data.qvel = joint velocities
data.ctrl = motor commands
data.time = current simulation time
"""

# Resets the robot to its home position to start the test from a consistent starting point
mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
# Calculates the initial joint positions, forces, and sensor readings
mujoco.mj_forward(model, data)   
# Finding the index of the motor that controls the shoulder pan joint (i.e., the base joint)
motor = model.actuator("shoulder_pan").id
# Finding where the encoder measurement for this joint is stored
    # The encoder is what measures the angular position of the joint (I need this info for building the controller)
encoder = int(model.sensor("encoder_shoulder_pan_joint").adr[0])

# Ensuring there is initially no viewer
viewer = None
# Opening the mujoco viewer to visualise the movement of the robot (if set to true earlier)
if run_viewer:
    viewer = mujoco.viewer.launch_passive(model, data)
# Saving the output of this test to a results folder in a file called "joint_test"
output = folder / "results/joint_test.csv"

# Open the csv file and allow writing (and close the file when this block finishes)
with output.open("w", newline="") as file:
    log = csv.writer(file)
    # Writing the column headings to record the time, torque, and joint angle
    log.writerow(["time_s", "torque_Nm", "encoder_angle_rad"])
    # Record the current time
    start = time.perf_counter()
    # When to print the next measurement
    next_print = 0.0

    # Running the simulation for 3 seconds
    for step in range(round(3.0 / model.opt.timestep)):
        # Automatically stops the test if I close the window
        if viewer is not None and not viewer.is_running():
            break

        # The amount of torque to apply
        torque = 0.0
        # Apply 0.5 N/m of torque from 0.5 to 1 second
        if 0.5 <= data.time < 1.0:
            torque = 0.5       
        # Apply -0.5 N/m of torque from 1 to 1.5 seconds  
        elif 1.0 <= data.time < 1.5:
            torque = -0.5        # Negative torque slows it down.
        # Send the torque command to the motor (input)
        data.ctrl[motor] = torque

        # Read the joint angle from the encoder (output)
        angle = float(data.sensordata[encoder])
        # Write one row of data to the results file
        log.writerow([data.time, torque, angle])
        # Print it if the next print time has been reached (every 0.5 seconds)
        if data.time >= next_print:
            print(f"{data.time:.2f} s | {torque:+.1f} Nm | {angle:+.4f} rad")
            next_print += 0.5

        # Advance the simulation by one timestep
        mujoco.mj_step(model, data)  
        # Update the viewer if it's enabled
        if viewer is not None:
            # This updates the robot's displayed position
            viewer.sync()
            # This compares the simulation time with real time so I can watch it at normal speed if it's running fast
            time.sleep(max(0.0, data.time - (time.perf_counter() - start)))

# Closes the viewer if it was open
if viewer is not None:
    viewer.close()