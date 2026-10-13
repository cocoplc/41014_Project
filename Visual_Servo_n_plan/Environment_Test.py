print("Visual Seroving and Plannning test by josie")

# Step 1:  Install Mujoco and Numpy
#   " python -m pip install mujoco numpy " in terminal

# Step 2: Create simulation environement 
import mujoco
import mujoco.viewer 

# Define the simulation environment (MuJoCo XML)
# Describes the objects, dimensions and positions 

scene = """
<mujoco model="Pick and Place">

    <option gravity="0 0 -9.81"/>

    <worldbody>

        <!-- Lighting -->
        <light pos="0 0 3"/>

        <!-- Floor -->
        <geom name="floor"
              type="plane"
              size="3 3 0.1"
              rgba="0.8 0.8 0.8 1"/>

        <!-- Table -->
        <body name="table" pos="0.5 0 0.05">
            <geom type="box"
                  size="0.3 0.3 0.05"
                  rgba="0.6 0.4 0.2 1"/>
        </body>

        <!-- Cube on the table -->
        <body name="cube" pos="0.5 0 0.75">
            <freejoint/>
            <geom type="box"
                  size="0.05 0.05 0.05"
                  mass="0.1"
                  rgba="0.1 0.4 1 1"/>
        </body>

    </worldbody>

</mujoco>
"""

# Step 3: Load the Mujoco environment

# The XML description into simulation 
# The model stores the physical properties and scene structure
model = mujoco.MjModel.from_xml_string(scene)

# Stores the simulation data for changing values
# (positions and velocities of objects)
data = mujoco.MjData(model)

# Step 4: Display 

# Open MuJoCo viewer
mujoco.viewer.launch(model, data)