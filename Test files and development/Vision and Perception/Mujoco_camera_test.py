import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------
# 1. Create MuJoCo environment
# ------------------------------------

xml = """
<mujoco model="RGBD Camera Test">

    <option gravity="0 0 -9.81"/>

    <worldbody>

        <!-- Light above the scene -->
        <light name="main_light"
               pos="0 0 2"
               dir="0 0 -1"
               directional="true"/>

        <!-- Floor -->
        <geom name="floor"
              type="plane"
              size="2 2 0.1"
              rgba="0.8 0.8 0.8 1"/>

        <!-- Red cube -->
        <body name="target" pos="0.2 0.1 0.05">
            <geom name="cube"
                  type="box"
                  size="0.05 0.05 0.05"
                  rgba="1 0 0 1"/>
        </body>

        <!-- Camera looking down -->
        <camera name="rgbd_camera"
                pos="0 0 1"
                quat="1 0 0 0"
                fovy="60"/>

    </worldbody>

</mujoco>
"""


# ------------------------------------
# 2. Load MuJoCo simulation
# ------------------------------------

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)


# ------------------------------------
# 3. Camera parameters
# ------------------------------------

width = 640
height = 480

# Calculate focal length from vertical FOV
fovy = np.deg2rad(60)

fy = height / (2 * np.tan(fovy / 2))
fx = fy

cx = width / 2
cy = height / 2

# Camera intrinsic matrix
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

print("Camera intrinsic matrix:")
print(K)


# ------------------------------------
# 4. Create renderer
# ------------------------------------

renderer = mujoco.Renderer(
    model,
    height=height,
    width=width
)


# ------------------------------------
# 5. Capture RGB image
# ------------------------------------

renderer.update_scene(data, camera="rgbd_camera")

rgb = renderer.render().copy()

print("\nRGB image shape:", rgb.shape)


# ------------------------------------
# 6. Capture depth image
# ------------------------------------

renderer.enable_depth_rendering()

renderer.update_scene(data, camera="rgbd_camera")

depth = renderer.render().copy()

renderer.disable_depth_rendering()

print("Depth image shape:", depth.shape)


# ------------------------------------
# 7. Detect red cube
# ------------------------------------

red_mask = (
    (rgb[:, :, 0] > 150) &
    (rgb[:, :, 1] < 100) &
    (rgb[:, :, 2] < 100)
)

# Only use pixels with valid depth
valid_mask = (
    red_mask &
    np.isfinite(depth) &
    (depth > 0)
)

if np.any(valid_mask):

    # Find detected cube pixel centre
    v_pixels, u_pixels = np.where(valid_mask)

    u = np.mean(u_pixels)
    v = np.mean(v_pixels)

    # Measure cube depth
    cube_depth = np.median(depth[valid_mask])

    print("\nCube pixel centre:", (u, v))
    print("Cube measured depth:", cube_depth, "m")

    # Expected depth to top surface
    expected_depth = 0.9

    depth_error = abs(cube_depth - expected_depth)

    print("Expected cube depth:", expected_depth, "m")
    print("Depth error:", depth_error, "m")


    # ------------------------------------
    # 8. Convert pixel + depth to 3D
    # ------------------------------------

    pixel = np.array([u, v, 1.0])

    P_C = cube_depth * (np.linalg.inv(K) @ pixel)

    print("\nPosition in camera frame:", P_C)


    # ------------------------------------
    # 9. Convert camera frame to world frame
    # ------------------------------------

    # Camera position in world coordinates
    camera_position = np.array([0.0, 0.0, 1.0])

    # Convert optical camera coordinates to world
    # Valid for this downward-facing camera
    P_world = np.array([
        P_C[0],
        -P_C[1],
        camera_position[2] - P_C[2]
    ])

    print("Detected world position:", P_world)


    # ------------------------------------
    # 10. Compare against known position
    # ------------------------------------

    # Cube centre is at (0.2, 0.1, 0.05)
    # Top surface is at z = 0.1
    expected_position = np.array([0.2, 0.1, 0.1])

    position_error = np.linalg.norm(
        P_world - expected_position
    )

    print("Expected world position:", expected_position)
    print("3D position error:", position_error, "m")

else:
    print("\nRed cube not detected!")


# ------------------------------------
# 11. Display RGB and depth images
# ------------------------------------

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# RGB image
ax[0].imshow(rgb)
ax[0].set_title("MuJoCo RGB Camera")
ax[0].set_xlabel("Pixel X")
ax[0].set_ylabel("Pixel Y")

# Mark detected cube centre
if np.any(valid_mask):
    ax[0].scatter(
        u, v,
        color="yellow",
        marker="x",
        s=100,
        label="Detected centre"
    )
    ax[0].legend()

# Depth image
im = ax[1].imshow(depth, cmap="viridis")
ax[1].set_title("MuJoCo Depth Camera")
ax[1].set_xlabel("Pixel X")
ax[1].set_ylabel("Pixel Y")

plt.colorbar(
    im,
    ax=ax[1],
    label="Depth (m)"
)

plt.tight_layout()
plt.show()


# ------------------------------------
# 12. Open MuJoCo viewer
# ------------------------------------

mujoco.viewer.launch(model, data)

renderer.close()

