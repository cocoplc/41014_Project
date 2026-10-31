import os
import mujoco
import numpy as np
import matplotlib.pyplot as plt
from scene_setup import reset_scene, get_wrist_images  # reuse your setup functions

os.makedirs("evidence", exist_ok=True)                  # folder for evidence files (made if it doesn't exist)

model = mujoco.MjModel.from_xml_path("scene.xml")       # load scene
data = mujoco.MjData(model)                             # simulation state
renderer = mujoco.Renderer(model, 480, 640)             # off-screen renderer

lines = []                                              # text lines to save
for seed in range(5):                                   # seeds 0 to 4
    a = reset_scene(model, data, seed)                  # place ball with this seed
    b = reset_scene(model, data, seed)                  # same seed again - should give the same position
    rgb, depth = get_wrist_images(model, data, renderer)                # what the wrist camera sees
    plt.imsave(f"evidence/seed{seed}_rgb.png", rgb)                     # save colour image
    plt.imsave(f"evidence/seed{seed}_depth.png", depth, vmax=1)         # save depth image (capped at 1 m)

    line = f"seed {seed}: ball at {a.round(3)}  repeatable: {np.allclose(a, b)}"
    print(line)
    lines.append(line)

with open("evidence/scene_repeatability.txt", "w") as f:     # save the results as a text file
    f.write("\n".join(lines))