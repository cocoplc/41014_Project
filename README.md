# A4_SC – UR3 Vision-Guided Pick (MuJoCo)

41014 Sensors and Control for Mechatronic Systems – Assessment 4 Integrated Project (Track B).
A UR3 robot in MuJoCo localises a ball using a wrist-mounted RGB-D camera and picks it up with a gripper.

## Resources
- [MuJoCo documentation](https://mujoco.readthedocs.io/)
- [MuJoCo Python bindings](https://mujoco.readthedocs.io/en/stable/python.html)
- 41014 tutorial notebooks (state-space control, Kalman filtering, visual servoing)

## Install info
Requires Python 3.10+.

```
git clone https://github.com/cocoplc/A4_SC.git
cd A4_SC
pip install mujoco numpy opencv-python matplotlib
```

## Run commands
Run the entire Simulation from: 
```
SIMULATION.py
```
This file imports the camera calibration, the scene environment, the Kalman filter, the full Track B system movement, robustness testing, disturbance testing and generation of graphs.

## Changing the tests
At the top of SIMULATION.py, change these booleans to choose what runs:
```
ROBUSTNESS_TEST = False   # True runs 10 ball positions x 3 conditions (baseline, noisy, disturbed)
                          # False runs a single pick
DEMO_DISTURBANCE = True   # True applies a 0.5 N m torque pulse during the single pick
SHOW_VIEWER = True        # True opens the MuJoCo window during the single pick
WRIST_VIEW = False        # True shows the wrist camera view instead of the outside view
SEED = 0                  # Ball position used for the single pick
```

## Where results are saved
```
results/pick_evidence/demo_disturbed/seed_0/    # single pick
    trial_data.csv
    trial_analysis.png

results/pick_evidence/                          # robustness test
    robustness_results.csv
    robustness_summary.txt
    comparison_errors.png
    ball_positions.png

evidence/vision/                                # camera calibration graphs
    calibration_reprojection.png                # (only saved when Camera_calibration.py is run on its own)
    example_detection.png                   
```

## Contributors
- Chloe Sargiss – MuJoCo scene, robot and gripper setup
- Hannah Murphy – RGB-D vision and localisation
- Angus Tedeschi – State-space control and observer
- Josephine Thu – Visual servoing and planning
