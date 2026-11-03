#41014_Project_Group_13 – UR3 Vision-Guided Pick (MuJoCo)

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

## Use of generative AI
AI was frequently utilised to assist in the interpretation and application of relevant theory. It was also used to assist with generating, adapting, reviewing, and debugging code, developing testing procedures, and interpreting simulation results. For example, the code used to prepare graphs from the obtained data was predominantly AI generated.

With this being said, the resulting system required substantial practical development, debugging, testing, and refinement by the team. The system objectives, design choices, and implementation decisions are our own.

We acknowledge that generative AI made a significant contribution to the development process, including portions of the final codebase. However, the submitted work represents the team's application, integration, evaluation and understanding of these tools and techniques, rather than the uncritical submission of AI-generated material.

## Contributors
- Chloe Sargiss – MuJoCo scene, robot and gripper setup
- Hannah Murphy – RGB-D vision and localisation
- Angus Tedeschi – State-space control and observer
- Josephine Thu – Visual servoing and planning
