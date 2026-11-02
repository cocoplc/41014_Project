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
Load the scene with the look pose and a random ball position:
```
python scene_setup.py
```

Run the full pick (vision → plan → grasp):
```
python Josie_snp.py
```

Evaluate vision localisation for camera calibration every time the conditons change (results saved to `results/` and `evidence/`):
```
python evaluate_vision.py
```

Kalman filter joint-state estimation test:
```
python kalman_filter_test.py
```

## Contributors
- Chloe Sargiss – MuJoCo scene, robot and gripper setup
- Hannah Murphy – RGB-D vision and localisation
- Angus Tedeschi – State-space control and observer
- Josephine Thu – Visual servoing and planning
