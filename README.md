# Robot Arm Manipulator
MuJoCo simulation for a mini robot arm tasked with inserting an end-effector box into a moving/vibrating box mould. This repository contains the MuJoCo model (MJCF), mesh assets, and Python control code used for the CS403 Box Control competition.

## Repository structure

- `RunMiniArmBox.py` — Entry point. Launches the MuJoCo viewer and runs the control loop.
- `YourControlCode.py` — Your controller implementation (IK + PD) called every simulation step.
- `BoxControlHandler.py` — Utility / environment helper (box pose estimation, goal checking, difficulty).
- `Robot/miniArmBox.xml` — MuJoCo MJCF model for the robot + box mould + sensors.
- `Robot/meshes/` — STL/OBJ assets referenced by the MJCF.
- `Project_Paper.pdf` — Project write-up / report.

## Requirements

- Python 3.9+ (recommended)
- MuJoCo (Python bindings)
- GLFW
- NumPy
- `absl-py` (used when running `RunMiniArmBox.py` as a script)

A minimal install usually looks like:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install mujoco glfw numpy absl-py
```

## Run the simulation

From the repository root:

```bash
python RunMiniArmBox.py
```

This will open the MuJoCo GUI and load `Robot/miniArmBox.xml`.

### What happens when you run it?

- The MuJoCo model is loaded from `Robot/miniArmBox.xml`.
- The controller class `YourCtrl` (in `YourControlCode.py`) is instantiated.
- Each simulation step, `YourCtrl.update()` computes control torques for the 6 arm joints.
- The box mould body (`box_mould`) is driven with a sinusoidal vibration (difficulty-dependent).
- The task ends when the end-effector box contacts the `detection_plate` (goal reached) or after ~30 simulated seconds.

## Customize the controller

Edit `YourControlCode.py` (class `YourCtrl`). The main method is:

- `YourCtrl.update()` — called every simulation tick; should write into `d.ctrl[0:6]` to control the robot joints.

Some useful helpers are available via `BoxControlHandle` (instantiated as `self.boxCtrlhdl`), including:

- End-effector pose access (`_get_ee_position()`, `_get_ee_orientation()`)
- Box pose estimation (`box_orientation`, `box_midpoint`)
- Goal condition (`check_goal_reached()`)
- Difficulty parameters (`set_difficulty()`, `get_diff_params()`)

## Notes

- The MJCF uses `meshdir="meshes/"` so meshes must remain in `Robot/meshes/` relative to `Robot/miniArmBox.xml`.
- The controller currently logs box sensor data to `sensor_log.csv` when running.

## Acknowledgements

Developed for **CS403** at **UMass Amherst** (Box Control competition).
