# 03dog velocity tasks

The Python package is named `dog03`; assets remain under `data/Robots/03dog`.
Run `python3 scripts/tools/prepare_03dog_urdf.py` after replacing the CAD export.
The script preserves mesh filenames, inertias, joint frames, axes and the
30 Nm / 30 rad/s motor limits. It normalizes link/joint names, makes mesh paths
relative, and fixes the exported left rear foot joint to match the other feet.
The robot has twelve driven joints and four fixed, separately reported feet.
Isaac Lab imports the URDF automatically; no CSDog1 USD is needed.

| Original leg | Training prefix | Driven links/joints | Fixed foot |
| --- | --- | --- | --- |
| `left_front` | `FL` | `hip`, `thigh`, `calf` / `<leg>_<part>_joint` | `FL_FOOT` |
| `right_front` | `FR` | `hip`, `thigh`, `calf` / `<leg>_<part>_joint` | `FR_FOOT` |
| `left_back` | `RL` | `hip`, `thigh`, `calf` / `<leg>_<part>_joint` | `RL_FOOT` |
| `right_back` | `RR` | `hip`, `thigh`, `calf` / `<leg>_<part>_joint` | `RR_FOOT` |

Actions and joint observations follow FL, FR, RL, RR, each hip/thigh/calf.
`robot_lab/assets/dog03.py` uses the default angles from
`quadruped-robot-gc/policy/03dog/base.yaml`: hip/calf = 0 rad and thigh =
-0.7854 rad on all four legs. All hip axes are local +X; the right front/rear
thigh axes are local +Y,
matching the deployment URDF and MJCF. Forward kinematics puts foot origins
at approximately z=-0.30635 m and y=+/-0.186 m relative to the base; the
lowest foot mesh point is z=-0.3215 m. Initial base height remains 0.40 m
(about 7.85 cm clearance), with a 0.32 m base-height reward target and a
-0.26 m body-relative swing-foot target. Nominal PD gains are 30/1, torque
limits are 30 Nm and action scale is 0.25. These are starting training
parameters and need validation in simulation and tuning for the real robot.

All five tasks use `ManagerBasedRLEnv`, inheriting CSDog1's common manager
configurations, phase observations, rewards and PPO/HIM hyperparameters.
The 47-D actor frame is phase (2), angular velocity (3), projected gravity (3),
commands (3), joint offsets (12), joint velocities (12), previous actions (12).
The 282-D actor uses six frames. The default HIM wrapper assembles these frames
from oldest to latest, matching the shared export code. The critic is 50-D
throughout these tasks, which allows flat-282 weights to continue into domain
randomization and stairs. CSDog1's custom Cyberstar environment, 491-D critic,
USD foot reconstruction and automatic checkpoint paths are specific to that
robot and are not part of the 03dog tasks.

The model adapter sets URDF sensor paths, uses actual `*_FOOT` bodies for all
foot rewards, adapts stance width/height targets, and terminates on base contact.
Raw joint-angle mirror loss is disabled because the CAD joint zeros differ.
Initial flat tasks retain material/push variation; the robustness and stairs
tasks also enable mass, COM and PD-gain variation.

Task IDs and complete train/resume/play commands are in the **03dog 训练任务**
section of the root `AT_DESCRIPE.md`. All logs use independent `dog03_*`
directories. Continuation uses explicit 03dog checkpoint paths and the same
training backend throughout the sequence.

`DOG03_STAIR_STAGE=0...4` selects both stair height bands and the runner's default
additional iterations; the default is stage 0. Stages are changed between runs.
The robot shares CSDog1's terrain geometry implementation: 30% flat patches and
70% stairs, with 65% ascent / 35% descent among stair patches. Each reset starts
on the center platform with a random heading. Stages and budgets are configured
locally in `stairs_config.json`, independently of `CSDOG1_STAIR_STAGE`.

CPU checks:

```bash
python3 -m unittest discover -s tests -p 'test_dog03*.py'
```
