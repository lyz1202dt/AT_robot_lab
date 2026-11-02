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

The registered tasks use standard and full Cyberstar training paths. All
implementations are local copies under `dog03`; none import or inherit from
`csdog1`. Shared Isaac Lab/RobotLab infrastructure and the HIM model remain in use.

| Task suffix | Environment | Actor | Critic | Reward implementation |
| --- | --- | --- | --- | --- |
| Flat / Rough | Standard ManagerBasedRLEnv | 47 | 50 | `rewards.py`, `rewards_cfg.py` |
| Flat-282 | `cyberstar.lab.Dog03CyberstarEnv` (domain randomization off) | 282 | 491 | `cyberstar/core.py`, `cyberstar/config.json` |
| Flat-282-DomainRand | `cyberstar.lab.Dog03CyberstarEnv` | 282 | 491 | `cyberstar/core.py`, `cyberstar/config.json` |
| Stairs-282 | `cyberstar.stairs_lab.Dog03CyberstarStairsEnv` | 282 | 491 | `cyberstar/core.py`, `cyberstar/stairs_core.py`, `cyberstar/stairs_config.json` |

The directory layout follows CSDog1, including the full `cyberstar/` package:

```text
dog03/
├── agents/                 # Local PPO, HIM and CUSRL configurations
├── flat_env_cfg.py         # Standard flat stages
├── rough_env_cfg.py
├── observations.py
├── rewards.py             # Standard-task formulas
├── rewards_cfg.py         # Standard-task weights and parameters
├── stairs_env_cfg.py       # Full Cyberstar stair environment configuration
└── cyberstar/
    ├── core.py            # Full robust-flat reward/command/history logic
    ├── stairs_core.py     # Stair-specific commands and rewards
    ├── runtime.py
    ├── stairs_runtime.py  # DOG03_STAIR_STAGE and iteration selection
    ├── lab.py             # Simulator adapter and Ji22 reward aggregation
    ├── stairs_lab.py
    ├── env_cfg.py
    ├── asset.py           # Spawn 03dog URDF with its existing FOOT bodies
    ├── config.json        # Robust-flat parameters and reward scales
    ├── stairs_config.json # Stair parameters, reward scales and stages
    ├── terrain.py
    ├── stairs_terrain.py
    ├── stairs_geometry.py
    ├── math.py
    ├── LICENSE
    └── provenance.json
```

For independent reward editing, change `_reward_*` methods in the local
`cyberstar/core.py` and `cyberstar/stairs_core.py`, and weights under
`env.rewards.scales` in the corresponding JSON. Zero weights disable terms.
Robust flat enables 78 reward terms plus termination; stairs enables 85 plus
termination. The Ji22 aggregator combines positive and negative rewards before
adding termination. Standard tasks continue to use local `rewards.py` and
`rewards_cfg.py`. The manager-only domain-randomization class in `flat_env_cfg.py`
is retained as in CSDog1; the registered DomainRand task uses `cyberstar/env_cfg.py`.

The actor frame is phase (2), angular velocity (3), projected gravity (3),
commands (3), joint offsets (12), joint velocities (12), previous actions (12).
HIM history contains six complete frames, oldest to latest. The full Cyberstar
stages also preserve the 491-D privileged critic, command scale `[2, 2, 0.25]`,
contact-dependent stopping, action delay, explicit PD and structured commands.
Flat-282 uses the same full Cyberstar environment, rewards, terrain and command
scale as DomainRand. Only its `domain_rand.randomize_*` flags and
`domain_rand.push_robots` are disabled; observation noise and reset/command
sampling remain enabled. Flat / Rough use command scale `[1, 1, 0.25]`.

### Domain randomization parity with ATDog

The 282-D DomainRand and stairs stages already include link/base mass, inertia,
COM, joint zero offsets, motor strength, PD gains, control delay, static friction,
observation noise, reset joint/root velocity sampling and periodic pushes.
The following ATDog randomizations are also implemented in the Cyberstar adapter:

| Addition | Range | Sampling |
| --- | --- | --- |
| Independent dynamic friction | 0.3–0.8 (static friction keeps its existing range) | Startup, 64 buckets |
| Restitution | 0–0.5 | Startup, 64 buckets |
| Per-collision-shape material assignment | Existing coefficient ranges | Startup, shared pool of 64 complete materials |
| Base external force / torque | Each axis ±10 N / ±10 N·m, local body frame | Reset, held until the next reset (same as ATDog) |
| Reset height / roll / pitch / yaw | 0–0.02 m / ±0.05 / ±0.05 / ±0.1 rad | Reset, composed with the existing spawn pose |

Stairs retain their existing randomized spawn yaw and heading reference; only
height, roll and pitch are added there. Existing XY placement, reset velocities,
PD/mass ranges and push magnitudes remain unchanged rather than copying stronger
ATDog settings onto a different robot. Each addition has a separate
`domain_rand.randomize_*` switch in `cyberstar/config.json` and
`cyberstar/stairs_config.json`. Flat-282 disables all these switches; playback
(`cyberstar_play`) skips the added randomizations and clears external wrenches.
Actor and critic dimensions, observation ordering and HIM network sizes are unchanged.
Material assignment now follows ATDog's per-collision-shape sampling; disable
`randomize_material_per_shape` to share one material within each environment.
The existing one-dimensional critic friction input holds the mean static friction.
Ground restitution is the multiplicative neutral value 1, so randomized robot
restitution actually affects contacts; disabling it still gives zero restitution.
Joint reset states are clipped to asset soft limits. Playback disables observation
noise and pushes inside the environment, including callers outside the RSL-RL script.

PD gains now come from `env.control.stiffness` and `env.control.damping` in JSON,
including exact-joint or full-match regex overrides. The identified passive motor
model is independent of PD damping: `env.control.motor_friction.coulomb` (N·m)
and `.viscous` (N·m·s/rad) contain all 12 joints, initially zero.
`domain_rand.randomize_joint_friction` samples additive coefficient offsets at
reset; both offset ranges initially contain only zero. Nominal friction also
applies in Flat-282 and playback, while coefficient randomization does not.
Drive torque is saturated before passive friction is subtracted, without a
second motor clamp on the resulting joint torque.

The renewed ATDog audit also adds mass-dependent inertia scaling
(`domain_rand.recompute_inertia`), reset joint velocity sampling (initially zero),
and per-environment resampled push intervals (`randomize_push_interval`): 10–15 s
on robust flat ground and 15–20 s on stairs. Zero offsets or disabling the switch
restore fixed intervals. The action-delay buffer follows the configured
maximum instead of assuming one delayed step. Link mass/COM sampling excludes
the base, dynamic friction cannot exceed static friction, and the robust-flat
push gate measures terrain-relative height. JSON timestep, decimation, action
scale and episode duration also drive the simulator and sensor timing. See
[`DOMAIN_RANDOMIZATION.md`](DOMAIN_RANDOMIZATION.md) for the complete source
inventory, units, configuration examples and intentional differences from ATDog.
Stair clearance rewards also honor `stair_clearance_progress_floor`: commanded
motion retains a lifting signal when blocked, while zero commands earn none.

Actor/critic hidden dimensions are `[512, 256, 128]` with ELU. HIM retains the
`[128, 64, 16]` encoder, `[128, 64]` target encoder and 32 prototypes.
The full Cyberstar critic starts with 491 inputs, matching the corresponding
CSDog1 stages. Robot-specific posture and geometry are adapted to 03dog:
0/-0.7854/0 joint angles, 0.40 m spawn height, 0.32 m height targets, URDF joint
limits, FK-derived footholds, and about 0.373 m stance width. All four FOOT
bodies, mesh collisions, masses and inertias come directly from the 03dog URDF;
CSDog1's USD foot reconstruction is replaced by local URDF spawning.

Train Flat-282, then continue its checkpoint into DomainRand or stairs using
the same training backend. All three now share the 491-D critic architecture.
Old 50-D critic checkpoints (including the previous standard Flat-282) cannot
be loaded directly into these tasks;
the observation semantics also differ. No CSDog1 checkpoint is selected
automatically. Training, resume and playback commands are in `AT_DESCRIPE.md`.
Logs continue to use independent `dog03_*` directories.
For a fresh robust-flat run, the documented command requests 2000 iterations
to cover the copied recovery curriculum's 1200-1600 iteration window. The
runner's default 1000 iterations remains the same as CSDog1's continuation budget.

`DOG03_STAIR_STAGE=0...4` selects terrain height bands and additional iterations.
The default is stage 0, independent of `CSDOG1_STAIR_STAGE`. The full local config
is `cyberstar/stairs_config.json`. Geometry, stage distributions and budgets are
copied from CSDog1; stage 4 uses 5-17 cm bands and 1500 iterations. Patches are
30% flat and 70% stairs, with 65% ascent among stairs. Resets place the robot on
the center platform near the stair edge, with a small heading perturbation.
Stages are selected between runs.

CPU checks (tensor runtime and a full-size HIM/PPO update, without physics):

```bash
python3 -m unittest discover -s tests -p 'test_dog03*.py'
```
