# CSDog1 deployment observation contract

The training defaults follow `Cyberstar_rl_deploy/src/cyber1_edu_sdk/config/mnn_model_policy_stairs.yaml`
and `src/cyber1_edu_sdk/src/mnn_model.cpp`. All CSDog1 tasks inherit these defaults.

| Frame indices | Quantity | Scale |
| --- | --- | --- |
| 0–1 | sin/cos of gait phase | 1 |
| 2–4 | body angular velocity | 0.25 |
| 5–7 | projected gravity | 1 |
| 8–10 | vx, vy, yaw command | 1, 1, 0.25 |
| 11–22 | joint position minus default | 1 |
| 23–34 | joint velocity | 0.05 |
| 35–46 | previous clipped network action | 1 |

Joints are ordered FL, FR, RL, RR, each hip/thigh/calf. The default position
per leg is `[0, 0.785, -0.623]`; nominal PD gains are 30/1. The action is
clipped to ±100 before applying `q_target = q_default + 0.25 * action`.
Training keeps its configured observation noise and dynamics randomization.

At 50 Hz, phase advances by 0.02/0.50 per control step. A stop is requested
only when `hypot(vx, vy) < 0.05` and `abs(yaw) < 0.05`. After release, phase
continues for 0.50 s, then stays at zero (sin/cos = `[0, 1]`). A brief zero
crossing followed by a new command resumes the advancing clock. Reset clears
each environment's clock independently. Repeated observation reads within a
control step do not advance it.

Deployment's 0.10 s debounce and subsequent measured settling window do not
change this phase trajectory. Deployment's heading correction and stop-only
joint-target slew limits are runtime controls and are not reproduced here.

The HIM runner assembles six frames from oldest to latest into 282 dimensions
using `policy_frame_term_order`. Use the registered HIM training/play path for
that deployment export; Isaac Lab's raw concatenated per-term history has a
different layout. Existing checkpoints need retraining or fine-tuning after
the observation and reference-posture changes.

CPU regression checks (without Isaac Sim):

```bash
python3 -m unittest discover -s tests -p 'test_csdog1_*.py'
```
