# CSDog1 训练与 Sim2Sim 核查

核查日期：2026-10-19。用户确认部署策略来自同步工作站的 `RobotLab-Isaac-Velocity-Flat-CSDog1-282-DomainRand-v0`。本地没有该策略对应的训练 checkpoint，因此以下是当前仓库资产与部署代码的核查，不是对工作站训练快照的追溯证明，也未运行闭环行走验证。

## 结论

机器人本体在数值精度内一致；部署的策略输入和控制参考没有与当前 DomainRand 任务对齐。两处确定的配置错误应优先处理，停步相位还需要实现层面对齐。

## 本体数值核查

训练实际读取 `source/robot_lab/data/Robots/csdog1/usd/csdog1.usd`，不是运行时读取 MJCF。

使用 OpenUSD 解析实际 USD（包括实例内的碰撞体），使用 MuJoCo 编译训练目录的 MJCF 和 `/space2/Project/rl_sar/src/rl_sar_zoo/csdog1_description/mjcf/scene.xml`，比较质量、质心、还原到 link 坐标系的完整惯量矩阵、关节轴、关节锚点、角度限位、力矩限幅，以及各碰撞体的中心、半径、胶囊端点、盒体半尺寸。USD 的角度换算成弧度，关节轴结合 localRot 还原，避免把 USD 的 X 轴关节误判为与 MJCF 的 Y 轴关节不同。

| 项目 | 最大绝对误差 |
|---|---:|
| link 质量 | 1.91e-7 kg |
| link 质心 | 1.86e-9 m |
| 完整惯量矩阵元素 | 2.10e-8 kg·m² |
| 关节父侧锚点 | 1.10e-8 m |
| 关节角度限位 | 2.16e-7 rad |
| 关节轴分量 | 1.35e-7 |
| 关节力矩限幅 | 0 N·m |
| 碰撞体中心 | 6.83e-9 m |
| 胶囊端点 | 1.75e-8 m |
| 足端球半径 | 8.95e-10 m |

总质量：MuJoCo 为 **15.176730 kg**，USD 为 **15.176729828 kg**。17 个源网格文件逐个 SHA-256 相同。两份 MJCF 编译后的质量、惯量、link 位姿、关节参数、执行器限幅、armature、被动阻尼和摩擦损失均相同。原始指标见同目录 `csdog1_model_audit_metrics.json`。

DomainRand 的 `cyberstar/asset.py:spawn_cyberstar` 会把 USD 的四个足球独立为 FOOT 刚体，所以运行时训练是 17 个刚体，MuJoCo 是 13 个。每只脚的 0.001 kg 从 calf 转移而来，代码同时补偿质心和惯量，并保留足球碰撞位置；这不是额外增加四个足端质量。RL 足球中心有约 1 µm 的构造差异，不足以解释明显迁移偏差。

双方均关闭机器人自碰撞。Sim2Sim 的传感器、IMU site、home keyframe 和相邻 link 排除项属于额外定义，没有发现改变本体尺寸或惯量的修改。home 中 `ctrl` 写的是角度，但 actuator 为 torque motor；不能把它当作 position target。当前运行时 PhysicsControl 会写入 PD 力矩。

## 确定的部署不匹配

| 项目 | 当前 DomainRand 训练 | 当前 Sim2Sim |
|---|---|---|
| 四个 calf 的默认角 | -0.823 rad | -0.623 rad |
| commands_scale | [2, 2, 0.25] | [1, 1, 0.25] |
| 停步相位 | 接触、摆腿落脚与机体状态参与结束判断 | 松开命令后固定 0.5 秒，再归零 |

默认角训练来源：`cyberstar/config.json` 的 `init_state.default_joint_angles`，以及 `cyberstar/env_cfg.py` 对 calf 初始角的覆盖。命令缩放训练来源：`cyberstar/runtime.py` 构造 `commands_scale`，从 `cyberstar/config.json` 读取 `normalization.obs_scales.lin_vel=2.0`。不要用普通 `rough_env_cfg.py` 中的 -0.623 和 1.0 推断这个任务；DomainRand 已使用独立 runtime 计算策略观测。

部署来源：`/space2/Project/rl_sar/policy/csdog1/robot_lab/config.yaml` 第 51、56 行，以及 `policy/csdog1/base.yaml` 中的默认角。

calf 参考角错误同时影响 `(q - q_default)` 和 `q_target = q_default + 0.25 * action`。在相同状态、相同 action 下，目标角偏差 +0.2 rad，Kp=30 时未限幅 PD 力矩偏差为 +6 N·m/腿；对 action scale=0.25，相当于 0.8 个 action 单位。命令缩放错误使 1 m/s 的命令在网络中变成训练时 0.5 m/s 的输入值。

停步逻辑来源：训练 `cyberstar/core.py` 的 DomainRand `_post_physics_step_callback` 记录接触、抬脚/落脚、对角支撑序列，并允许 0.5/0.9/1.2 秒等状态条件；部署 `library/core/rl_sdk/stop_phase_clock.hpp` 只读取 step 和 command。该差异无法只改一个 YAML 时长就完整解决，尤其可能影响松开命令后的落脚和站稳。

## 已对齐及后续验证顺序

当前关节/action 顺序均为 FL、FR、RL、RR，每腿 hip、thigh、calf；每帧 47 维，6 帧按时间从旧到新拼接为 282 维。部署 ONNX 的实际输入为 `[1, 282]`。物理步长 0.005 s、策略周期 0.02 s、Kp/Kd=30/1、action scale=0.25、力矩上限 31.5/31.5/40.5 均对齐。

建议先在两个部署 YAML 中将所有 calf 默认角改为 -0.823，在策略 config 中将 commands_scale 改为 [2, 2, 0.25]；lin_vel_scale 可同步设为 2.0，但当前 observations 没有 lin_vel，直接起作用的是 commands_scale。然后验证恒定小速度，再验证松开命令后的停步行为并移植训练的停步状态机。

训练 DomainRand 的控制延迟、质量/质心、摩擦、PD 和电机强度随机化，与 MuJoCo 固定名义参数也有差异；这是训练分布与名义评估的区别，不能直接认定为迁移错误。PhysX contact_offset=0.01 与 MuJoCo 接触求解同样不完全等价，应在前述输入和参考角修正后，再通过统一状态/动作的力矩与接触记录定位剩余误差。

本次仅新增核查文档及指标，没有修改机器人、策略或部署配置。工作站实际训练时的 env 配置、代码版本和导出 metadata 仍应与当前仓库对应，才能排除同步版本漂移。
