# 新 MNN 策略数值核查（2026-10-19）

## 结论

转换数值通过；相同输入的 MNN 与 ONNX 输出相符。当前 Sim2Real 和 rl_sar 的静态观测/动作缩放已一致，但完整输入输出处理尚未完全对齐。

核查使用实际 `MNNModel::Impl::compute`，链接现有 libMNN、使用部署配置的 CPU 两线程运行。ONNX 使用 ONNX Runtime CPU。没有连接机器人或启动控制器。不是只检查模型结构或 NumPy 重建权重。

## 模型与转换结果

转换源 `/home/lyz/桌面/cyberstartrainanddeploy/policy.onnx` 与 `/space2/Project/rl_sar/policy/csdog1/robot_lab/policy.onnx` 的 SHA-256 相同：

`f995ff2dfa4668ee31caad5d72297cf3c875109a6ccd7615e5317d073efa6de1`

本次 MNN 的 SHA-256：

`62771d7e992683ce013283a86f4a01d80090615869b82652379ef309b20f78b1`

共 965 次推理，包含站立、500 次随机运动状态、低速、运动到零命令的连续停步、航向保持，以及超量程观测裁剪压力样本。输入 282 维，输出 12 维。每次把实际部署构造并送入 MNN 的历史观测直接送入 ONNX，隔离模型转换误差。

| 指标 | 结果 |
|---|---:|
| 常规样本最大 action 绝对误差 | 2.981e-6 |
| 含裁剪压力样本的最大 action 绝对误差 | 1.221e-4 |
| 全样本平均 action 绝对误差 | 1.734e-6 |
| action 裁剪并乘 0.25 后最大转换误差 | 2.671e-5 rad |
| 全部输出有限 | 是 |
| allclose(atol=1e-4, rtol=1e-4) | 通过 |

压力样本人为使用极高角速度/关节速度，触发观测 ±100 裁剪；不能将其最大误差当成正常行走误差。

## 观测、缩放、输出

同一相位、命令和 previous action 下，按 rl_sar `ComputeObservation` 的公式、配置重建完整历史，与实际 MNN 前处理结果比较。常规片段误差不超过 3.58e-7。四元数使用 wxyz、单位四元数，投影重力方向和符号一致。Sim2Real 会额外归一化四元数，rl_sar 的该函数本身不归一化，因此不能推广到非单位/无效四元数。

| 单帧索引（从 0 开始） | 内容 | 缩放 |
|---|---|---|
| 0–1 | sin/cos 相位 | 1 |
| 2–4 | body 角速度 | 0.25 |
| 5–7 | 投影重力 | 1 |
| 8–10 | vx/vy/yaw 命令 | 2 / 2 / 0.25 |
| 11–22 | q - q_default | 1 |
| 23–34 | 关节速度 | 0.05 |
| 35–46 | 上一帧裁剪后的 action | 1 |

顺序：FL、FR、RL、RR；每腿 hip、thigh、calf。默认角每腿 [0, 0.785, -0.823]。六帧从旧到新排列，当前帧在最后 47 维，观测裁剪 ±100。Kp/Kd=30/1。

基础输出公式一致：`q_target = q_default + 0.25 * clip(action, -100, 100)`。未启用停步限速的样本中，实际 MNN 输出的目标与这一公式逐元素完全相同。这里验证的是角度目标，不代表实际机器人力矩与 MuJoCo 力矩完全相同。

## 尚未对齐的行为

1. **低速相位**：Sim2Real 使用新配置的平移 >0.2 或 |yaw|>0.05；rl_sar 仍使用平移 >=0.05 或 |yaw|>=0.05。0.1 m/s 测试下，sin/cos 分量最大差异 1.992。Sim2Real 已按用户要求向训练对齐，而 rl_sar 尚未同步此改动。
2. **航向保持**：Sim2Real 开启 translation_heading_hold，rl_sar 直接使用给定命令。姿态变化的纯平移测试中，yaw 命令观测最大差异 0.0625（0.25 rad/s × 0.25），与配置上限相符。该差异不是观测缩放错误。
3. **停步目标限速**：Sim2Real 根据上一目标限制 calf 伸展以及 settling 时的 hip/thigh/calf 目标变化；rl_sar 无此处理。合成运动→停步测试中，最终目标与基础公式最大差异 1.104 rad。该值只证明额外处理确实生效，不预测实际行走幅度。
4. **初始历史**：Sim2Real 复制首个实际观测六次，与训练一致；rl_sar 在 InitRL 中复制合成的名义姿态观测，随后逐帧插入实际观测。前五次策略输入不同，第六次合成帧才全部移出。
5. **控制循环平滑**：两套代码完整执行链还存在前次报告中的 1 kHz 目标平滑、停步状态机及真实下位机 PD/限幅差异。本次 harness 在 MNNModel 输出处取值，未运行 UDP/ROS 控制循环，不能把 MNNModel 的目标误差等同于发给机器人的最终目标误差。

rl_sar 相位使用其实际 `StopPhaseClock` C++ 类进行比对，其余观测使用源码公式和 YAML 重建，并未启动完整 rl_sar 模拟器。停步测试使用一条合成状态轨迹，相位在该轨迹中数值一致，不证明所有停步状态逻辑相同。

## 复核材料

- `metrics.json`：误差指标、配置核查、模型哈希。
- `vectors.npz`：全部合成状态、MNN 实际历史、MNN/ONNX action、MNNModel 目标和 rl_sar 相位。
- `audit.cpp`：离线调用实际部署前后处理和 MNN 推理的 harness，通过仅在测试翻译单元中开放 private 来检查中间历史，没有修改生产代码。
- `run.py`：固定随机种子 20261019 的样本与对比脚本；运行目录固定为 `/tmp/csdog1-numeric-audit`，重新运行需把 audit.cpp 编译为该目录的 audit，并将 run.py 放入该目录。依赖 numpy、onnxruntime、pyyaml 和现有 MNN/yaml-cpp。

本次没有修改策略、生产配置、rl_sar 或 Sim2Real 控制逻辑。后续若需要完整端到端一致，需明确统一低速阈值、航向保持、初始历史及停步输出限速的行为。
