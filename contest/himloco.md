# HIMLoco -> AT_rl_sar 移植记录

## 1. 目标与范围

本文记录将 `HIMLoco` 算法接入 `AT_rl_sar` 的完整过程，目标是：

- 不修改 docker、镜像和外部 `IsaacLab` 源码。
- 在 `AT_rl_sar` 仓内本地落一套 `HIM` 算法实现。
- 兼容 `Isaac Lab + Gymnasium + rsl_rl` 的训练、回放和导出链路。
- 让 `ATDog` 和标准四足任务都能通过 `rsl_rl_him_cfg_entry_point` 启动训练。
- 训练日志尽量贴近现有 PPO 输出，并补齐速度误差指标。

这次移植最终采用的是：

- 算法层面保留 `HIMLoco` 的核心结构。
- 训练配置、任务注册、场景组织、观测管理沿用 `AT_rl_sar` 现有体系。
- 所有适配都在本仓完成，不依赖外部改 `rsl-rl-lib`。


## 2. 最终落地结构

### 2.1 本地 HIM 模块

新增目录：

- `source/robot_lab/robot_lab/him/`

包含文件：

- `him_actor_critic.py`
- `him_estimator.py`
- `him_ppo.py`
- `him_rollout_storage.py`
- `him_on_policy_runner.py`
- `him_vec_env_wrapper.py`
- `exporter.py`

这套实现负责把 `HIMLoco` 的策略网络、估计器、PPO 更新、rollout 存储、runner 和导出链路全部本地化。

### 2.2 训练 / 回放入口

已修改：

- `scripts/reinforcement_learning/rsl_rl/train.py`
- `scripts/reinforcement_learning/rsl_rl/play.py`
- `scripts/reinforcement_learning/rsl_rl/play_cs.py`

入口侧新增了 `HIMOnPolicyRunner` 分支，并在需要时自动使用 `HIMVecEnvWrapper` 包装环境。

### 2.3 环境与任务侧

关键环境配置入口：

- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/commands.py`

任务注册和配置已补到：

- `quadruped/unitree_a1`
- `quadruped/atdog/dog2`
- `quadruped/atdog/dog2_arm`
- `wheeled/atdog`
- `wheeled/atdog_arm`


## 3. 为什么这样接

这次不是把 `AT_rl_sar` 移植到 `HIMLoco`，而是把 `HIMLoco` 算法接到 `AT_rl_sar` 训练框架里。原因很直接：

- `AT_rl_sar` 已经沉淀了任务注册、场景配置、terrain、reward、观测管理和 Isaac Lab 适配。
- `HIMLoco` 的核心增量主要在算法层，不在任务框架层。
- 如果反过来把 `AT_rl_sar` 的大量任务和场景系统搬到 `HIMLoco`，改动面会更大，回归风险也更高。

因此本次策略是：

1. 固定 `AT_rl_sar` 的任务系统。
2. 在本仓补本地 `HIM` 算法栈。
3. 用最小侵入方式接到现有 `rsl_rl` 入口。


## 4. 核心适配点

### 4.1 Runner 接线

`train.py` 原本只识别：

- `OnPolicyRunner`
- `DistillationRunner`

现在新增：

- `HIMOnPolicyRunner`

逻辑是：

- `agent_cfg.class_name == "HIMOnPolicyRunner"` 时使用 `HIMVecEnvWrapper`
- 否则继续走原有 `RslRlVecEnvWrapper`

这样不会影响现有 PPO 和蒸馏链路。

### 4.2 Isaac Lab / Gymnasium 接口适配

`HIMLoco` 原始实现默认面向较旧的 legged gym 风格接口，直接接 `Isaac Lab` 会出现多类问题：

- `env.reset()` / `env.step()` 返回签名不同
- 外层可能是 `OrderEnforcing` wrapper
- `RslRlVecEnvWrapper` 不直接暴露 `num_privileged_obs`
- 观测是按 `policy` / `critic` group 组织，而不是单个拼好的 tensor

为此新增了 `HIMVecEnvWrapper`，负责：

- 包装 `RslRlVecEnvWrapper`
- 从 `policy` group 提取 actor obs
- 从 `critic` group 提取 privileged obs
- 缓存最近一次 actor / critic obs
- 提供 `get_observations()`、`get_privileged_observations()`、`get_critic_observations()`
- 兼容 `reset()` / `step()` 的 Isaac Lab 返回格式

这个 wrapper 是整次移植里最关键的一层。

### 4.3 观测契约适配

`HIMLoco` 对观测有两个硬约束：

- actor 输入是 history obs
- estimator / critic 依赖 privileged obs

而 `AT_rl_sar` 的观测是由 observation manager 按 term 组织的，所以不能再硬编码维度和切片。现在的处理方式是：

- 从 `policy` group 自动推导 one-step obs 维度
- 将 Isaac Lab 的 term-major 历史布局重排为 HIM 使用的 current-frame-first 布局
- 从 `critic` group 自动推导 estimator velocity slice
- 从 policy / critic 的共享 term 自动推导 estimator target slice

对应能力都落在：

- `source/robot_lab/robot_lab/him/him_vec_env_wrapper.py`

这解决了最开始的两个典型报错：

- `AttributeError: 'RslRlVecEnvWrapper' object has no attribute 'num_privileged_obs'`
- `ValueError: Actor observation dim 256 is not divisible by one-step dim 45`

后者的根因是原始 `HIMLoco` 把 `45` 当成固定 one-step 输入，但当前任务实际 policy obs 已经开启 history，actor 输入维度不再是写死值。

### 4.4 Estimator 切片不再写死

`HIMLoco` 原版默认：

- `next_critic_obs[:, 45:48]` 是 `base_lin_vel`
- `next_critic_obs[:, 3:48]` 是 `target input`

这在 `AT_rl_sar` 里并不可靠，因为 observation term 排布由环境配置决定。现在统一以 observation manager 的实际布局为准：

- `num_one_step_obs` 从 policy observation terms 自动推导
- `estimator_vel_slice` 从 critic 的 `base_lin_vel` term 自动推导
- `estimator_target_slice` 从 policy / critic 共享且连续的 terms 自动推导
- 即使旧配置中的切片维度合法，只要和环境布局不一致，也优先使用环境推导结果

对应逻辑在：

- `source/robot_lab/robot_lab/him/him_on_policy_runner.py`

这样可以避免配置中的 `(45, 48)` 在当前 critic 布局中误指向 action，而不是 `base_lin_vel`。

### 4.5 对 done 样本做 mask

原始 `HIMLoco` 某些实现会依赖环境返回额外的 termination privileged obs。为了减少对环境接口的侵入，这次采用的是更保守的方案：

- estimator 更新时直接对 `done` 样本做 mask
- 不强依赖环境额外返回 `termination_privileged_obs`

这样更适合 `AT_rl_sar` 当前环境栈。

### 4.6 导出链路修复

`HIMLoco` 原始导出逻辑常把输入维度写死成 `45`，这在当前仓里会直接卡住部署。现在本地 exporter 已改成按当前 actor-critic 实例的真实维度导出：

- `num_one_step_obs` 来自模型实例
- ONNX dummy input 使用 `actor_critic.num_actor_obs`

文件：

- `source/robot_lab/robot_lab/him/exporter.py`

同时 `play.py` / `play_cs.py` 已支持对 `HIMOnPolicyRunner` 正常加载和导出。


## 5. 配置注册方式

### 5.1 配置文件组织原则

`HIM` 配置现在按本仓 PPO 配置风格显式展开，不再依赖公共 `_him_cfg.py` 模板。这样做是为了：

- 更接近现有 `cusrl_ppo_cfg.py` / `rsl_rl_ppo_cfg.py`
- 每个任务的超参更直观
- 后续局部改动不会牵一发动全身

例如：

- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/atdog/dog2_arm/agents/rsl_rl_him_cfg.py`

### 5.2 注册入口

每个任务的 `__init__.py` 都补了：

- `rsl_rl_him_cfg_entry_point`

这样 Hydra 才能识别：

```bash
--agent=rsl_rl_him_cfg_entry_point
```

这个问题对应修复了下面的报错：

```text
ValueError: Could not find configuration for the environment ...
Please check that the gym registry has the entry point: 'rsl_rl_him_cfg_entry_point'
```


## 6. 训练日志适配

### 6.1 目标

用户希望 `HIM` 的终端训练反馈尽量接近现有 PPO，例如要能看到：

- `Mean reward`
- `Mean episode length`
- `Episode_Reward/*`
- `Episode_Termination/*`
- `Metrics/base_velocity/*`

并且这两项必须优先展示：

- `Metrics/base_velocity/error_vel_xy`
- `Metrics/base_velocity/error_vel_yaw`

### 6.2 实现

`HIMOnPolicyRunner` 现在会：

- 聚合 `infos["log"]`
- 读取 Isaac Lab manager 自动写回的 reward / metrics / termination 信息
- 缓存 `reset()` 阶段的 log，避免前几轮完全没有 metrics
- 当本轮没有完整 episode 结束时，回退到当前 partial trajectory 的平均 reward 和长度
- 打印更接近 PPO 的大块终端日志

另外，`HIMPPO.update()` 现在还会回传：

- `mean_entropy_loss`

因此日志中可同时看到：

- `Mean estimation loss`
- `Mean swap loss`
- `Mean entropy loss`

### 6.3 为什么之前只有两个速度误差

不是 runner 少打了，而是 Isaac Lab 原始 `UniformVelocityCommand` 默认只统计：

- `error_vel_xy`
- `error_vel_yaw`

所以即便 runner 能打印，也没有更多字段可打。

### 6.4 本仓补充的速度误差指标

由于不能改外部 `IsaacLab` 源码，因此在本仓本地 command 扩展了 metric。修改文件：

- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/commands.py`

本地 `UniformThresholdVelocityCommand` 现在额外输出：

- `Metrics/base_velocity/error_vel_x`
- `Metrics/base_velocity/error_vel_y`
- `Metrics/base_velocity/error_vel_z`
- `Metrics/base_velocity/error_vel_dir`

含义如下：

- `error_vel_x`: base frame 下 x 方向线速度误差
- `error_vel_y`: base frame 下 y 方向线速度误差
- `error_vel_z`: base z 方向线速度绝对值，等价于相对目标 `0` 的误差
- `error_vel_dir`: 实际 XY 速度方向与命令 XY 速度方向的夹角误差

同时 runner 里也把这些 metric 调整到优先输出顺序，因此现在终端能先看到细粒度速度误差，再看到其他 reward 和 termination 指标。


## 7. 这次实际遇到的问题与修复

### 7.1 `ModuleNotFoundError: No module named 'isaaclab'`

原因：

- 直接用系统 `python` 启动，没进 Isaac Lab 对应 Python 环境。

处理：

- 需要在 Isaac Lab 对应环境中运行，或者走 Isaac Lab 提供的启动脚本。

### 7.2 启动时被系统 `Killed`

现象：

- scene 创建完成后进仿真阶段被直接杀掉。

原因：

- 常见是显存或内存不足，尤其是 rough terrain + `4096` env。

处理建议：

- 先把 `num_envs` 降到 `64` 或 `128`
- 回放时关闭不必要 terrain curriculum
- 验证链路先用标准四足 rough 任务

### 7.3 `num_privileged_obs` / `get_privileged_observations` 缺失

原因：

- `RslRlVecEnvWrapper` 并不是按 `HIMLoco` 预期设计的。

处理：

- 在 `HIMVecEnvWrapper` 中统一适配，并从 observation groups 中提取 `policy` / `critic` 观测。

### 7.4 `HIMActorCritic.__init__() got multiple values for argument 'num_one_step_obs'`

原因：

- `HIMOnPolicyRunner` 初始化 `HIMActorCritic` 时手动传了一次 `num_one_step_obs`
- `policy_cfg` 里又带了一次

处理：

- runner 初始化前先 `pop("num_one_step_obs", None)`，避免重复传参。

### 7.5 `Actor observation dim 256 is not divisible by one-step dim 45`

原因：

- 任务已经启用了 history obs，但配置里还写死 `45`。

处理：

- 让 wrapper 从 observation manager 自动推导 one-step obs dim。
- 若配置值和推导值冲突，以推导值为准，并打印 warning。

### 7.6 Hydra 找不到 HIM 配置

原因：

- 没有在任务注册入口中补 `rsl_rl_him_cfg_entry_point`。

处理：

- 给相关任务的 `__init__.py` 全部补齐 entry point。

### 7.7 HIM 日志不像 PPO

原因：

- 原始 runner 只打印 loss，不消费 `infos["log"]`。

处理：

- 在 `HIMOnPolicyRunner` 中增加 `infos["log"]` 聚合、缓存和格式化打印。


## 8. 建议的任务接入顺序

如果后续继续扩展任务，建议按下面顺序验证：

1. `UnitreeA1 rough`
2. `ATDog Dog2 rough`（使用已注册的 PPO 配置）
3. `stairs / sand / slope`
4. `wheeled`
5. `arm`
6. `humanoid`

原因：

- 标准四足 rough 的观测契约更稳定，最适合先验证算法链路。
- `ATDog` 的任务配置更复杂，直接上业务机器人调试成本更高。


## 9. 后续新增任务时的操作模板

### 9.1 环境侧

确保 observation contract 满足：

- `policy` 只保留单帧本体观测，并通过 history 提供时序输入
- `critic` 保留单帧 privileged obs
- actor 和 critic 的 term 命名尽量保持一致，便于自动推 estimator target slice

### 9.2 配置侧

新增一个任务专属 `rsl_rl_him_cfg.py`，内容至少包括：

- `class_name = "HIMOnPolicyRunner"`
- `policy.class_name = "HIMActorCritic"`
- `algorithm` 沿用 PPO 配置起步

优先调的超参一般是：

- `history_length`
- `entropy_coef`
- `learning_rate`
- `estimator_learning_rate`
- `actor_hidden_dims`
- `critic_hidden_dims`

### 9.3 注册侧

在任务目录的 `__init__.py` 里补：

- `rsl_rl_him_cfg_entry_point`

### 9.4 验证侧

先检查四件事：

1. 训练入口能否正常实例化 `HIMOnPolicyRunner`
2. actor obs 和 critic obs 维度是否正确
3. 日志中是否出现 `Metrics/base_velocity/error_vel_xy` 和 `error_vel_yaw`
4. 日志中是否出现 `error_vel_x` / `error_vel_y` / `error_vel_z` / `error_vel_dir`


## 10. 当前结果

截至 2026-07-25，这次移植已经完成以下能力：

- 本仓本地 HIM 算法栈已接入
- `train.py` / `play.py` / `play_cs.py` 已支持 HIM runner
- `Isaac Lab + Gymnasium` 接口已适配
- Isaac Lab term-major history 已转换为 HIM current-frame-first history
- estimator 切片不再强依赖原始 `HIMLoco` 固定索引
- ATDog 和 Unitree A1 已补 HIM 配置与注册
- 导出链路已改为按真实维度导出
- HIM 终端日志已接近 PPO 风格
- 已补细粒度速度误差指标：
  - `error_vel_x`
  - `error_vel_y`
  - `error_vel_z`
  - `error_vel_dir`
  - `error_vel_xy`
  - `error_vel_yaw`


## 11. 最小启动示例

训练：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0 \
  --agent=rsl_rl_him_cfg_entry_point
```

如果机器资源紧张，建议加：

```bash
--num_envs=64
```

回放：

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0 \
  --agent=rsl_rl_him_cfg_entry_point \
  --checkpoint=<checkpoint_path>
```


## 12. 总结

这次移植的核心不是“把 `HIMLoco` 代码拷进来”，而是把它的算法约束重新映射到 `AT_rl_sar` 的任务和观测体系中。真正决定成败的是三点：

- `policy` / `critic` 观测的稳定拆分
- estimator 切片从硬编码改成配置化和自动推导
- 日志、导出、Hydra 注册这些工程链路全部补齐

只要后续继续遵守这三个约束，新增四足任务的 HIM 接入成本会比较低；真正复杂的部分会更多集中在任务观测契约，而不是算法本身。


## 13. 2026-07-26 移植与六帧历史评估

### 13.1 提交评估

- `7d5e5be52aa67a6076a1e516102ff449ec396dd4` 完成了 HIM 算法栈移植，覆盖 actor-critic、estimator、PPO、storage、runner、vec wrapper、导出和任务注册。整体移植方向正确，并且比原始 `HIMLoco` 更适合本仓的一点是 estimator 的 `base_lin_vel` / target slice 会优先从 Robot Lab observation layout 自动推断，而不是继续依赖原始工程的固定 `(45, 48)` / `(3, 48)` 索引。
- `d32ca17d59b8906260176b48065e796086ca7ac5` 把 HIM policy 观测切到 6 帧历史，训练、play、play_cs 都会在 HIM runner 下自动设置 `policy.history_length = 6` 和 `flatten_history_dim = True`。这解决了原始移植后 actor 实际仍可能只吃单帧输入的问题。
- 当前工作区新增部署元数据导出后，HIM 模型导出会额外生成 `policy_metadata.json` 和 AT_robot-lab 历史配置片段，便于实机侧检查 one-step 维度、history 长度、term 顺序和不支持 term。

### 13.2 六帧历史效果

六帧历史对 HIM 是必要项，不只是提高性能的超参。HIM actor 实际输入为：

```text
[current_one_step_obs, estimated_base_lin_vel, latent]
```

其中 `estimated_base_lin_vel` 和 `latent` 都来自 estimator 对完整历史观测的编码。没有稳定的历史输入时，estimator 只能从单帧本体状态猜速度和隐变量，退化明显。

当前实现的历史顺序是：

```text
[t0_all_terms, t-1_all_terms, t-2_all_terms, ..., t-5_all_terms]
```

即 time-major、latest-to-oldest。Isaac Lab 内部历史 buffer 输出 oldest-to-current，`HIMVecEnvWrapper` 会倒序重排后再给 HIM。这个顺序与 AT_robot-lab 的 `ObservationBuffer` 在 `observations_history_priority: "time"`、`observations_history: [0, 1, 2, 3, 4, 5]` 下完全一致。

### 13.3 AT_robot-lab sim2real 适配性

AT_robot-lab 侧已经支持历史观测：

- `observations_history` 中 `0` 表示最新帧，`5` 表示最旧帧。
- `observations_history_priority: "time"` 会按整帧拼接，符合 HIM 导出的 JIT/ONNX 输入。
- `InitRL()` 会用当前观测预填满 history buffer，避免实机启动前几帧喂零历史。

需要注意的部署约束：

- `observations` 顺序必须和训练 policy term 顺序一致。ATDog2 当前训练侧顺序是 `ang_vel, gravity_vec, commands, dof_pos, dof_vel, actions`。
- `num_observations` 必须是一帧维度，HIM ATDog2 为 45；模型实际输入是 `45 * 6 = 270`。
- stairs / sand / bar / slope 等 AT_robot-lab 配置目前仍多为 `observations_history: []`，如果部署 HIM checkpoint，必须改为 `[0, 1, 2, 3, 4, 5]`。

### 13.4 已做优化

- estimator 更新现在跟随 PPO adaptive KL 调度后的 `learning_rate`，与原始 HIMLoco 的训练节奏一致，避免 PPO 学习率已经降/升而 estimator 仍固定在初始 lr。
- AT_robot-lab 配置片段导出文件改为 `at_robot_lab_history.yaml`，不再默认覆盖导出目录下可能已有的正式 `config.yaml`。
- 导出的 AT_robot-lab 片段增加 `policy_input_dim` 注释，便于部署时确认 JIT/ONNX 输入维度是否等于 `num_observations * len(observations_history)`。

### 13.5 后续优化优先级

1. 先用 `--export-only` 对每个 HIM checkpoint 生成 `policy_metadata.json`，确认 `policy_terms` 与 AT_robot-lab YAML 的 `observations` 一致。
2. 把 AT_robot-lab 的 stairs / sand / bar / slope / bridge HIM 策略配置统一开启六帧历史。
3. 训练侧保持 policy 不含 `base_lin_vel`，critic 保留 `base_lin_vel`，否则 estimator 目标会失去 sim2real 意义。
4. 若继续强化粗糙地形能力，优先微调奖励和 domain randomization，不建议把 height scan 加回 policy；加回会提高仿真成绩，但会破坏当前无高度传感器的实机输入契约。


## 14. 2026-07-26 Dog2 抖动问题调参记录

现象：

- 无速度命令时，机身前后上下抖动。
- 有速度命令时，步态推进效果差，前后俯仰/上下振荡明显。

主要判断：

- 零速命令样本比例原来只有 `rel_standing_envs=0.02`，策略很少真正学习“停住”。
- `feet_air_time=50`、`feet_gait=15`、`track_ang_vel_z=40` 偏激进，容易鼓励持续抬腿和强行转向，零速附近也会把策略推向动态步态。
- `upward` 项返回的是姿态偏差平方，正权重会奖励偏差，不适合作为稳身项。
- `flat_orientation_l2`、`ang_vel_xy_l2`、`lin_vel_z_l2` 和 `body_lin_acc_l2` 对前后俯仰/上下振荡约束不足。
- reset 初始 roll/pitch/速度扰动过大，不适合作为当前阶段的稳定步态起点。

已调整的训练起点：

- 零速样本比例提高到 `0.2`。
- 命令范围收窄到 `x=(-0.8, 0.8)`、`y=(-0.25, 0.25)`、`yaw=(-0.6, 0.6)`。
- 降低速度追踪、腾空时间和步态同步奖励，避免为了追踪命令牺牲机身稳定。
- 增强 `stand_still`、`joint_pos_penalty`、`action_rate_l2`、`joint_acc_l2`、`joint_power`。
- 开启 `flat_orientation_l2=-2.0`，加大 `lin_vel_z_l2=-8.0`、`ang_vel_xy_l2=-1.0`，新增 `body_lin_acc_l2=-1e-4`。
- 关闭 `upward`。
- reset 初始姿态/速度扰动收窄，先让策略学会稳定站立和低速步态。
- 补充 `action_smoothness_2_l2` 二阶动作差分惩罚，抑制相邻动作变化方向来回翻转造成的高频抖动。

建议训练流程：

1. 先训练这个保守版本，观察零速站立是否不再周期性点头/弹跳。
2. 如果零速稳定但走得慢，再逐步提高 `track_lin_vel_xy_exp` 到 `30-40`，不要直接回到 `50+`。
3. 如果脚拖地，再把 `feet_air_time` 从 `8` 提到 `12-18`，或把 threshold 从 `0.25` 提到 `0.3`，不要回到 `0.5/50`。
4. 如果台阶能力不足，优先做地形课程或命令课程，而不是把姿态稳定惩罚降掉。

### 14.1 有速度命令时上下抖和非对角步态

新现象：

- 无速度命令时已经基本不抖，说明站立项和零速样本比例有效。
- 有速度命令时仍有机身上下弹跳，且步态没有稳定形成对角 trot。

本轮判断：

- `track_lin_vel_xy_exp=70`、`track_ang_vel_z_exp=50` 对当前阶段偏强，策略容易用弹跳/冲击换速度追踪。
- 仅靠 `feet_gait` 的接触/腾空时间同步约束不够直接，不能强力排除 pacing、bounding 或四脚同相跳。
- 需要把“有命令时的动态稳定”和“当前接触模式必须接近对角步态”分开约束，避免破坏已经稳定的零速站立。

已调整：

- 新增 `diagonal_trot_contact_pattern`，只在命令范数超过阈值时启用，惩罚非 `FL+RR` / `FR+RL` 对角接触模式。
- `feet_gait` 提高到 `12.0`，继续约束对角腿相位同步。
- `track_lin_vel_xy_exp` 降到 `50.0`，`track_ang_vel_z_exp` 降到 `25.0`，减少为追踪命令牺牲机身稳定的倾向。
- `lin_vel_z_l2=-10.0`、`ang_vel_xy_l2=-6.0`、`body_lin_acc_l2=-2e-4`，增强行走时机身上下/俯仰抑振。
- `feet_air_time=5.0`、`feet_air_time_variance=-6.0`，降低鼓励大幅抬腿的强度，同时保持四腿节律一致。

下一轮观察重点：

1. 如果速度明显变慢但步态变成对角步，先保持该配置继续训练，再逐步把 `track_lin_vel_xy_exp` 提到 `55-60`。
2. 如果仍四脚跳或 pacing，把 `diagonal_trot_contact_pattern.weight` 从 `-1.0` 加到 `-1.5`，不要优先提高 `feet_air_time`。
3. 如果对角步态出现但脚拖地，再小幅提高 `feet_air_time` 到 `6-8`，或把 threshold 从 `0.25` 提到 `0.28`。



• 以 ATDog Dog2 为例，当前 HIMLoco 移植“接线基本完整”，但仍有几个需要特别注意的适配问题。最重要的结论是：

  > Dog2 的 HIM 配置可以进入训练流程，观测维度和网络维度原则上能够自动匹配；但它并不是 HIMLoco 原始 A1/Go1 配置的严格等价版本，主要差
  > 异来自 Dog2 的观测项、动作缩放、命令维度和训练奖励。

  ## 1. Dog2 当前配置结构

  Dog2 已注册 6 个 HIM 任务入口：

   任务      环境配置             HIM 配置
  ━━━━━━━━  ━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Flat      flat_env_cfg.py      ATDogDog2FlatHIMRunnerCfg
  ────────  ───────────────────  ─────────────────────────────
   Rough     rough_env_cfg.py     ATDogDog2RoughHIMRunnerCfg
  ────────  ───────────────────  ─────────────────────────────
   Stairs    stairs_env_cfg.py    ATDogDog2StairsHIMRunnerCfg
  ────────  ───────────────────  ─────────────────────────────
   Sand      sand_env_cfg.py      ATDogDog2SandHIMRunnerCfg
  ────────  ───────────────────  ─────────────────────────────
   Slope     slope_env_cfg.py     ATDogDog2SlopeHIMRunnerCfg
  ────────  ───────────────────  ─────────────────────────────
   Bar       bar_env_cfg.py       ATDogDog2BarHIMRunnerCfg

  注册位置：

  source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/atdog/dog2/init.py (source/robot_lab/robot_lab/
  tasks/manager_based/locomotion/velocity/config/quadruped/atdog/dog2/__init__.py)

  这部分是完整的，Hydra 理论上可以通过：

  --agent rsl_rl_him_cfg_entry_point

  加载对应 HIM runner。

  ## 2. Dog2 的实际 actor observation 组成

  Dog2 在 rough_env_cfg.py 中关闭了：

  self.observations.policy.base_lin_vel = None
  self.observations.policy.height_scan = None

  因此策略侧保留：

  1. base_ang_vel
  2. projected_gravity
  3. velocity_commands
  4. joint_pos
  5. joint_vel
  6. actions

  Dog2 有 12 个关节，因此：

  - base_ang_vel: 3
  - projected_gravity: 3
  - velocity_commands: 通常为 4
  - joint_pos: 12
  - joint_vel: 12
  - actions: 12

  所以 Dog2 的 one-step actor observation 预计为：

  3 + 3 + 4 + 12 + 12 + 12 = 46

  HIM 配置指定：

  history_length = 6

  因此 actor 输入维度应为：

  46 × 6 = 276

  这里与 HIMLoco 原始 A1/Go1 实现存在一个重要差异：原始代码通常按 45 维 one-step observation 处理，而 Dog2 的 velocity_commands 可能
  包含 heading，因此是 4 维而不是 3 维。

  当前移植通过 HIMVecEnvWrapper 动态推导 one-step 维度，因此不会硬编码为 45。这是正确的适配方式，但也意味着 Dog2 的网络输入并不与
  HIMLoco 原始 checkpoint 兼容。

  ## 3. Dog2 的 critic observation 组成

  Dog2 的 critic 侧没有关闭 base_lin_vel，因此包含：

  1. base_lin_vel
  2. base_ang_vel
  3. projected_gravity
  4. velocity_commands
  5. joint_pos
  6. joint_vel
  7. actions

  预计 critic 维度为：

  3 + 3 + 3 + 4 + 12 + 12 + 12 = 49

  这与文件中的注释相互印证：

  critic 第一层权重 shape mismatch（例如 48 vs 235）

  当前 HIM critic 的第一层输入维度会动态设置为环境实际的 num_privileged_obs，Dog2 大概率为 49，而不是 HIMLoco 原版常见的 48。

  ## 4. estimator velocity slice 是否正确

  当前 HIMVecEnvWrapper 会根据 critic observation term 自动寻找：

  base_lin_vel

  Dog2 critic 的第一项就是 base_lin_vel，因此推导结果应为：

  estimator_vel_slice = (0, 3)

  这和 HIMLoco 原始实现的：

  next_critic_obs[:, 45:48]

  完全不同，但对当前 Dog2 是正确的。

  这也是当前移植中不能继续使用 HIMLoco 原始硬编码切片的原因。

  ## 5. estimator target slice 是否正确

  当前代码的 target slice 推导逻辑是：

  _shared_policy_target_slice()

  它寻找 policy 和 critic 中名称相同、并且在 critic 中连续的 observation term。

  Dog2 中：

  - policy 去掉了 base_lin_vel
  - critic 保留了 base_lin_vel
  - 其余 term 顺序基本一致

  因此共享项在 critic 中应为：

  base_ang_vel       [3:6]
  projected_gravity  [6:9]
  velocity_commands  [9:13]
  joint_pos          [13:25]
  joint_vel           [25:37]
  actions             [37:49]

  推导出的 target slice 大概率是：

  estimator_target_slice = (3, 49)

  这对应：

  critic 的 base_ang_vel 到 actions

  也就是“critic privileged base velocity 之外的完整 one-step actor observation”。

  这个结果在语义上与 HIMLoco 原始实现的：

  target_obs = next_critic_obs[:, 3:48]

  是一致的：都使用 privileged base velocity 加上可观测 one-step 状态作为 estimator target。

  ### 这里有一个隐患

  当前 _shared_policy_target_slice() 假设：

  1. policy 和 critic 的 term 名称一致；
  2. critic 中共享项连续；
  3. policy 与 critic 的顺序相同；
  4. 所有共享项都应该纳入 target。

  Dog2 当前满足这些条件，但如果后续在 critic 中插入独有 term，或者调整 term 顺序，slice 推导可能返回 None 或者包含不应作为 target 的
  项。

  建议在初始化时打印并强校验：

  policy term layout
  critic term layout
  estimator_vel_slice
  estimator_target_slice

  当前 HIMOnPolicyRunner 已经打印部分布局信息，但还应该把具体 term slice 一并输出，方便确认 Dog2 的实际结果。

  ## 6. Dog2 的历史观测顺序

  Isaac Lab 的 observation manager 通常按 term-major 组织历史，例如：

  base_ang_vel[t0:t6], gravity[t0:t6], joint_pos[t0:t6], ...

  而 HIMLoco 使用的是：

  frame_0: 当前时刻所有 term
  frame_1: 上一时刻所有 term
  ...
  frame_5: 更早时刻所有 term

  当前 HIMVecEnvWrapper._reorder_policy_history() 会把各 term 的历史重新组织为：

  [current frame, previous frame, ..., oldest frame]

  这一步对 Dog2 是必要的，否则 estimator 的输入会把不同时间的同一观测项拼在一起，时间结构会被破坏。

  当前代码的处理逻辑是合理的，但需要注意：

  - history_length 必须对所有 policy term 一致；
  - joint_pos、joint_vel 的历史长度必须与其他 term 相同；
  - actions 也会被纳入历史；
  - 当前帧必须位于最前面。

  Dog2 配置中没有为单独 term 指定 history length，因此主要依赖环境级：

  policy_obs_cfg.history_length = 6
  policy_obs_cfg.flatten_history_dim = True

  这部分由 train.py 的 apply_him_observation_history() 自动设置。

  ## 7. Dog2 与 HIMLoco 原始算法的实际差异

  ### 7.1 one-step 维度不同

  HIMLoco 原版常见：

  one-step obs = 45
  history = 6
  actor obs = 270
  critic obs = 48

  Dog2 预计为：

  one-step obs = 46
  history = 6
  actor obs = 276
  critic obs = 49

  因此：

  - 原 HIMLoco checkpoint 不能直接加载到 Dog2；
  - 当前兼容加载函数只能加载形状一致的参数；
  - actor 第一层和 critic 第一层通常会被跳过；
  - estimator encoder 第一层也会因输入维度变化而无法复用。

  ### 7.2 command 维度可能不同

  Dog2 的命令配置使用：

  heading_command=True

  因此 generated_commands 可能包含：

  lin_vel_x, lin_vel_y, ang_vel_z, heading

  即 4 维。

  HIMLoco 原始任务通常使用 3 维速度命令。这个差异会影响：

  - actor 输入维度；
  - estimator target 输入；
  - checkpoint 兼容性；
  - 策略行为；
  - 训练超参数与论文结果可比性。

  如果目标是严格复现 HIMLoco，建议关闭 heading command，或者显式改成与原始 3 维命令一致。

  ### 7.3 动作缩放不同

  Dog2 使用：

  self.actions.joint_pos.scale = {
      ".*_hip_joint": 0.125,
      "^(?!.*_hip_joint).*": 0.25,
  }

  而 HIMLoco 原始 A1/Go1 通常使用统一 action scale，或者与机器人默认关节比例不同。

  这意味着即使网络输出相同，最终关节目标也不同。动作缩放会显著影响：

  - 步态幅度；
  - 稳定性；
  - 探索噪声实际大小；
  - reward 分布；
  - sim-to-real 行为。

  所以不能直接将 HIMLoco 的训练曲线或 checkpoint 性能映射到 Dog2。

  ### 7.4 observation normalization 被关闭

  Dog2 HIM 配置中：

  actor_obs_normalization=False
  critic_obs_normalization=False

  HIMLoco 原始 ActorCritic 实现中虽然有 normalization 类，但是否启用取决于训练版本和配置。

  当前关闭 normalization 的后果是：

  - joint_vel 经过 scale=0.05 后幅值较小；
  - command、gravity、joint position 的量纲不同；
  - estimator encoder 直接接收混合尺度输入；
  - 对噪声和随机化更加敏感。

  Dog2 中 joint_vel 的 policy scale 已改为 0.05，这有利于抑制速度量级，但应确认这与 HIMLoco 原始输入尺度一致。

  ## 8. Dog2 各地形配置的一致性

  Dog2 的 HIM agent 配置中，rough/sand/slope/stairs 使用独立 runner class，但网络与 PPO 超参数完全相同：

  num_steps_per_env = 24
  num_learning_epochs = 5
  num_mini_batches = 4
  learning_rate = 1e-3
  gamma = 0.99
  lam = 0.95
  schedule = "adaptive"

  Flat 和 Bar 通过 rough runner 继承，只修改：

  max_iterations = 5000
  experiment_name

  这在结构上没有问题，但需要确认对应环境配置是否真的只改变地形而没有改变 observation term。尤其要检查：

  - flat_env_cfg.py
  - sand_env_cfg.py
  - slope_env_cfg.py
  - stairs_env_cfg.py
  - bar_env_cfg.py

  是否重新启用了 height_scan 或改变了 policy/critic 的 term 顺序。

  如果某个子环境重新启用 height scan：

  - actor one-step 维度会增加；
  - critic 维度会增加；
  - target slice 可能改变；
  - 所有该环境之前生成的 checkpoint 都会不兼容。

  ## 9. 当前实现中需要修正或加强的地方

  ### 9.1 HIMActorCritic 默认 slice 仍不安全

  当前默认值仍是：

  estimator_vel_slice=(45, 48)
  estimator_target_slice=(3, 48)

  虽然 Dog2 通过 HIMOnPolicyRunner 会被环境推导值覆盖，但直接实例化仍可能出错。

  建议改成：

  estimator_vel_slice: tuple[int, int] | None = None
  estimator_target_slice: tuple[int, int] | None = None

  并在构造函数中强制校验：

  if estimator_vel_slice is None:
      raise ValueError(...)

  或者只允许 runner 注入。

  ### 9.2 缺少 critic 时不应静默回退

  当前 HIMVecEnvWrapper 在没有 critic observation 时可能回退到 actor observation：

  if critic_obs is None:
      critic_obs = self._compute_group("critic")

  而 HIM estimator 明确需要：

  - privileged base velocity；
  - target observation；
  - next critic observation。

  因此 Dog2 虽然有 critic，不会触发这个问题，但通用实现中建议改成：

  缺少 critic group
  → 直接报错

  而不是静默使用 actor observation。否则其他任务可能“能运行但 estimator 学错目标”。

  ### 9.3 target slice 需要防止错误包含额外 critic term

  Dog2 当前 target (3, 49) 是合理的。但未来如果 critic 中增加额外共享 term，例如 height scan，当前逻辑可能把它一起纳入 target。

  建议 target 推导不要只依赖“所有共享 term 连续”，而应明确指定：

  target terms = policy terms excluding base_lin_vel

  或者显式在配置中声明 target term 名称列表。

  ### 9.4 需要验证命令实际输出维度

  不能只根据配置猜测 velocity_commands 是 4 维。应在真实环境初始化后打印：

  env.observation_manager.group_obs_term_dim["policy"]
  env.observation_manager.group_obs_term_dim["critic"]

  重点确认：

  velocity_commands = 3 还是 4
  actor one-step = 45 还是 46
  critic obs = 48 还是 49

  这是 Dog2 适配的第一项运行时检查。

  ## 10. 推荐的 Dog2 验证脚本/流程

  ### 第一步：打印实际 observation layout

  启动最小环境后，应看到类似：

  HIM observation layout:
  actor_obs=276,
  one_step_obs=46,
  history=6,
  critic_obs=49,
  vel_slice=(0, 3),
  target_slice=(3, 49)

  如果得到：

  actor_obs=270, one_step_obs=45, critic_obs=48

  说明 command 实际为 3 维，这也可以接受，但需要记录下来。

  ### 第二步：短训练

  建议先运行：

  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task RobotLab-Isaac-Velocity-Rough-ATDog-Dog2-v0 \
    --agent rsl_rl_him_cfg_entry_point \
    --max_iterations 2

  重点检查：

  - 环境能否 reset；
  - HIMVecEnvWrapper 能否提取 policy/critic；
  - actor 输入维度是否匹配；
  - estimator update 是否报 shape error；
  - dones mask 是否正常；
  - rollout storage 是否能完成；
  - checkpoint 是否保存。

  ### 第三步：检查训练指标

  HIM runner 应重点观察：

  mean_value_loss
  mean_surrogate_loss
  mean_estimation_loss
  mean_swap_loss
  mean_entropy_loss

  期望现象：

  - estimation_loss 能下降；
  - swap_loss 不出现 NaN；
  - action std 不快速爆炸；
  - adaptive KL 不持续极端调节学习率；
  - reward 不在最初几轮直接变成 NaN。

  ### 第四步：回放和导出

  验证：

  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task RobotLab-Isaac-Velocity-Rough-ATDog-Dog2-v0 \
    --agent rsl_rl_him_cfg_entry_point \
    --checkpoint <checkpoint>

  然后检查：

  - checkpoint 是否能加载；
  - actor history 输入是否保持 276/270 维；
  - JIT 输出动作是否与实时策略一致；
  - ONNX 输入名是否为 obs_history；
  - 导出 metadata 中的 slice 是否为实际 Dog2 值。

  ## 最终判断

  针对 ATDog Dog2：

  ### 已经完成

  - HIM 算法模块接入；
  - 6 个 Dog2 任务的 HIM 注册；
  - policy/critic 分组观测适配；
  - 6 帧历史观测；
  - velocity slice 动态推导；
  - target slice 动态推导；
  - PPO、estimator、rollout、导出链路接通。

  ### 仍未证明

  - Dog2 实际 one-step observation 到底是 45 还是 46；
  - 所有地形子配置的 observation layout 完全一致；
  - Dog2 短训练能稳定反向传播；
  - estimator loss 与 HIMLoco 原论文数值一致；
  - HIMLoco checkpoint 可以有效迁移；
  - ONNX/JIT 在 Dog2 实际 history layout 下与仿真动作一致。

  ### 适配性评价

   项目                            Dog2 评价
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   代码结构适配                    良好
  ──────────────────────────────  ──────────────────────────────────
   观测维度适配                    基本正确，需运行时确认
  ──────────────────────────────  ──────────────────────────────────
   estimator 切片                  Dog2 当前设计合理
  ──────────────────────────────  ──────────────────────────────────
   历史观测顺序                    已处理，逻辑正确
  ──────────────────────────────  ──────────────────────────────────
   PPO 接口                        已接通
  ──────────────────────────────  ──────────────────────────────────
   任务注册                        完整
  ──────────────────────────────  ──────────────────────────────────
   原始 HIMLoco checkpoint 兼容    较差，维度大概率不同
  ──────────────────────────────  ──────────────────────────────────
   论文结果复现                    尚未证明
  ──────────────────────────────  ──────────────────────────────────
   多地形泛化                      配置完整但尚未验证
  ──────────────────────────────  ──────────────────────────────────
   实机部署                        导出链路有，实际动作一致性未验证

  总体上，Dog2 的移植可以评为“工程接入完成、算法验证未完成”。最优先的工作不是继续增加任务，而是运行一次 Dog2 rough 的最小 HIM 训练，
  并把实际打印出的 actor_obs / one_step_obs / critic_obs / vel_slice / target_slice 作为后续所有配置和 checkpoint 管理的基准。




这段话里有一个**关键的事实错误**，先把源码核实清楚再讲名词和坑。
先把**最关键的一个错误**纠正掉——这段话里有个数是猜的，而猜错了，会导致后面整套推算全错。

---

# 〇、核心纠正：`velocity_commands` 是 **3 维不是 4 维**

那段文字的推理链是：

> `heading_command=True` → `generated_commands` 可能包含 heading → 4 维 → one_step=46 → actor=276 → critic=49

**这条链是错的。** 源码证据：

```105:108:IsaacLab/source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py
    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b
```

`vel_command_b` 只有 3 列 `[lin_vel_x, lin_vel_y, ang_vel_z]`，而 `heading_target` 是**另一个独立 buffer**（第 83 行），**从不进观测**。

本仓的子类 `UniformThresholdVelocityCommand` 只覆盖了 `__init__` / `_update_metrics` / `_resample_command`，**没有覆盖 `command` 属性**，所以继承父类，就是 3 维。

**第二条独立佐证**——代码里这条注释：

```91:93:.../atdog/dog2/rough_env_cfg.py
        # 为了与历史 checkpoint 的网络输入维度保持一致，同时关闭 critic 侧高度扫描观测
        # 否则 resume 时会出现 critic 第一层权重 shape mismatch（例如 48 vs 235）
        self.observations.critic.height_scan = None
```

`48 vs 235` —— 写这条注释的人明确知道：**关掉 height_scan 后 critic 就是 48**（不是 49）。差 187 就是 height_scan 的维度。

## 所以 Dog2 的真实数字（已确定，不用再"验证"）

| 量 | 那段文字猜的 | **实际** |
|---|---|---|
| `velocity_commands` | 4 | **3** |
| one-step obs | 46 | **45** |
| actor 输入 | 276 | **270**（45×6） |
| critic obs | 49 | **48** |
| `estimator_vel_slice` | (0, 3) ✅ | **(0, 3)** |
| `estimator_target_slice` | (3, 49) | **(3, 48)** |

两个 slice 的**起点是对的，终点差 1**。这个差 1 在部署上是致命的（输入维度直接错）。

---

# 一、名词解释

## 1.1 算法层

| 名词 | 解释 |
|---|---|
| **HIM / HIMLoco** | 一种四足运动算法。核心思想：策略**看不到**真实速度，只能靠一个"估计器"从历史本体观测里**估**出速度和环境隐变量，再喂给 actor。这样训练出来的策略可以直接上真机（因为真机也没有真实速度）。 |
| **actor** | 策略网络，输出动作（12 个关节的目标位置偏移）。 |
| **critic** | 价值网络，评估"当前状态有多好"，只在训练时用，用来算优势函数、指导 actor 更新。**部署时不需要**。 |
| **privileged observation（特权观测）** | 训练时 critic 能看、但 actor 和真机都看不到的信息。这里就是 `base_lin_vel`（真实线速度）。这是 HIM 的关键设计——把"真值"隔离在 critic 里。 |
| **estimator（估计器）** | HIM 的核心模块。吃 6 帧历史（270 维），吐出两样东西：估计的 `base_lin_vel`（3 维）+ **latent**（隐变量）。 |
| **latent（隐变量）** | estimator 从历史里压缩出来的"环境/身体状态的隐含编码"（16 维，由 `estimator_encoder_hidden_dims=[128,64,16]` 的末层决定）。它编码了地面摩擦、负载、机身姿态趋势等无法直接从单帧观测看出的信息。 |
| **prototype / `estimator_num_prototypes`** | 一组可学习的"原型向量"（32 个）。estimator 用对比学习的方式把 latent 往最近的原型上靠（类似 VQ / 信息瓶颈），迫使 latent 变成**离散化、可泛化**的表征，而不是连续噪声。这就是 HIM 里 "I"（Information bottleneck）的来源。 |
| **encoder / target network** | estimator 内部两个网络：`encoder`（学生，吃历史）和 `target`（老师，吃 critic 的单帧特权观测）。训练时让学生输出去逼近老师的输出——这就是**"用特权信息蒸馏到历史编码器"**。 |

## 1.2 观测层

| 名词 | 解释 |
|---|---|
| **observation term（观测项）** | 一个最小的观测单元，比如 `base_ang_vel`（3 维）、`joint_pos`（12 维）。 |
| **observation group（观测组）** | 一组 term 的集合。这里有两个：`policy`（喂 actor）和 `critic`（喂 critic）。 |
| **observation manager** | Isaac Lab 里负责按配置拼接、加噪、缩放、维护历史的模块。 |
| **one-step observation** | **单帧**观测，不含历史。Dog2 = 45 维。 |
| **history observation** | 把连续多帧 one-step 拼起来。Dog2 = 45×6 = 270 维。 |
| **`num_one_step_obs`** | 单帧维度，45。`270 / 6 = 45`，estimator 靠这个把历史"切"回帧。 |
| **`history_length` / `history_size`** | 历史帧数，6。 |
| **`flatten_history_dim`** | 是否把 `(N, H, D)` 压平成 `(N, H*D)`。HIM 必须是 `True`（网络要吃 2 维输入）。 |
| **term-major vs frame-major（time-major）** | **历史在内存里怎么排**，这是最容易踩的坑：<br>• Isaac Lab 内部（term-major）：`[ang_vel×6帧, gravity×6帧, joint_pos×6帧, ...]`<br>• HIM 需要（frame-major / time-major）：`[第0帧全部term, 第-1帧全部term, ..., 第-5帧全部term]`<br>不重排的话 estimator 看到的是"同一个量的 6 个时刻"，**时间结构完全被破坏**。 |
| **`num_privileged_obs`** | 老式 legged_gym 接口里"critic 观测维度"的名字。这里 = 48。`HIMVecEnvWrapper` 专门做了适配，因为 Isaac Lab 的 `RslRlVecEnvWrapper` 不暴露这个属性。 |
| **corruption / noise / scale / clip** | 观测的四种后处理：`enable_corruption`（是否加噪，policy 开、critic 关）、`noise`（均匀噪声）、`scale`（缩放）、`clip`（截断）。 |

## 1.3 切片层

| 名词 | 解释 |
|---|---|
| **`estimator_vel_slice`** | `base_lin_vel` 在 **critic 观测**里的位置 `(start, end)`。这是 estimator 的**回归目标**（真值速度在哪）。Dog2 = `(0, 3)`。 |
| **`estimator_target_slice`** | critic 观测里**和 policy 共享的那一段连续区间**。这是 estimator 的 `target` 网络输入（老师的输入）。Dog2 = `(3, 48)`，正好 45 维 = one-step obs。 |
| **`group_obs_dim` / `group_obs_term_dim`** | observation manager 暴露的维度字典，前者是每组总维度，后者是**逐 term** 的维度。调试时打这个最准。 |

## 1.4 训练框架层

| 名词 | 解释 |
|---|---|
| **Hydra** | 配置管理框架。`--task` 和 `--agent` 就是 Hydra 的配置名。 |
| **entry point（配置入口点）** | `gym.register()` 里注册的字符串，指向一个配置类。如 `rsl_rl_him_cfg_entry_point`。 |
| **runner** | 训练主循环。`OnPolicyRunner`（普通 PPO）、`HIMOnPolicyRunner`（HIM）。 |
| **vec env wrapper** | 把 Gym 环境包装成 RL 库期望的接口。`HIMVecEnvWrapper` 额外负责：拆分 policy/critic 观测、**重排历史**、提供 `get_privileged_observations()`。 |
| **rollout storage** | 存放一整个 rollout（采样轨迹）的缓冲区：观测、动作、回报、价值、done 等。HIM 的 storage 额外要存 critic obs 和 next critic obs。 |
| **dones mask** | 回合结束标记。HIM 在更新 estimator 时会**把 done 样本 mask 掉**（因为回合边界的历史是断裂的，不能用来回归）。 |
| **checkpoint / resume** | 训练断点。`.pt` 文件，含 `model_state_dict` + `optimizer_state_dict` + `iter`。 |
| **JIT / ONNX / `obs_history`** | 部署用的模型格式。导出的 ONNX 输入名固定为 `obs_history`，维度 270。 |

## 1.5 损失函数层

| 名词 | 解释 |
|---|---|
| **`mean_value_loss`** | critic 的价值函数回归损失。 |
| **`mean_surrogate_loss`** | PPO 的策略损失（clipped surrogate objective）。 |
| **`mean_estimation_loss`** | **estimator 的速度回归损失**——估计的 `base_lin_vel` 和 critic 里真值速度的差距。**这个必须能降下来**，降不下来说明 slice 或历史布局错了。 |
| **`mean_swap_loss`** | **HIM 特有的对比/交换损失**——让 latent 和 prototype 对齐（信息瓶颈那部分）。出现 NaN 基本就是训练崩了。 |
| **`mean_entropy_loss`** | 策略熵相关项，控制探索。 |
| **adaptive KL / `desired_kl`** | PPO 的自适应学习率：根据"新旧策略的 KL 散度"和 `desired_kl=0.01` 的差距自动放大/缩小学习率。 |

## 1.6 控制与命令层

| 名词 | 解释 |
|---|---|
| **`heading_command`** | 是否用"朝某个绝对朝向走"代替"以某个角速度转"。开启后，`ω_z` 不再是采样值，而是由朝向误差 P 控制算出来的。 |
| **`rel_heading_envs`** | 有多大比例的环境使用 heading 模式。Dog2 rough = 0.5（一半一半）。 |
| **`rel_standing_envs`** | 多大比例的环境命令为 0（站着不动）。Dog2 rough = 0.2。 |
| **`resampling_time_range`** | 多久重新采样一次命令。Dog2 = (10, 10) 秒。 |
| **action scale** | 网络输出（约 [-1,1]）乘上这个系数，再加上 `default_pos`，才是真正的关节目标。**决定实际探索幅度**。 |
| **`use_default_offset`** | 动作是否是"相对默认关节角的偏移"。True 时，action=0 就是默认站姿。 |

---

# 二、坑详解（按危险程度排序）

## 坑 1 ⭐⭐⭐⭐⭐：把 heading 当成了第 4 维观测

**现象**：推算成 46 / 276 / 49。
**根因**：以为 `heading_command=True` 会把 heading 塞进观测。实际上 heading 只在**内部**参与计算 `ω_z`，从不作为独立维度输出。

```150:159:IsaacLab/.../velocity_command.py
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error, ...)
```

**后果**：如果按 46 去配实机的 `num_observations`，或者按 276 去导出/校验 ONNX，维度对不上，直接报错或静默错位。
**避坑**：记住 `command.shape == (N, 3)`，永远。

---

## 坑 2 ⭐⭐⭐⭐⭐：estimator 切片的"原版硬编码"陷阱

**这段最重要的洞见是对的，但成因要讲清楚。**

原始 HIMLoco 的 critic 布局是**把 `base_lin_vel` 放在末尾**：

```
原版 critic: [ang_vel(3), gravity(3), commands(3), dof_pos(12), dof_vel(12), actions(12), base_lin_vel(3)]
             → base_lin_vel 在 [45:48]  ✅ 原版的 (45,48) 是对的
```

而**本仓的 `CriticCfg` 把 `base_lin_vel` 放在最前面**：

```
本仓 critic: [base_lin_vel(3), ang_vel(3), gravity(3), commands(3), dof_pos(12), dof_vel(12), actions(12)]
             → base_lin_vel 在 [0:3]，而 [45:48] 现在是 actions 的最后 3 个！❌
```

**后果**（如果照搬原版 `(45,48)`）：estimator 会拿**上一帧的 3 个关节动作**当"真实线速度"去回归。训练出来的 latent 完全是垃圾，但**训练不会报错、loss 也会降**（因为动作确实可预测）——**这是最阴险的坑：静默错误**。

**好消息**：本仓已经改成从 observation manager 自动推导，你不会踩到。**但要确认自动推导生效了**——训练日志里应该打印实际的 `vel_slice`，务必核对是 `(0, 3)` 而不是 `(45, 48)`。

---

## 坑 3 ⭐⭐⭐⭐⭐：历史布局 term-major vs frame-major

**现象**：estimator 学不出来，`estimation_loss` 不降或降得很慢。
**根因**：Isaac Lab 的 `CircularBuffer` 是**每个 term 一个 buffer**，拼起来是 `[term1的6帧, term2的6帧, ...]`。而 HIM 的 encoder 期望 `frame0全部, frame-1全部, ...`。
**后果**：如果不重排，estimator 看到的第一帧是"6 个时刻的角速度"，第二帧是"6 个时刻的重力"——**时间维度被 term 维度污染**，encoder 学不到任何时序模式。
**避坑**：确认 `HIMVecEnvWrapper._reorder_policy_history()` 被调用，且顺序是 **当前帧在最前、最旧帧在最后**（latest-to-oldest）。

对应到部署侧（AT_robot-lab）：

```yaml
observations_history: [0, 1, 2, 3, 4, 5]   # 0=最新帧，5=最旧帧
observations_history_priority: "time"       # 按整帧拼接
```

这两行**必须和训练侧完全一致**，否则实机上历史顺序反了，策略直接失控。

---

## 坑 4 ⭐⭐⭐⭐：形状兼容 ≠ 语义兼容（这段漏掉的最危险的坑）

那段文字说"原 HIMLoco checkpoint 不能直接加载到 Dog2，因为 45 vs 46 维度不同"。

**纠正**：既然实际两边都是 **45 / 270 / 48**，那**形状是完全兼容的**！

这意味着：
- 原版 A1/Go1 的 HIM checkpoint **能成功加载**到 Dog2（形状全对）
- 但**语义完全不同**：term 顺序、scale、action scale、heading 语义、机器人动力学全不一样
- **结果：加载成功、能跑、但行为是错的** —— 比"加载失败"危险得多

**避坑**：`load_compatible_checkpoint()` 这种"按形状匹配、跳过不匹配参数"的兼容加载函数，会**静默跳过**关键层。用它加载别人的 checkpoint 前，必须确认 `policy_terms` 顺序和 `action_scale` 一致，否则宁可从头训。

---

## 坑 5 ⭐⭐⭐⭐：`target_slice` 推导的脆弱性

自动推导依赖四个假设（那段文字列得很准确）：term 名称一致、critic 中共享项**连续**、顺序相同、所有共享项都该进 target。

**爆点**：一旦有人在 critic 中间插入一个 term（比如在 `base_lin_vel` 后面加 `height_scan`），连续性被打破 → 推导返回 `None` 或只取到一段错误的区间。

**避坑**：
1. 训练日志里必须打印 `vel_slice` 和 `target_slice` 的**实际推导值**（目前只打印了部分布局，那段文字建议"把具体 term slice 一并输出"是对的）。
2. 更稳妥：显式声明 target term 名称列表，而不是靠"连续共享"推断。

---

## 坑 6 ⭐⭐⭐：各子环境 layout 必须一致，否则 checkpoint 全部作废

Dog2 六个任务都继承了 rough 的观测配置，都关了 `height_scan`，所以**目前都是 45/48，一致**。

**但是**：任何一个子环境一旦重新启用 `height_scan`：
- critic: 48 → **235**（+187）
- policy one-step: 45 → **232**
- actor 输入: 270 → **1392**

**后果**：该环境所有历史 checkpoint 全部不兼容，`resume` 时 critic 第一层权重直接 shape mismatch（就是注释里那个 "48 vs 235"）。

**避坑**：改动任何环境的 `observations` 之前，先确认是不是所有地形都要改；只改一个的话，务必换一个新的 `experiment_name`。

---

## 坑 7 ⭐⭐⭐：critic 缺失时静默回退

那段文字指出：wrapper 在拿不到 critic 时可能回退到 actor 观测。

**后果**：estimator 会拿"没有真值速度的观测"当监督目标 → 学出来的是恒等映射，`estimation_loss` 很低但完全没用。**能跑、loss 好看、结果是废的**。

**避坑**：改成**缺少 critic group 直接报错**。Dog2 有 critic，不会触发，但如果你要给新机器人接 HIM，这条会咬人。

---

## 坑 8 ⭐⭐⭐：observation normalization 关闭 + 量纲混杂

Dog2 是 `actor_obs_normalization=False`、`critic_obs_normalization=False`。

各 term 的原始量纲：

| term | 量级 |
|---|---|
| `projected_gravity` | [-1, 1] |
| `velocity_commands` | [-1, 1] |
| `base_ang_vel` | 约 [-3, 3]，**scale=0.25** → [-0.75, 0.75] |
| `joint_vel` | 约 [-20, 20]，**scale=0.05** → [-1, 1] |
| `joint_pos` | 相对默认角，约 [-2, 2] |
| `actions` | [-1, 1] × scale |

**后果**：estimator 的 encoder 第一层直接吃混合尺度输入，又没有 running normalization 兜底 → 对 domain randomization（尤其是摩擦、质量随机化）**特别敏感**，容易出现训练后期突然崩。

**避坑**：要么打开 `actor_obs_normalization`（推荐先试），要么仔细核对每个 term 的 `scale` 让量级接近。至少要保证 `policy` 和 `critic` 的**同名 term 的 scale 一致**，否则 target 网络和 encoder 看到的分布不一样。

---

## 坑 9 ⭐⭐：action scale 与探索噪声耦合

Dog2：hip `0.125`，其他 `0.25`。

网络输出 `a ~ N(0, init_noise_std=1.0)`，实际关节目标偏移 = `a × scale`。

**后果**：hip 关节的实际探索幅度只有小腿的一半 → **hip 学得慢**。而 `init_noise_std=1.0` 是作用在**归一化动作**上的，你调 `init_noise_std` 时，实际影响被 scale 放大/缩小了。

**避坑**：调探索时记住：`实际探索幅度 = init_noise_std × action_scale`。想统一探索强度，要么用统一 scale，要么按关节分别设 noise。

---

## 坑 10 ⭐⭐：heading 模式带来的命令分布差异（语义坑）

这是第 1 个坑的"语义版本"，同样危险：

- 训练时 50% 的 env，`ω_z = clip(0.5 × wrap_to_pi(heading_target − heading_w), ±1)`
- 这个值是**随时间变化并收敛到 0** 的（机器人转到目标朝向就不转了）
- 另外 50% 的 env，`ω_z` 是从 `(-1, 1)` 直接采样的**固定值**

**后果**：如果实机是"摇杆直接给 yaw rate"，策略会见到训练时很少见的"长期非零 yaw 命令"，转向行为可能异常。

**避坑**：训练和部署必须同构。要么实机也用 heading P 控制（同样的 `stiffness=0.5` 和 `±1` clip），要么训练时把 `heading_command=False`。

---

## 坑 11 ⭐⭐：动作延迟与历史窗口的关系

`DelayedPDActuatorCfg(min_delay=1, max_delay=2)`，物理步长 `dt=0.005s` → 延迟 5~10ms。
历史 6 帧 @ 0.005s → **30ms 观测窗口**。

**注意**：历史窗口（30ms）要**显著大于**动作延迟（5~10ms），否则历史里全是"被延迟污染"的信息，estimator 学不到干净的动态。目前 30ms vs 10ms 是够的，但如果你把 `history_length` 调小到 2（10ms），就会踩到。

---

# 三、给你的实操核对清单

跑起来后，在日志里**逐项确认**（这几项对了，HIM 就基本没跑偏）：

```text
policy  term 顺序:  base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, actions
critic  term 顺序:  base_lin_vel, base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, actions

one_step_obs      = 45        ← 不是 46
history_length    = 6
actor_obs         = 270       ← 不是 276
critic_obs        = 48        ← 不是 49
vel_slice         = (0, 3)    ← 绝不能是 (45, 48)
target_slice      = (3, 48)   ← 绝不能是 (3, 49)
历史顺序          = [当前帧, ..., 最旧帧]
```

然后短训练（`--max_iterations 2`）确认：
1. `mean_estimation_loss` **有数值且在下降**（不是 NaN、不是恒 0）
2. `mean_swap_loss` 不 NaN
3. `action std` 没有一轮内爆炸
4. checkpoint 能存下来

**其中最要盯的是 `estimation_loss`** —— 它是 HIM 唯一的"我知道 estimator 有没有学对"的信号。它不降 = 坑 2 或坑 3 中招了。

需要我帮你写一个打印上述 layout 的诊断脚本，或者在 `HIMOnPolicyRunner` 里补上 term 级 slice 的日志输出吗？
Isaac Lab原始270
        │
        │ 重排
        ▼
[t, t-1, t-2, t-3, t-4, t-5]
每帧45维
        │
        ▼
   HIMEstimator
        │
   ┌────┴────┐
   ↓         ↓
3维速度     16维latent
   └────┬────┘
        ↓
当前45维 + 3 + 16
        ↓
       64维
        ↓
      Actor
        ↓
      12维
     action















                    ┌──────────────────────────────┐
                    │        Isaac Sim / PhysX     │
                    │   机器人 + 地形 + 接触 + 动力学 │
                    └──────────────┬───────────────┘
                                   │
                                   │ 产生机器人状态
                                   ▼
                    ┌──────────────────────────────┐
                    │       Isaac Lab Environment   │
                    │    ManagerBasedRLEnv          │
                    │                              │
                    │ Scene                         │
                    │ Commands                      │
                    │ Observations                  │
                    │ Actions                      │
                    │ Rewards                      │
                    │ Events / Randomization       │
                    │ Terminations                 │
                    │ Curriculum                   │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │      RslRlVecEnvWrapper       │
                    │          ↓                     │
                    │       HIMVecEnvWrapper        │
                    │                              │
                    │ policy obs  →  历史重排        │
                    │ critic obs  →  privileged obs │
                    └──────────────┬───────────────┘
                                   │
                     ┌─────────────┴─────────────┐
                     ▼                           ▼
              Policy History                 Critic Obs
                 270                        48
                     │                           │
                     │                           ▼
                     │                       Critic
                     │                       48 → 1
                     │
                     ▼
                HIM Estimator
                  270
                   │
            ┌──────┴──────┐
            ▼             ▼
       velocity 3      latent 16
            │             │
            └──────┬──────┘
                   ▼
        当前帧 obs 45 + 3 + 16
                   │
                   ▼
                 64
                   │
                   ▼
                Actor
                 64
                   │
                   ▼
                 12
                actions
                   │
                   ▼
        JointPositionAction
                   │
            scale + offset
                   │
                   ▼
           joint position target
                   │
                   ▼
          Delayed PD Actuator
                   │
                   ▼
             12 个关节
                   │
                   ▼
                机器人
