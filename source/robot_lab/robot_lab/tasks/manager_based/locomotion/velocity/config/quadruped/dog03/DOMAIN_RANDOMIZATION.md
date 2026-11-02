# 03dog 282 维：电机辨识参数与域随机化复核

## 配置入口

- 平地域随机化 `RobotLab-Isaac-Velocity-Flat-03Dog-282-DomainRand-v0`：`cyberstar/config.json` 的 `env`。
- 平地预训练 `RobotLab-Isaac-Velocity-Flat-03Dog-282-v0`：同一份 JSON，但 `randomize_*` 和 `push_robots` 被关闭。
- 楼梯 `RobotLab-Isaac-Velocity-Stairs-03Dog-282-v0`：`cyberstar/stairs_config.json` 的 `env`。
- 三者共用电机实现；Actor 282 维、Critic 491 维及历史顺序保持不变。
- 参数在进程启动时读取，修改后须重新启动。以下配置不作用于非 282 的普通 Flat/Rough 任务。

## PD 增益与机械摩擦分开填写

`env.control.stiffness` 是 Kp，`env.control.damping` 是控制器 Kd，默认仍为 30 / 1。
现在二者由 `cyberstar/runtime.py` 读取，不再在 `CyberstarJointAction` 中写死。
增益可用 `{"joint": 30.0}` 统一指定，也可加完整匹配的正则或精确关节名：

```json
{
  "joint": 30.0,
  ".*_thigh_joint": 32.0,
  "FL_hip_joint": 28.0
}
```

精确名称优先于正则，正则优先于 `joint` 默认值；同一关节匹配多个非精确正则时报错。
未匹配任何关节的键、缺少关节值、负值和非有限数均被拒绝。
`env.domain_rand.kp_scale_range`、`kd_scale_range` 只随机缩放对应的 PD 增益。
`env.control.action_scale` 和 `decimation` 同样接入实际动作配置。

辨识的机械摩擦写入 `env.control.motor_friction`，不是 `damping`，也不是地面接触的 `friction_range`：

- `coulomb.<关节名>`：库仑摩擦力矩 a，单位 N·m。
- `viscous.<关节名>`：黏性摩擦系数 b，单位 N·m·s/rad。
- 两个字典均已列出 FL/FR/RL/RR 的 hip/thigh/calf 共 12 个关节，全部初值为 0。
- 使用减速器输出关节侧的参数与关节速度；不要直接填入未经换算的电机轴侧参数。
- 可将字典替换为 `{"joint": 0.0}`，再按需要添加正则或精确名称覆盖。

每个物理步的计算顺序：

```text
tau_PD = Kp * kp_scale * (q_target + zero_offset - q) - Kd * kd_scale * dq
tau_drive = clip(motor_strength * tau_PD, -motor_limit, motor_limit)
tau_friction = a * sign(dq) + b * dq
tau_joint = tau_drive - tau_friction
```

`cyberstar/actuator.py` 在驱动力矩限幅后扣除被动摩擦；摩擦不会乘上 PD 随机倍率或电机强度，
也不会再次按电机输出上限截断。驱动上限读取执行器的 `effort_limit`，不使用为显式执行器放宽的
PhysX 求解器力矩上限。`runtime.torques` 继续记录驱动力矩，执行器 `applied_effort` 记录含摩擦的净力矩。
PhysX 原生关节摩擦显式置零，避免与自定义模型重复计算。

这是库仑 + 黏性模型：`dq = 0` 时摩擦为零，不包含静摩擦保持、Stribeck、温度或齿隙模型。
ATDog 的原生 `friction` 是 PhysX 关节摩擦参数，不应不经验证就当成辨识所得的 N·m 力矩系数。

## 摩擦随机化：占位值全为零

两份 JSON 的 `env.domain_rand` 均包含：

```json
{
  "randomize_joint_friction": true,
  "joint_coulomb_friction_offset_range": [0.0, 0.0],
  "joint_viscous_friction_offset_range": [0.0, 0.0]
}
```

训练重置时，对每个环境、每个关节独立采样均匀分布偏移：

```text
a_actual = max(0, a_nominal + uniform(coulomb_offset_range))
b_actual = max(0, b_nominal + uniform(viscous_offset_range))
```

这两个范围是**加法偏移**，不是倍率或绝对系数。初始零范围表示不增加不确定性；
修改 nominal 后，即使不改范围，也会使用填写的固定摩擦。关闭随机化或回放时，仍保留 nominal，
不会把已辨识的机械摩擦清零。只更新正在重置的环境，其他环境的参数保持不变。

## 从当前 ATDog 源码逐项复核

复核范围是当前仓库 `assets/atdog.py`、ATDog 四足/轮足及带机械臂的全部环境配置、其共同父类
`velocity_env_cfg.py`、`mdp/events.py`，以及实际 Isaac Lab 执行器实现。
不把注释掉的事件当成启用项，也不把固定资产属性当成随机事件。

| 当前 ATDog 生效项 | 03dog 282 对应配置 / 实现 | 时机 |
| --- | --- | --- |
| 接触静摩擦、动摩擦、恢复系数，逐碰撞形状分配材质 | `randomize_friction` / `friction_range`、`randomize_dynamic_friction`、`randomize_restitution`、`randomize_material_per_shape` | Startup |
| 机身附加质量 | `randomize_base_mass`、`added_mass_range` | Startup |
| 非机身连杆质量缩放 | `randomize_link_mass`、`link_mass_scale_range`；排除机身，避免叠加机身附加质量随机化 | Startup |
| 质量变化后同步惯量，ATDog 的 `recompute_inertia=True` | 新接入 `recompute_inertia`：按质量比缩放完整惯量张量；之后再叠加原有独立惯量随机倍率 | Startup |
| 机身 COM | `randomize_com`、`com_range_x/y/z`；03dog 另有连杆 COM 随机化 | Startup |
| 机身外力、力矩 | `randomize_external_force_torque`、`external_force_range`、`external_torque_range`，重置采样后持续施加 | Reset |
| 关节初始位置、初始速度 | `init_state.reset_dof_scale_range`；新接入 `reset_dof_velocity_range`，初值 `[0, 0]` | Reset |
| PD 刚度 / 阻尼倍率 | `randomize_pd_gains`、`kp_scale_range`、`kd_scale_range`；基值来自 JSON | Reset |
| 初始根位置 / 姿态 / 速度 | 原有 XY 位置采样、`randomize_reset_pose`、`reset_pose_range`、`init_state.reset_root_velocity_range` | Reset |
| 定时随机速度推扰 | 原有推扰模式、幅度和安全门；新接入 `randomize_push_interval`、`push_interval_offset_range_s` | Interval |
| 资产中的关节摩擦 | 新接入独立 Coulomb / viscous 力矩模型、每关节零占位和随机偏移 | 固定模型 + Reset |
| `DelayedPDActuator` 的随机延迟 | 原有 `randomize_control_delay` / `control_delay_steps`；缓冲区现按最大延迟分配 | Reset |
| Actor 观测噪声 | 原有 `noise.add_noise` / `noise_scales`，覆盖角速度、重力、关节位置和关节速度 | 观测采样 |

ATDog 共同父类当前有 9 个启用事件，表中前 10 行将接触材质和重置状态展开列出。
其中关节初速度的 ATDog 默认范围本身也是零；此前 03dog 将其硬编码为零，现在提供可填写的入口。
ATDog 资产当前有固定关节摩擦，但没有独立的电机摩擦参数随机事件；本次给 03dog 同时补上模型和随机化入口。
ATDog 资产注释把 `damping` 称为“黏性摩擦”，但 `DelayedPDActuator` 的实现把它用于 PD 速度误差反馈，不能据此替代机械摩擦模型。
ATDog 的独立惯量事件是注释掉的；实际联动来自质量事件的 `recompute_inertia=True`。

### 材质随机化的实际覆盖

ATDog 调用的 Isaac Lab `randomize_rigid_body_material` 是从 64 个完整材质三元组中，
为每个环境的每个碰撞形状独立分配一个材质。03dog 此前虽已有三个系数的随机开关，
但同一环境内所有形状共用一组参数，缺少这种接触差异。

- 两份 JSON 新增 `domain_rand.randomize_material_per_shape=true`，补齐逐形状采样；
  Flat-282 随其他 `randomize_*` 开关自动关闭。
- 三个系数先组成共用的 64 个材质桶，再分配给碰撞形状；不是将三套 64 个系数任意组合，
  避免并行环境和碰撞形状增多时产生过量不同材质。
- 静摩擦、动摩擦和恢复系数仍由各自开关控制；逐形状开关本身不会启用任何系数随机化。
  仅关闭逐形状开关，可恢复每个环境共用材质的模式。
- `critic_friction` 使用该环境各碰撞形状静摩擦的均值，仍占原来的一维，不改变 491 维 critic。
- 地面继承的 `restitution_combine_mode="multiply"` 此前与地面恢复系数 0 组合，
  导致机器人采样的恢复系数全部失效。现在地面恢复系数设为乘法中性值 1，
  实际机器人—地面恢复系数由机器人采样值决定；关闭随机化时机器人系数仍为 0。

### 刻意保留的差异

- 不照搬不同机型的质量、COM、增益和推扰强度；03dog 已有数值保持原样，新增摩擦与速度占位均为零。
- `push_interval_s` 保留基础周期，偏移范围默认 `[0, 5]`：鲁棒平地为 10–15 s，楼梯为 15–20 s。
  每个环境的定时器在重置及到期后独立重采样；填写 `[0, 0]` 或关闭 `randomize_push_interval` 可恢复固定周期。
  平地已有安全门和训练课程仍可跳过某次推扰；楼梯仍使用原有平面速度推扰。
- `control_delay_steps` 仍按**策略控制步**计，不是 ATDog 的物理步；没有悄悄改变已有检查点的时序语义。
- 楼梯保留已有 XY 放置与随机 yaw/航向参考，不额外叠加 yaw；平地仍使用原 XY 采样。
- 282 actor 没有线速度与高度扫描通道；不为复制 ATDog 的这些噪声配置增加观测维度，也不污染干净的特权 critic。
- ATDog 部分 URDF 有被动 foot 关节；03dog 使用固定 FOOT 刚体，不凭空增加被动自由度或其阻尼。
- `armature`、电机温度、齿隙、Stribeck、重力随机化等没有在本次 ATDog 的启用随机事件中发现，不虚构为已存在功能。

## 逐电机物理步延迟复核

ATDog 的 `DelayedPDActuatorCfg(min_delay=1, max_delay=2)` 不属于 `EventCfg`，
但执行器会在每次重置时重采样延迟。只统计上述 9 个事件会漏掉这一项。
03dog 原来的 `control_delay_steps=[0, 1]` 是整条动作向量共用的策略步延迟，
既没有 5–10 ms 的物理步粒度，也没有逐电机延迟差异，不能视为完整等价实现。

两份鲁棒配置现增加：

```json
{
  "randomize_motor_delay": true,
  "motor_delay_steps": [1, 2]
}
```

- 在 `runtime.compute_motor_torques()` 中缓存位置目标，每个物理步推进一次；
  每个环境、每个关节独立采样，参数在一个回合内保持不变。
- 只延迟目标，不延迟关节位置、速度反馈，也不直接延迟已算好的力矩。
  原有增益、零位偏移、强度缩放、限幅和被动摩擦顺序保持不变。
- 重置只清理所选环境的电机历史并重采样延迟；不足的历史使用本回合首个有效目标，
  不会将上一回合目标或绝对零角度作为初始电机命令。
- 平地预训练、回放和 `env.test=true` 都关闭该随机化；旧配置缺少这两个键时默认关闭。
- 原有策略步延迟不改名、不改单位，也不关闭。默认 `sim.dt=0.005`、`decimation=4` 时，
  新电机延迟为 5/10 ms，与原有 0/20 ms 动作延迟串联，总延迟为 5/10/25/30 ms。
  若只需要 ATDog 式执行器延迟，可显式关闭 `randomize_control_delay`，不要把两者误当同一开关。
- Actor 282 维、Critic 491 维和历史顺序不变；critic 原有 `delay` 槽仍表示策略步延迟，
  不将逐电机延迟硬塞入该槽，也不改变检查点张量尺寸。输入尺寸兼容不代表旧策略已经适应该新分布。

## 本次额外修正

- 原有 `randomize_link_mass` 会再次缩放机身，使机身实际质量偏移超出 `added_mass_range`，也让 critic 的质量归一化范围失真。
  现在连杆质量和连杆 COM 均排除 `base_link`；机身只由各自独立开关控制。
- 动摩擦采样后取 `min(dynamic, static)`，避免默认随机区间重叠时产生动摩擦大于静摩擦的材质。
  对不相容的自定义区间，也以静摩擦作为动摩擦上限。
- 重置关节位置与速度在写入仿真前按资产软限位裁剪，避免扩大角度偏移或速度范围后直接写入越界状态。
  正常默认范围不受影响，局部重置仍只作用于所选环境。
- `cyberstar_play=true` 在环境内同时关闭观测噪声与定时推扰，不再只依赖 RSL-RL 回放脚本修改配置。
  通过其他入口直接构造回放环境时也生效；原有 XY 放置和楼梯航向采样仍保留，不承诺全确定性回放。
- 保留 JSON 的 `env.test=true`，不再被默认的 `cyberstar_play=false` 覆盖；任一开关启用都进入测试模式。
- 回放原先从 `_randomize_startup()` 提前返回，跳过 03dog 的标称接触材质初始化，
  导致导入资产的摩擦/恢复系数可能与无随机化训练不同，且 critic 仍错误报告摩擦为 1。
  现在测试模式不采样质量、COM、惯量及材质随机系数，但仍写入标称材质 `(1, 1, 0)` 并同步 critic。
- 推扰偏移旧默认值 `[0, 0]` 虽有开关却没有实际周期变化，现默认启用非退化区间；固定周期按控制步向上取整且至少为一步。
- 鲁棒平地推扰安全门改用局部地面相对高度，而非世界 Z，避免高地形上的正常站姿被误判。
  只对本次到期的环境查询地形，不额外扫描所有并行环境。
- JSON 中 `sim.dt`、`control.decimation`、`control.action_scale`、`env.episode_length_s` 同步到仿真配置；
  接触传感器、地形扫描和渲染周期随之联动，不再残留硬编码的 0.02 s 扫描周期。
- 楼梯 `stair_clearance_progress_floor=0.35` 原先只写在 JSON 中，净空奖励直接乘以速度进度，受阻静止时仍为零。
  现在有运动指令但受阻时保留配置下限的抬脚奖励，正常前进获得更高奖励，零指令依然为零。
  此修正仅作用于 03dog，不修改 CSDog1 的既有实现。

这些修正会改变鲁棒训练的采样分布，但不改网络输入、关节顺序或已有检查点的张量尺寸。
仍应从已验证的平地检查点继续训练并重新评估，不能把 CPU 回归通过等同于实机或物理仿真验证。

## 回归验证

`tests/test_dog03_motor_dynamics.py` 验证 JSON 增益、关节匹配、限幅与摩擦顺序、零默认值、
随机范围、回放/关闭开关、局部重置关节顺序、质量-惯量联动、长延迟和随机推扰定时器，
并覆盖机身/连杆隔离、静/动摩擦一致性、逐形状材质差异、完整材质桶数量、地面恢复系数、
重置软限位、环境内回放开关、地形相对推扰安全门、JSON 仿真时序和楼梯受阻净空奖励。
其中 ATDog 四足/轮足/带臂配置的事件清单直接从当前 AST 收集；新增未覆盖的事件或覆盖键缺失会使测试失败。
原有 03dog 测试继续检查 282 / 491 维观测与 HIM 检查点契约。
新增回归还覆盖 ATDog 资产中的执行器延迟清单、逐电机采样、物理步推进、当前 PD 反馈、
首目标填充、局部重置、关闭/回放/旧配置兼容、非法范围，以及测试模式的标称材质和 JSON 测试开关。

本轮 55 项 03dog CPU 回归全部通过，包含新增的 7 项检查和完整尺寸的 HIM/PPO 更新。
全仓 CPU 测试共 82 项，81 项通过；唯一失败是 CSDog1 的
`test_blocked_clearance_has_signal_but_zero_command_does_not`。
另从修改前的 Git HEAD 导出 CSDog1 测试及依赖单独运行，同一断言仍失败，确认是既有问题；本轮未修改 CSDog1。

另在 Isaac Sim 的 headless GPU / PhysX 仿真中，对 Flat-282、DomainRand、楼梯和
DomainRand 的 JSON 测试模式分别使用 4 个环境，每组运行 33 个控制步并插入局部重置，均通过。
核对实际材质、静/动摩擦关系、64 桶上限、critic 摩擦均值、观测/奖励有限性，
以及电机延迟 `[1, 2]`、物理步历史推进、局部重置隔离和测试模式延迟归零。
短步验证不代表策略已收敛或通过实机验证。
