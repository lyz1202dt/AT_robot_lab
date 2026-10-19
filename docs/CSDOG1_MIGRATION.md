# CSDog1 flat-282 DomainRand 与台阶任务迁移

`RobotLab-Isaac-Velocity-Flat-CSDog1-282-DomainRand-v0` 现在运行 Cyberstar 的完整 flat-282 robustness 任务。
`RobotLab-Isaac-Velocity-Stairs-CSDog1-282-v0` 复用这一运行时，并增加参考工程的台阶指令、地形和奖励。
默认入口为 RSL-RL HIM；原有 flat47、flat282、rough 保留原来的任务逻辑。

## 迁移内容

- 按参考工程继承后的实际配置冻结所有参数，包括置零的历史实验项，见 `cyberstar/config.json`。
- 原样保留张量奖励公式、结构化指令、停止调整步、航向保持、落脚软着地和 capture recovery 状态机。
- 78 个非终止奖励项，另有 termination；奖励合成为 Ji22：`positive * exp(clamp(negative / 0.02, -20, 0)) + termination`。正负分量先乘策略步长。
- Actor 为 6×47 的完整帧历史，旧到新，reset 用首个有效观测填满历史。
- Critic 为 491 维：clean history282、真实线速度3、足端竖向力4、摩擦1、电机强度1、控制延迟1、停车状态2、停车航向误差1、停车进入方向2、地形187、局部高度1、附加质量1、COM3、平均 Kp/Kd2。
- 50 Hz 策略、200 Hz 仿真；动作延迟按策略步采样 0/1，显式 PD 每个物理步计算，电机强度在力矩限幅前生效。
- 接触 offset=0.01 m、rest offset=0、bounce threshold=0.5 m/s，TGS 4 次位置迭代/0 次速度迭代；训练 seed=1。
- 85% 平地与 15% 低离散障碍，障碍为 ±1/±2 cm，20 个 1–2 m 矩形，中心 3 m 平台。
- 摩擦 `[0.6,1.3]`、质量 ±0.25 kg、COM ±5/±5/±3 mm、电机强度与 PD 倍率 `[0.95,1.05]`。
- 推扰每 10 s；平面 0.8→1.0 m/s（iteration1000–1200），capture 激活课程1200–1600，角扰动1500–1800。安全状态门控与互斥推扰模式来自参考工程。
- 基座接触、roll>0.8 rad 或 pitch>1 rad 终止；20 s 超时。

`core.py` 是不依赖 Isaac Gym 的参考张量代码；`lab.py` 是仿真适配边界。
参考文件 SHA256 记录在 `cyberstar/provenance.json`，该目录保留 Cyberstar BSD-3-Clause 许可。
运行时不依赖原参考目录，也不导入 `isaacgym`。

## 模型与接触

参考目录的 URDF 路径失效，但其 `resources/robots/csdog1/mjcf/csdog1_simple.xml` 存在。
当前项目 MJCF 的 13 个刚体、12 个关节和 13 组惯量与它一致。
现有 USD 把足端碰撞球合并进 calf，不能直接区分正常脚接触和 calf 碰撞。
任务在场景实例化时恢复四个固定 FOOT 刚体，把原碰撞球转交给 FOOT；从 calf 转移 1 g 质量并用平行轴定理分解惯量，保持整条腿的总质量、质心和惯量。
磁盘中的共享 USD 不修改，其他任务不受该适配影响。

Isaac Gym 和 Isaac Sim 的接触求解器不同；迁移保证任务公式及控制语义，不承诺相同随机种子下轨迹逐位一致。
Critic 地形高度在 Lab 中用静态 mesh raycast 查询，替代源侧 heightfield 数组查询。

## 训练和检查点

推荐从原工程的 flat282 `model_1000.pt` 接续：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Flat-CSDog1-282-DomainRand-v0 \
  --resume --checkpoint /absolute/path/to/model_1000.pt --headless
```

默认使用 `logs/rsl_rl/csdog1_flat_282/flat_282/model_1000.pt`，若不存在会明确报错。
训练默认再运行 1000 iterations，学习率 `5e-5`、entropy `0.003`、fixed schedule。
默认不加载优化器，保留检查点 iteration，使推扰/恢复课程从原进度继续。
原工程 `history_encoder/target_encoder/prototypes` 参数自动映射到本项目 HIM。
297 维 Critic 的旧权重列保留，新增194列初始化为零；已迁移的491维检查点也可加载。
缺失 Actor/estimator 权重或不兼容的 Critic 会报错，禁止静默部分加载。
当前项目旧版 50 维 Critic 检查点不满足此迁移合同，不能作为等价源模型使用。

只验证环境或从头训练时可显式关闭续训：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Flat-CSDog1-282-DomainRand-v0 \
  --headless agent.resume=false
```

从头训练不是原工程“先 flat282、再 DomainRand”训练流程的等价替代；完整课程仍推荐使用 flat282 权重。
恢复已迁移任务并需要延续 Adam 时，额外设置 `agent.load_optimizer=true`。
回放支持 `--keyboard`，输入同时进入策略观测、停车状态和奖励门控；回放关闭噪声、推扰和动力学随机化。
此任务使用标准 `play.py` 入口，例如：

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-Flat-CSDog1-282-DomainRand-v0 \
  --checkpoint /absolute/path/to/model.pt --num_envs 1 --keyboard
```

增加 `--export_only --headless` 可仅导出 JIT、ONNX 和部署元数据。

## 部署约定

此任务采用参考训练约定：每条腿默认角 `[0,0.785,-0.823]`，指令缩放 `[2,2,0.25]`，角速度缩放0.25，关节速度缩放0.05，动作缩放0.25，PD30/1。
不能直接沿用此前 stairs 部署配置中的 calf=-0.623、平移缩放1。
导出的 `policy_metadata.json` 中 `cyberstar_contract` 包含关节顺序、默认姿态、缩放、策略周期与历史初始化规则。
停止相位依赖接触和测量状态的调整步；部署需同步该状态机。固定“松手后0.5 s清零”不能完整复现训练中的停车相位。

## 验证

```bash
python3 -m unittest discover -s tests -p 'test_csdog1_*.py'
python scripts/tools/check_csdog1_migration.py --headless --num_envs 16 --steps 600
```

CPU 的 14 项检查覆盖历史reset、491维布局、观测限幅、20 ms延迟、capture动作限幅、停车状态、78项奖励与Ji22、推扰课程、严格检查点转换。
仿真脚本检查物理rollout、独立足端接触、拆分前后的质量/质心/惯量守恒、源格式检查点加载前后策略输出一致、两轮HIM PPO、JIT/ONNX与元数据导出。
验证产物写入 `outputs/csdog1_migration/`；其中 `source_format_random_fixture.pt` 和 `smoke_model_1802.pt` 是随机初始化的测试权重，不能当作训练好的步态。
标准 `train.py` 的 Hydra 参数解析、源格式续训（1000→1002）、配置/模型保存及 `play.py --export_only` 也已实测。
当前工程和参考目录中未找到训练好的源 `model_1000.pt`；需要自行指定该权重，才能进行与原流程一致的 DomainRand 微调。
短程检查验证可执行性和有限数值，不作为步态收敛或实机效果证明。

## 台阶训练

台阶任务保留相同的 282 维 Actor 和 491 维 Critic，包括观测顺序、缩放、默认站姿及动作延迟。
`cyberstar/stairs_config.json` 冻结参考工程当前第 4 阶段配置：3–20 cm 台阶，30% 平地回放；台阶部分为 65% 上行、35% 下行。
256 个地形块均匀复用；台阶指令为 50% 前进、25% 左移、25% 右移，前进与侧移互斥。
85 个非终止奖励项包含落脚冲击、落脚速度、支撑和前进进度；下台阶加强落脚惩罚。
默认从鲁棒平地的 `model_2000.pt` 开始，再训练 1200 iterations，固定学习率 `5e-5`，entropy `0.003`，不恢复 Adam 状态。
已训练好的 491 维台阶 checkpoint 可以通过 `--checkpoint` 显式指定：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Stairs-CSDog1-282-v0 --headless \
  --resume --checkpoint /absolute/path/to/model_10000.pt
```

台阶配置不启用平地的高级推扰课程，使用每 15 s、平面速度 ±0.5 m/s 的基础推扰。
基础推扰通过 Lab 的速度写入接口实现，避免缺失父类 `_push_robots()`；刚 reset 的环境不立即施加推扰。
台阶任务的 PhysX `gpu_collision_stack_size` 至少为 `134217728`（128 MiB），用于容纳大量机器人与台阶网格的碰撞。
若日志仍提示 `collisionStackSize buffer overflow`，应按实际需求继续增加该容量或减少 `--num_envs`；接触已被丢弃时不应忽略错误继续训练。

CPU 检查现有 23 项，覆盖基础推扰、台阶互斥指令、491 维权重迁移、Adam 状态兼容性及实际 HIM PPO 更新。
台阶仿真检查：

```bash
python scripts/tools/check_csdog1_migration.py \
  --task RobotLab-Isaac-Velocity-Stairs-CSDog1-282-v0 \
  --headless --num_envs 16 --steps 120 --output_dir outputs/csdog1_stairs_check
```
