# AT_robot_lab 项目逻辑导读

本文介绍 Isaac Lab 速度跟踪任务的代码结构，以 Go2 标准任务说明环境配置与训练入口。ATDog2 和 03dog 的实际训练命令见 [AT_DESCRIPE.md](../AT_DESCRIPE.md)。

## 项目结构

| 层次 | 作用 | 位置 |
| --- | --- | --- |
| Isaac Sim / PhysX | 场景、碰撞和刚体动力学 | 外部仿真平台 |
| Isaac Lab | 并行环境、资产、传感器、Manager | 外部 `isaaclab` 包 |
| Robot Lab | 机器人、地形、观测、动作、奖励、随机化 | `source/robot_lab/robot_lab/` |
| RSL-RL / HIM | 采样、网络更新、检查点和导出 | 外部 RSL-RL 与本地 `robot_lab/him/` |

```text
scripts/reinforcement_learning/rsl_rl/
  train.py                   # 训练入口
  play.py                    # 回放与导出入口
  cli_args.py                # RSL-RL 命令行参数
source/robot_lab/
  data/                      # 模型资源
  robot_lab/
    assets/                  # 机器人资产配置
    terrains/                # 自定义地形
    him/                     # HIM 网络、训练和导出
    tasks/
      manager_based/locomotion/velocity/
        velocity_env_cfg.py  # 通用速度跟踪环境
        mdp/                 # 观测、奖励、事件和指令函数
        config/quadruped/
          unitree_go2/       # Go2 环境与算法配置
          atdog/dog2/        # ATDog2 环境与算法配置
          dog03/             # 03dog 环境与算法配置
logs/                        # 检查点和训练日志
```

`assets/*.py` 定义机器人加载、关节初始状态和执行器；几何资源位于 `source/robot_lab/data/`。任务目录的 `__init__.py` 用 `gym.register()` 把 task ID 绑定到环境配置和算法配置。

## ManagerBased 环境

Go2 和 ATDog2 使用 `ManagerBasedRLEnv`，由配置项组合任务：

| 配置项 | 作用 |
| --- | --- |
| `scene` | 地形、机器人、传感器 |
| `commands` | 期望的前进、侧移和转向速度 |
| `observations` | Actor/Critic 输入 |
| `actions` | 将网络输出转为关节目标 |
| `rewards` | 速度跟踪、稳定性、能耗等评分 |
| `terminations` | 失败与超时判定 |
| `events` | 启动、重置和定时随机化 |
| `curriculum` | 难度递进 |

`ObsTerm(func=..., scale=...)` 指定观测，`RewardTermCfg(func=..., weight=...)` 指定奖励。`@configclass` 子类通常在 `super().__post_init__()` 后覆盖配置，因此阅读实际值时应沿继承链检查。

## 从命令到训练

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --num_envs=2048 --headless
```

1. `train.py` 解析参数并启动 Isaac Sim。
2. `robot_lab.tasks` 导入任务目录，注册 Gym 环境。
3. `@hydra_task_config` 从任务 ID 取环境和算法配置。
4. `gym.make()` 创建场景、机器人和各 Manager。
5. PPO 或 HIM 的 VecEnvWrapper 对接 Runner。
6. Runner 采样、更新策略并保存 `model_*.pt`。
7. `play.py` 加载检查点回放，并导出 `policy.pt`、`policy.onnx`。

`num_steps_per_env` 是每次更新前每个环境的采样步数，`max_iterations` 是更新轮数。两者都不是物理仿真步数。

## Go2 标准路线

```text
unitree_go2/__init__.py
  -> UnitreeGo2FlatEnvCfg
    -> UnitreeGo2RoughEnvCfg
      -> LocomotionVelocityRoughEnvCfg
        -> ManagerBasedRLEnvCfg
```

Go2 Flat 继承 Rough 后将地形切为平面，关闭高度扫描观测和地形课程。Actor 的单帧输入为机身角速度 3、重力投影 3、速度指令 3、相对关节位置 12、关节速度 12、上一步动作 12，共 45 维。动作表示相对默认姿态的偏移：`q_target = q_default + action_scale * action`。关节顺序是 FR、FL、RR、RL，每腿 hip/thigh/calf。

HIM 在 PPO 上增加历史状态估计。`him_vec_env_wrapper.py` 整理历史帧，`him_actor_critic.py` 和 `him_estimator.py` 定义网络，`him_on_policy_runner.py` 组织训练。部署时仍需使关节顺序、观测预处理、历史顺序、控制频率与训练一致。
