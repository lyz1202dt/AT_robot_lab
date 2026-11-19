# ATDog2 与 03dog 训练命令

以下命令在项目提供的 Isaac Lab 容器内、从项目根目录执行。宿主机使用时，将 `/workspace/isaaclab/isaaclab.sh -p` 换成已安装 Isaac Lab 的 Python 解释器。每个代码块中的训练或回放命令独立执行，按显存调整 `--num_envs`。

```bash
docker/container.sh build
docker/container.sh start
docker/container.sh enter
cd /workspace/isaaclab_extension_template
```

首次加载 03dog，或替换 CAD 导出的 URDF 后，先在项目根目录执行：

```bash
python3 scripts/tools/prepare_03dog_urdf.py
```

建议首次用 `--num_envs=32 --max_iterations=2` 做训练冒烟检查。训练日志位于 `logs/rsl_rl/<实验名>/<时间>_<run_name>/`；`--run_name` 可省略。以下续训和回放命令中的绝对 checkpoint 路径均需替换成实际 `model_*.pt` 文件。

## ATDog2

ATDog2 任务均注册 RSL-RL PPO 和 CusRL，以下使用 `scripts/reinforcement_learning/rsl_rl/` 的 RSL-RL 入口。关节、接触和奖励实现见 `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/atdog/dog2/`；资产配置见 `source/robot_lab/robot_lab/assets/atdog.py`。`Flat270` 和 `Rough270` 是 45 维单帧乘 6 帧历史，不能直接加载 45 维策略权重。

| 任务 | 用途 | 实验名（`logs/rsl_rl/` 下） | 配置默认轮数 |
| --- | --- | --- | ---: |
| `RobotLab-Isaac-Velocity-Flat-ATDog-Dog2-v0` | 平地，45 维 | `atdog_dog2_flat` | 5000 |
| `RobotLab-Isaac-Velocity-Flat270-ATDog-Dog2-v0` | 平地，270 维 | `atdog_dog2_flat270` | 5000 |
| `RobotLab-Isaac-Velocity-Rough-ATDog-Dog2-v0` | 崎岖地形，45 维 | `atdog_dog2_rough` | 20000 |
| `RobotLab-Isaac-Velocity-Rough270-ATDog-Dog2-v0` | 崎岖地形，270 维 | `atdog_dog2_rough270` | 20000 |
| `RobotLab-Isaac-Velocity-Stairs-ATDog-Dog2-v0` | 楼梯 | `atdog_dog2_stairs` | 20000 |
| `RobotLab-Isaac-Velocity-Sand-ATDog-Dog2-v0` | 沙地 | `atdog_dog2_sand` | 20000 |
| `RobotLab-Isaac-Velocity-Slope-ATDog-Dog2-v0` | 斜坡 | `atdog_dog2_slope` | 20000 |
| `RobotLab-Isaac-Velocity-Bar-ATDog-Dog2-v0` | 横杆 | `atdog_dog2_bar` | 5000 |

### 从头训练

```bash
# 45 维平地
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-ATDog-Dog2-v0 \
  --agent=rsl_rl_cfg_entry_point --num_envs=4096 --max_iterations=5000 --headless

# 270 维平地
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat270-ATDog-Dog2-v0 \
  --agent=rsl_rl_cfg_entry_point --num_envs=4096 --max_iterations=5000 --headless

# 45 维崎岖地形
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-ATDog-Dog2-v0 \
  --agent=rsl_rl_cfg_entry_point --num_envs=4096 --max_iterations=20000 --headless

# 270 维崎岖地形
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough270-ATDog-Dog2-v0 \
  --agent=rsl_rl_cfg_entry_point --num_envs=4096 --max_iterations=20000 --headless

# 楼梯
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Stairs-ATDog-Dog2-v0 \
  --agent=rsl_rl_cfg_entry_point --num_envs=4096 --max_iterations=20000 --headless

# 沙地
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Sand-ATDog-Dog2-v0 \
  --agent=rsl_rl_cfg_entry_point --num_envs=4096 --max_iterations=20000 --headless

# 斜坡
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Slope-ATDog-Dog2-v0 \
  --agent=rsl_rl_cfg_entry_point --num_envs=4096 --max_iterations=20000 --headless

# 横杆
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Bar-ATDog-Dog2-v0 \
  --agent=rsl_rl_cfg_entry_point --num_envs=4096 --max_iterations=5000 --headless
```

### 续训

续训使用相同任务 ID 和算法，`--max_iterations` 指本次追加轮数。以下以崎岖地形为例；其他任务替换 `--task` 和 checkpoint 路径即可。使用同实验目录内的运行时，也可指定 `--load_run=时间戳目录名 --checkpoint=model_1000.pt`。

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-ATDog-Dog2-v0 \
  --agent=rsl_rl_cfg_entry_point --num_envs=4096 --max_iterations=2000 --headless \
  --resume --checkpoint=/absolute/path/to/atdog_dog2_rough/model_1000.pt \
  --run_name=rough_continue
```

### 回放与导出

`play.py` 加载 checkpoint 后会在其相邻的 `exported/` 目录生成 `policy.pt` 和 `policy.onnx`。以下使用 `--export_only --headless` 仅导出；要观看回放，去掉这两个参数，并按需要调整 `--num_envs`。

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-ATDog-Dog2-v0 \
  --checkpoint=/absolute/path/to/atdog_dog2_flat/model_5000.pt --num_envs=10 --export_only --headless

/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat270-ATDog-Dog2-v0 \
  --checkpoint=/absolute/path/to/atdog_dog2_flat270/model_5000.pt --num_envs=10 --export_only --headless

/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Rough-ATDog-Dog2-v0 \
  --checkpoint=/absolute/path/to/atdog_dog2_rough/model_20000.pt --num_envs=10 --export_only --headless

/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Rough270-ATDog-Dog2-v0 \
  --checkpoint=/absolute/path/to/atdog_dog2_rough270/model_20000.pt --num_envs=10 --export_only --headless

/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Stairs-ATDog-Dog2-v0 \
  --checkpoint=/absolute/path/to/atdog_dog2_stairs/model_20000.pt --num_envs=10 --export_only --headless

/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Sand-ATDog-Dog2-v0 \
  --checkpoint=/absolute/path/to/atdog_dog2_sand/model_20000.pt --num_envs=10 --export_only --headless

/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Slope-ATDog-Dog2-v0 \
  --checkpoint=/absolute/path/to/atdog_dog2_slope/model_20000.pt --num_envs=10 --export_only --headless

/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Bar-ATDog-Dog2-v0 \
  --checkpoint=/absolute/path/to/atdog_dog2_bar/model_5000.pt --num_envs=10 --export_only --headless
```

## 03dog

03dog 的 Python 配置在 `config/quadruped/dog03/`，模型在 `source/robot_lab/data/Robots/03dog/`。关节顺序为 FL、FR、RL、RR，每腿 hip/thigh/calf。47 维任务使用 PPO；282 维任务使用 6 帧历史、491 维 Critic，默认是 HIM。旧的 50 维 Critic checkpoint 无法直接用于新的 282 维任务。奖励、观测与环境参数详见 [dog03/README.md](source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/dog03/README.md)。

| 任务 | 用途 | 实验名（`logs/rsl_rl/` 下） | 配置默认轮数 |
| --- | --- | --- | ---: |
| `RobotLab-Isaac-Velocity-Flat-03Dog-v0` | 47 维平地，PPO | `dog03_flat_47` | 2000 |
| `RobotLab-Isaac-Velocity-Rough-03Dog-v0` | 崎岖地形，PPO | `dog03_rough` | 2000 |
| `RobotLab-Isaac-Velocity-Flat-03Dog-282-v0` | 282 维平地，HIM | `dog03_flat_282` | 1000 |
| `RobotLab-Isaac-Velocity-Flat-03Dog-282-DomainRand-v0` | 282 维地域随机化，HIM | `dog03_flat_282_domain_rand` | 1000 |
| `RobotLab-Isaac-Velocity-Stairs-03Dog-282-v0` | 282 维楼梯，HIM | `dog03_stairs_282` | 按阶段 |

### 从头训练

```bash
# 47 维平地
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-03Dog-v0 \
  --agent=rsl_rl_cfg_entry_point --num_envs=4096 --max_iterations=2000 --headless

# 崎岖地形
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-03Dog-v0 \
  --agent=rsl_rl_cfg_entry_point --num_envs=4096 --max_iterations=2000 --headless

# 282 维平地：先建立与后续阶段兼容的 491 维 Critic 权重
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-03Dog-282-v0 \
  --agent=rsl_rl_him_cfg_entry_point --num_envs=4096 --max_iterations=1000 --headless

# 282 维地域随机化：也可以直接从头训练，2000 轮覆盖恢复课程
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-03Dog-282-DomainRand-v0 \
  --agent=rsl_rl_him_cfg_entry_point --num_envs=4096 --max_iterations=2000 --headless
```

### 续训与阶段迁移

同任务续训或从 282 维基础平地迁移到地域随机化时，明确指定 HIM 和实际 checkpoint。`--max_iterations` 是本次追加轮数；不要将 47 维 PPO 权重用于 282 维 HIM。

```bash
# 从 282 维基础平地进入地域随机化
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-03Dog-282-DomainRand-v0 \
  --agent=rsl_rl_him_cfg_entry_point --num_envs=4096 --max_iterations=1000 --headless \
  --resume --checkpoint=/absolute/path/to/dog03_flat_282/model_1000.pt \
  --run_name=domain_rand_from_flat

# 地域随机化任务继续训练
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-03Dog-282-DomainRand-v0 \
  --agent=rsl_rl_him_cfg_entry_point --num_envs=4096 --max_iterations=1000 --headless \
  --resume --checkpoint=/absolute/path/to/dog03_flat_282_domain_rand/model_2000.pt \
  --run_name=domain_rand_continue
```

### 楼梯课程

`DOG03_STAIR_STAGE=0..4` 在启动时选择楼梯高度分布和默认追加轮数，阶段不会自动升级。默认阶段为 0；配置见 `dog03/cyberstar/stairs_config.json`。整体地形为 30% 平地、70% 楼梯；楼梯中 65% 上楼、35% 下楼。Stage 0 从已验证的 03dog 282 维地域随机化 checkpoint 开始，后续阶段使用上一阶段验证通过的 checkpoint。

| 阶段 | 楼梯高度分布（cm：占比） | 默认追加轮数 |
| --- | --- | ---: |
| 0 | 3-5：100% | 500 |
| 1 | 3-5：30%；5-8：70% | 500 |
| 2 | 3-5：10%；5-8：25%；8-12：65% | 800 |
| 3 | 3-5：5%；5-8：10%；8-12：20%；12-16：65% | 1000 |
| 4 | 5-8：5%；8-12：5%；12-15：20%；15-17：70% | 1500 |

```bash
DOG03_STAIR_STAGE=0 /workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Stairs-03Dog-282-v0 \
  --agent=rsl_rl_him_cfg_entry_point --num_envs=4096 --headless \
  --run_name=stairs_stage0 --resume \
  --checkpoint=/absolute/path/to/dog03_flat_282_domain_rand/model_2000.pt

DOG03_STAIR_STAGE=1 /workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Stairs-03Dog-282-v0 \
  --agent=rsl_rl_him_cfg_entry_point --num_envs=4096 --headless \
  --run_name=stairs_stage1 --resume --checkpoint=/absolute/path/to/stage0_best.pt

DOG03_STAIR_STAGE=2 /workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Stairs-03Dog-282-v0 \
  --agent=rsl_rl_him_cfg_entry_point --num_envs=4096 --headless \
  --run_name=stairs_stage2 --resume --checkpoint=/absolute/path/to/stage1_best.pt

DOG03_STAIR_STAGE=3 /workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Stairs-03Dog-282-v0 \
  --agent=rsl_rl_him_cfg_entry_point --num_envs=4096 --headless \
  --run_name=stairs_stage3 --resume --checkpoint=/absolute/path/to/stage2_best.pt

DOG03_STAIR_STAGE=4 /workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Stairs-03Dog-282-v0 \
  --agent=rsl_rl_him_cfg_entry_point --num_envs=4096 --headless \
  --run_name=stairs_stage4 --resume --checkpoint=/absolute/path/to/stage3_best.pt
```

### 回放与导出

以下命令仅导出模型到 checkpoint 相邻的 `exported/` 目录。03dog HIM 任务还会导出部署元数据；实际回放时去掉 `--export_only --headless`。楼梯回放和导出必须设置与训练相同的 `DOG03_STAIR_STAGE`。

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-03Dog-v0 \
  --checkpoint=/absolute/path/to/dog03_flat_47/model_2000.pt --num_envs=10 --export_only --headless

/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Rough-03Dog-v0 \
  --checkpoint=/absolute/path/to/dog03_rough/model_2000.pt --num_envs=10 --export_only --headless

/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-03Dog-282-v0 \
  --agent=rsl_rl_him_cfg_entry_point \
  --checkpoint=/absolute/path/to/dog03_flat_282/model_1000.pt --num_envs=10 --export_only --headless

/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-03Dog-282-DomainRand-v0 \
  --agent=rsl_rl_him_cfg_entry_point \
  --checkpoint=/absolute/path/to/dog03_flat_282_domain_rand/model_2000.pt --num_envs=10 --export_only --headless

DOG03_STAIR_STAGE=4 /workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Stairs-03Dog-282-v0 \
  --agent=rsl_rl_him_cfg_entry_point \
  --checkpoint=/absolute/path/to/stage4_best.pt --num_envs=10 --export_only --headless
```

更多 ATDog2 资产与任务配置说明见 [ATDOG_TRAINING_GUIDE.md](docs/ATDOG_TRAINING_GUIDE.md)，03dog 随机化细节见 [DOMAIN_RANDOMIZATION.md](source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/dog03/DOMAIN_RANDOMIZATION.md)。
