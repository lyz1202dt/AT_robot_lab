# Robot Lab训练机器人基本流程命令示例（GO2机器人）

CSDOG1
任务名称：
RobotLab-Isaac-Velocity-Flat-CSDog1-v0                 --47维平地
RobotLab-Isaac-Velocity-Flat-CSDog1-282-v0             --282维平地
RobotLab-Isaac-Velocity-Flat-CSDog1-282-DomainRand-v0  --282维平地域随机化
RobotLab-Isaac-Velocity-Stairs-CSDog1-282-v0           --282维台阶
RobotLab-Isaac-Velocity-Rough-CSDog1-v0                --崎岖

注意：ATDOG已经废弃，请使用ATDOG3

export http_proxy=192.168.2.180:7890
export https_proxy=192.168.2.180:7890
export socket_proxy=192.168.2.180:7890

1. ssh -p 3022  sw@shenweitechnology.com
2. cd /home/sw/code/AT_rl_sar
3.

export http_proxy=192.168.2.180:7890
export https_proxy=192.168.2.180:7890

4. docker/container.sh start

## 创建环境：

```bash
docker/container.sh build
docker/container.sh start
docker/container.sh enter
```

## 开始训练

```bash
cd /workspace/isaaclab_extension_template

# CSDog1 47维平地训练
/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab_extension_template/scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-CSDog1-v0 \
  --num_envs=4096 \
  --max_iterations=2000 \
  --headless

# CSDog1 崎岖地形训练
/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab_extension_template/scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-CSDog1-v0 \
  --num_envs=4096 \
  --max_iterations=2000 \
  --headless

# CSDog1 282维平地训练
/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab_extension_template/scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-CSDog1-282-v0 \
  --num_envs=4096 \
  --max_iterations=1000 \
  --headless

# CSDog1 282维平地域随机化续训
/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab_extension_template/scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-CSDog1-282-DomainRand-v0 \
  --num_envs=4096 \
  --max_iterations=1000 \
  --headless \
  --resume \
  --load_run=<csdog1_flat_282_run> \
  --checkpoint=<checkpoint.pt>

# CSDog1 楼梯从低到高续训：见下方“CSDog1 楼梯分阶段课程”章节。

/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab_extension_template/scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-ATDog-Dog2-v0 \
  --agent=rsl_rl_him_cfg_entry_point \
  --num_envs=6500 \
  --max_iterations=1000 \
  --headless

cd /workspace/isaaclab_extension_template
/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab_extension_template/scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-ATDog-Dog2-Arm-v0 \
  --num_envs=10000 \
  --max_iterations=2000\
  --headless \
  --resume \
  --load_run=2026-06-12_13-55-44 \
  --checkpoint=model_39600.pt 

/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab_extension_template/scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Slope-ATDog-Dog3-v0 \
  --num_envs=30000 \
  --max_iterations=5000\
  --headless \
  --resume \
  --load_run=2026-06-13_07-04-36\
  --checkpoint=model_18400.pt 


/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab_extension_template/scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-ATDog-Dog_Arm-v0 \
  --num_envs=10000 \
  --max_iterations=2000\
  --headless



/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab_extension_template/scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-ATDog-Dog2-Arm-v0 \
  --num_envs=6000 \
  --max_iterations=2000\
  --headless \
  --resume \
  --load_run=2026-06-14_11-56-28 \
  --checkpoint=model_41599.pt

/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab_extension_template/scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Slope-ATDog-Dog3-v0 \
  --num_envs=4000 \
  --max_iterations=2000\
  --headless \
  --resume \
  --load_run=2026-06-14_00-27-05 \
  --checkpoint=model_11400.pt
```

## CSDog1 楼梯分阶段课程（从低到高）

`CSDOG1_STAIR_STAGE=0…4` 在启动时选择阶段，同时联动实际地形的高度分布、最小/最大高度和默认追加训练轮数。不设置时读取 `stairs_config.json` 的 `stage`（当前为 4）。直接修改 JSON 的 `stage` 也会联动，不再需要手动修改 `env.terrain.stair_height_bands`。

| 阶段 | 楼梯高度分布（高度 cm：比例） | 默认追加训练轮数 |
| --- | --- | --- |
| 0 | 3–5：100% | 500 |
| 1 | 3–5：30%；5–8：70% | 500 |
| 2 | 3–5：10%；5–8：25%；8–12：65% | 800 |
| 3 | 3–5：5%；5–8：10%；8–12：20%；12–16：65% | 1000 |
| 4 | 3–5：5%；5–8：5%；8–12：10%；12–16：20%；15–20：60% | 1200 |

整体保留 30% 平地、70% 楼梯；楼梯内为 65% 上楼、35% 下楼。阶段不会自动升级。每阶段验证上楼、下楼、侧移和站立表现后，再执行下一阶段；选择通过率好的 checkpoint，不能仅按总奖励选择。

以下命令在现有 Docker 容器内执行，使用本文件已有的路径。宿主机运行时，切到项目根目录，将 `/workspace/isaaclab/isaaclab.sh -p` 替换为已安装 Isaac Lab 的 Python 解释器即可。无需重新从 47 维任务开始，Stage 0 从已验证的 491 维 Critic 鲁棒平地权重开始。

```bash
cd /workspace/isaaclab_extension_template

# Stage 0：3–5 cm，追加 500 轮。把路径改成实际的鲁棒平地权重。
CSDOG1_STAIR_STAGE=0 /workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Stairs-CSDog1-282-v0 \
  --agent=rsl_rl_him_cfg_entry_point --num_envs=4096 --headless \
  --run_name=stairs_stage0 --resume \
  --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/csdog1_flat_282_domain_rand/flat_282/model_2000.pt

# Stage 1：最高 8 cm，追加 500 轮。替换为 Stage 0 最好权重的绝对路径。
CSDOG1_STAIR_STAGE=1 /workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Stairs-CSDog1-282-v0 \
  --agent=rsl_rl_him_cfg_entry_point --num_envs=4096 --headless \
  --run_name=stairs_stage1 --resume \
  --checkpoint=/absolute/path/to/stage0_best.pt

# Stage 2：最高 12 cm，追加 800 轮。替换为 Stage 1 最好权重的绝对路径。
CSDOG1_STAIR_STAGE=2 /workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Stairs-CSDog1-282-v0 \
  --agent=rsl_rl_him_cfg_entry_point --num_envs=4096 --headless \
  --run_name=stairs_stage2 --resume \
  --checkpoint=/absolute/path/to/stage1_best.pt

# Stage 3：最高 16 cm，追加 1000 轮。替换为 Stage 2 最好权重的绝对路径。
CSDOG1_STAIR_STAGE=3 /workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Stairs-CSDog1-282-v0 \
  --agent=rsl_rl_him_cfg_entry_point --num_envs=4096 --headless \
  --run_name=stairs_stage3 --resume \
  --checkpoint=/absolute/path/to/stage2_best.pt

# Stage 4：最高 20 cm，追加 1200 轮。替换为 Stage 3 最好权重的绝对路径。
CSDOG1_STAIR_STAGE=4 /workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Stairs-CSDog1-282-v0 \
  --agent=rsl_rl_him_cfg_entry_point --num_envs=4096 --headless \
  --run_name=stairs_stage4 --resume \
  --checkpoint=/absolute/path/to/stage3_best.pt
```

每条命令单独执行，替换占位路径后再运行。各阶段日志保存在 `logs/rsl_rl/csdog1_stairs_282/<时间>_stairs_stageN/`，无需重命名实际 checkpoint 文件。默认不恢复 Adam 状态，保留 checkpoint 的累计迭代数。例如从 `model_2000.pt` 完整训练 Stage 0 后是 `model_2500.pt`；训练轮数表示本次追加轮数，若需延长某阶段可显式加 `--max_iterations=1000` 覆盖默认值。

回放时指定与训练相同的阶段。例如验证 Stage 0（带界面）：

```bash
CSDOG1_STAIR_STAGE=0 /workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Stairs-CSDog1-282-v0 \
  --agent=rsl_rl_him_cfg_entry_point --num_envs=32 \
  --checkpoint=/absolute/path/to/stage0_best.pt
```

这里只修复阶段选择联动，沿用当前项目的奖励和随机化参数；当前奖励与原项目已有差异，因此这组命令复现的是由低到高的课程流程。

## 导出模型：

```bash
cd /workspace/isaaclab_extension_template

# CSDog1 47维平地模型导出
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-CSDog1-v0 \
  --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/csdog1_flat_47/<run>/model_<iteration>.pt \
  --num_envs=10 \
  --headless

# CSDog1 崎岖地形模型导出
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Rough-CSDog1-v0 \
  --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/csdog1_rough/<run>/model_<iteration>.pt \
  --num_envs=10 \
  --headless

# CSDog1 282维平地模型导出
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-CSDog1-282-v0 \
  --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/csdog1_flat_282/<run>/model_<iteration>.pt \
  --num_envs=10 \
  --headless

# CSDog1 282维平地域随机化模型导出
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-CSDog1-282-DomainRand-v0 \
  --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/csdog1_flat_282_domain_rand/<run>/model_<iteration>.pt \
  --num_envs=10 \
  --headless

# CSDog1 282维台阶模型导出
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Stairs-CSDog1-282-v0 \
  --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/csdog1_stairs_282/<run>/model_<iteration>.pt \
  --num_envs=10 \
  --headless


/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py   --task=RobotLab-Isaac-Velocity-Slope-ATDog-Dog2-v0   --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog2_slope_him/2026-09-28_08-38-05/model_11000.pt     --num_envs=10  --agent=rsl_rl_him_cfg_entry_point 


/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Rough-ATDog-Dog2-Arm-v0 \
  --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog2_arm_rough/2026-06-14_13-10-55/model_43598.pt \
  --num_envs=1\
  --headless

/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Rough-ATDog-Dog3-Arm-v0 \
  --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog3_arm_rough/2026-06-10_14-05-13/model_38986.pt \
  --num_envs=1\
  --headless



/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-ATDog-Dog2-v0 \
  --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog_flat/2026-05-31_14-32-45/model_4999.pt\
  --num_envs=10

/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Slope-ATDog-Dog2-v0 \
  --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog2_slope_him/2026-09-12_11-58-40/model_15900.pt \
  --num_envs=10\
  --agent=rsl_rl_him_cfg_entry_point
```

## 复制导出的模型到宿主机

初始：2026-05-23_14-26-00

scp -P 3022 sw@shenweitechnology.com:/home/sw/code/AT_rl_sar/logs/rsl_rl/atdog_dog2_flat/2026-06-07_18-30-45/exported/policy.pt ~/桌面/

scp -P 3022 sw@shenweitechnology.com:/home/sw/code/AT_rl_sar/logs/rsl_rl/atdog_dog2_arm_rough/2026-06-14_03-42-01/exported/policy.pt ~/桌面/

scp -P 3022 sw@shenweitechnology.com:/home/sw/code/AT_rl_sar/logs/rsl_rl/atdog_dog3_arm_rough/2026-06-10_14-05-13/exported/policy.pt ~/桌面/

scp -P 3022 sw@shenweitechnology.com:/home/sw/code/AT_rl_sar/logs/rsl_rl/atdog_dog2_arm_rough/2026-06-10_11-04-22 ~/桌面/

```bash
docker cp robot-lab:/workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog2_arm_rough/2026-06-10_11-04-22  robot-lab:/workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog3_arm_rough/

docker cp ~/桌面/2026-06-12_13-55-44 robot-lab:/workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog2_arm_rough/


cp atdog_dog2_arm_rough/2026-06-10_11-04-22  atdog_dog3_arm_rough/
```




```
如果训练仿真跑到CPU上，容器内执行：

ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
      /usr/lib/x86_64-linux-gnu/libcuda.so

ldconfig
```
