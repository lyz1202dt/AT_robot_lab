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

# CSDog1 282维台阶续训
/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab_extension_template/scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Stairs-CSDog1-282-v0 \
  --num_envs=4096 \
  --max_iterations=500 \
  --headless \
  --resume \
  --load_run=<csdog1_flat_282_domain_rand_run> \
  --checkpoint=<checkpoint.pt>

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

