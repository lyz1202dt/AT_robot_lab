# 在 AT_robot_lab 中加入一条狗并训练不同任务

本文以仓库已有的 AT-DOG2 为例，说明如何把一台四足机器人接入 Isaac Lab，并开展平地、粗糙地形、楼梯、沙地、横杆和斜坡训练。

AT-DOG2 的训练配置、URDF 和网格已同步自 `AT_rl_sar`。实际 Dog2 使用 `merge_fixed_joints=False` 保留 `.*_foot` 接触刚体；初始机身高度为 0.40 m，腿部默认 thigh 角为 -0.7854 rad。Dog2 特有的高度、姿态、接触和镜像奖励在 `dog2/rewards.py` 中，避免改变 CSDog1 和 03dog 的公共奖励行为。

注意：`AT_DOG4_CFG` 原本也引用 `dog2/urdf/dog2.urdf`，因此本次 URDF/网格替换同样会改变 Dog4 加载的模型；Dog4 自身的任务配置和执行器配置未修改。

在项目根目录运行训练（以粗糙地形为例）：

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-ATDog-Dog2-v0 \
  --num_envs=10000 --max_iterations=2000 --headless
```

上述命令适用于项目提供的 Isaac Lab 容器；在宿主机上可改用已安装 Isaac Lab 的 Python 解释器。源项目 `AT_DESCRIPE.md` 中 Dog2 的平地、台阶、沙地和斜坡任务 ID 均在 `dog2/__init__.py` 注册；横杆任务也在此注册。继续训练时可追加 `--resume --load_run=<run_name> --checkpoint=<checkpoint_name>`，请使用目标项目自身生成的 checkpoint。

## 一、整体流程

```text
URDF/mesh 资源
      ↓
robot_lab/assets/<robot>.py       # ArticulationCfg：物理、初始姿态、电机
      ↓
tasks/.../config/quadruped/<robot>/ # 场景、观测、动作、奖励、地形
      ↓
__init__.py                       # gym.register 注册任务名
      ↓
scripts/reinforcement_learning/.../train.py
```

机器人资产配置和强化学习任务配置是两层概念：资产文件定义“这条狗是什么”，环境配置定义“这条狗要学什么”。

## 二、准备机器人资源

将文件放在类似目录中：

```text
source/robot_lab/data/Robots/mydog/
├── urdf/mydog.urdf
└── meshes/*.stl 或 *.dae
```

检查 URDF 中的：

1. `joint name` 是否唯一，并记录可驱动关节顺序；
2. `continuous/revolute` 关节的限位、轴方向和阻尼是否正确；
3. mesh 路径是否能从 URDF 位置正确解析；
4. 足端 link 是否为固定关节。如果设置 `merge_fixed_joints=True`，固定足端会并入父 link，接触奖励的 body 名称也要随之修改。

建议先用 Isaac Lab 的资产查看或一个最小环境确认机器人能加载、不会爆炸或穿地。

## 三、创建资产配置文件

复制 `source/robot_lab/robot_lab/assets/atdog.py`，或者在其中新增配置：

```python
from isaaclab import sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

MYDOG_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/mydog/urdf/mydog.urdf",
        fix_base=False,
        merge_fixed_joints=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        # 关闭 URDF 导入时的默认 PD，统一由 actuator 配置控制
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0, damping=0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.40),
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": -0.7854,
            ".*_calf_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=30.0,
            velocity_limit=30.0,
            stiffness=40.0,
            damping=1.0,
            friction=0.32,
            dynamic_friction=0.29274320,
            viscous_friction=0.02658733,
            min_delay=1,
            max_delay=2,
        ),
    },
)
```

`effort_limit` 是最大力矩，`velocity_limit` 是最大关节速度，`stiffness/damping` 是 PD 增益，`friction` 是执行器库仑摩擦近似，`min_delay/max_delay` 以物理步为单位模拟控制延迟。

如果 12 个电机的摩擦参数不同，可以拆成 12 个 actuator group；当前 Dog2 在 `atdog.py` 中使用一个 `legs` group。

## 四、创建环境配置

推荐以已有环境为模板：

```text
source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/mydog/
├── __init__.py
├── flat_env_cfg.py
├── rough_env_cfg.py
├── stairs_env_cfg.py
├── sand_env_cfg.py
├── bar_env_cfg.py
├── slope_env_cfg.py
└── agents/
    ├── __init__.py
    ├── rsl_rl_ppo_cfg.py
    └── cusrl_ppo_cfg.py
```

最省事的做法是复制 `.../atdog/dog2/` 目录后批量替换类名、导入路径和任务 ID。

在粗糙环境中至少需要修改：

```python
from robot_lab.assets.mydog import MYDOG_CFG

self.scene.robot = MYDOG_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot"
)
self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/base"
self.base_link_name = "base"       # 改成 URDF 实际机身 link
self.foot_link_name = ".*_foot"     # 改成实际足端 body
self.joint_names = [                # 必须与动作和观测顺序一致
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
```

`joint_names`、动作输出顺序和策略观测中的关节顺序必须一致，否则机器人可能出现“动作对应错关节”的问题。

## 五、配置不同训练任务

### 1. 平地行走

以 `flat_env_cfg.py` 为模板：

```python
self.scene.terrain.terrain_type = "plane"
self.scene.terrain.terrain_generator = None
self.scene.height_scanner = None
self.observations.policy.height_scan = None
self.observations.critic.height_scan = None
self.curriculum.terrain_levels = None
```

适合首先验证：站立、前进、后退、横移和转向是否正常。

### 2. 粗糙地形

在 `rough_env_cfg.py` 中使用 `TerrainGeneratorCfg`，可组合随机高度、斜坡或台阶。应保留高度扫描器，并确认扫描器挂载到真实机身 link。训练通常先从较小地形起伏开始，再增大难度。

### 3. 楼梯

`stairs_env_cfg.py` 通常使用 `MeshPyramidStairsTerrainCfg` 或 `MeshInvertedPyramidStairsTerrainCfg`。关键参数包括台阶高度、台阶宽度、平台宽度和是否有坑洞。建议先固定台阶高度，再启用地形 curriculum。

### 4. 沙地

`sand_env_cfg.py` 可通过改变接触材料、地形高度或随机化参数模拟松软地面。应重点观察足端滑移、身体高度奖励和能耗惩罚，必要时降低初始命令速度。

### 5. 横杆/障碍物

`bar_env_cfg.py` 用于训练跨越固定障碍。需要确保障碍物高度没有超过机器人可达范围，并调整碰撞惩罚、足端接触奖励和命令速度。

### 6. 斜坡

`slope_env_cfg.py` 用斜坡地形训练上下坡能力。建议逐步增加坡度，并检查机身 pitch 姿态、足端接触和防翻倒终止条件。

## 六、注册 Gym 任务

在新目录的 `__init__.py` 中注册环境：

```python
gym.register(
    id="RobotLab-Isaac-Velocity-Flat-MyDog-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:MyDogFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyDogFlatPPORunnerCfg",
    },
)
```

每个地形任务都要有唯一的 `id`。`env_cfg_entry_point` 指向环境配置，`rsl_rl_cfg_entry_point` 指向 RSL-RL 训练器配置；如果使用 CusRL 或 HIM，也要注册对应 entry point。

同时确认上级包会导入该目录，否则 Gym 不会执行注册代码。可参考现有 `atdog` 任务包的导入方式。

## 七、配置 PPO/HIM/CusRL

复制 `dog2/agents/` 下的配置，并重点检查：

- `num_actor_obs`、`num_critic_obs` 是否与新观测一致；
- 动作维度是否等于可驱动关节数；
- `experiment_name`、`run_name` 是否区分不同任务；
- 训练迭代次数、保存频率和日志目录；
- 如果修改观测维度，不能直接加载旧 checkpoint，否则会出现网络输入维度不匹配。

## 八、启动训练

在 Isaac Lab 环境中运行：

```bash
# 平地 PPO
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-MyDog-v0 \
  --num_envs 2048 --headless

# 粗糙地形 PPO
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-MyDog-v0 \
  --num_envs 2048 --headless

# CusRL
python scripts/reinforcement_learning/cusrl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-MyDog-v0 \
  --num_envs 2048 --headless
```

如果项目需要通过 Isaac Lab wrapper 启动，则使用仓库 README 中对应的 `isaaclab.sh -p` 形式。

## 九、回放和验证

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-MyDog-v0 \
  --num_envs 16 \
  --checkpoint /path/to/model.pt
```

常用选项：

```bash
--video --video_length 300   # 录制视频
--keyboard                   # 单机器人键盘控制（play 时使用）
```

验证时应依次检查：机器人是否正确生成、初始姿态是否稳定、12 个动作是否对应正确关节、足端接触是否正确、命令速度是否能跟踪、是否频繁触发翻倒终止。

## 十、推荐训练顺序

1. **静态站立**：关闭或降低速度命令，先确认初始姿态和 PD 参数；
2. **平地慢速行走**：小范围随机 reset，确认动作/观测顺序；
3. **平地全速度训练**：加入转向、横移和外力扰动；
4. **粗糙地形**：启用高度扫描器和轻微地形 curriculum；
5. **楼梯、斜坡、沙地**：逐项增加难度，分别保存 checkpoint；
6. **组合泛化**：混合地形、质量/质心/摩擦随机化，最后再做 sim-to-real 验证。

## 十一、常见错误排查

| 现象 | 常见原因 | 检查项 |
|---|---|---|
| 找不到任务 | Gym 没有执行注册文件 | `__init__.py` 导入链、任务 ID |
| 机器人加载失败 | URDF 或 mesh 路径错误 | `asset_path`、mesh 相对路径 |
| 机器人一生成就倒 | 初始高度/姿态或 PD 不合适 | `init_state`、`stiffness`、关节方向 |
| 动作对应错腿 | 关节顺序不一致 | `joint_names`、actuator 顺序、策略维度 |
| 足端奖励无效 | fixed joint 被合并 | 实际 body 名称、`foot_link_name` |
| resume 报 shape mismatch | 修改了观测或动作维度 | 新建 run，不要直接加载旧模型 |
| 训练极不稳定 | 接触、延迟或随机化过强 | 降低地形难度、外力和控制延迟 |

## 十二、提交前检查清单

- [ ] URDF、mesh 已放入 `source/robot_lab/data/Robots/`；
- [ ] `ArticulationCfg` 能独立导入；
- [ ] 机身、足端和 12 个关节名称与 URDF 一致；
- [ ] 动作、观测、奖励引用的关节顺序一致；
- [ ] 至少注册一个 flat 任务并成功启动；
- [ ] 再逐步启用 rough、stairs、sand、bar、slope；
- [ ] 每种任务使用独立日志目录和 checkpoint；
- [ ] 已用 `ast.parse` 或 Python 导入检查配置文件语法。
