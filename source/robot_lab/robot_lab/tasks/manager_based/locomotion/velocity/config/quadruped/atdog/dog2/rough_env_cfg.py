# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs import mdp as isaaclab_mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
# # use cloud assets
# from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
# use local assets
from robot_lab.assets.atdog import AT_DOG2_CFG  # isort: skip
from . import rewards as dog2_rewards

# 自定义台阶地形（倒金字塔上台阶）:
# - 台阶高 10cm
# - 台阶水平长度 30cm
# 说明:
# 使用 MeshInvertedPyramidStairsTerrainCfg（与 Isaac Lab 默认 rough 配置一致）
# 生成倒金字塔台阶，机器人可从低处向高处持续上台阶。
DOG2_ROUGH_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "stairs": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.01, 0.02),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs2": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.01, 0.02),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class ATDogDog2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    # 机身主刚体名称，用于:
    # 1) 传感器挂载（高度扫描器）
    # 2) 质量/质心/外力随机化时筛选 body
    # 3) 与 base 相关奖励项的 body 指定
    base_link_name = "base"
    # 保留 URDF 的固定关节，直接使用 foot 刚体的原点和接触数据。
    foot_link_name = ".*_foot"
    # fmt: off
    # 关节顺序与动作/观测向量顺序保持一致，避免策略输入输出错位
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    # fmt: on

    def __post_init__(self):
        # 先继承父类默认配置，再按 ATDog Dog2 粗糙地形任务覆写
        super().__post_init__()
        # 保持采样频率不变（sim.dt 与 decimation 沿用父类），仅将单回合时长设为 2s
        #self.episode_length_s = 2.0

        # ------------------------------Scene 场景与传感器------------------------------
        # 覆写默认 rough terrain：仅使用固定参数台阶地形
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = DOG2_ROUGH_TERRAIN_CFG
        # 指定机器人资产，并放置到每个并行环境的 Robot prim 下
        self.scene.robot = AT_DOG2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 将高度扫描器挂到机身 base 上，保证地形感知参考系一致
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations 观测------------------------------
        # 各观测分量缩放系数:
        # - scale 越大，对应量在策略输入中的数值幅度越大
        # - 需结合网络归一化习惯，避免某一类信号过强/过弱
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        # 关闭策略侧 base_lin_vel 与 height_scan 观测（即不输入给 actor）
        # 说明: 这里常用于做“部分可观测”训练，迫使策略更依赖本体状态
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        # 为了与历史 checkpoint 的网络输入维度保持一致，同时关闭 critic 侧高度扫描观测
        # 否则 resume 时会出现 critic 第一层权重 shape mismatch（例如 48 vs 235）
        self.observations.critic.height_scan = None
        # 明确 joint_pos/joint_vel 仅采集 joint_names 中定义的关节
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        # 添加姿态四元数观测，因为真实狗IMU能反馈角度（四元数形式）
        # self.observations.policy.base_orientation = ObsTerm(
        #     func=isaaclab_mdp.root_quat_w,
        #     noise=Unoise(n_min=-0.01, n_max=0.01),
        #     clip=(-1.0, 1.0),
        #     scale=1.0,
        # )
        # # 添加关节力矩观测，因为电机能反馈力矩
        # self.observations.policy.joint_effort = ObsTerm(
        #     func=isaaclab_mdp.joint_effort,
        #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names)},
        #     noise=Unoise(n_min=-0.01, n_max=0.01),
        #     clip=(-100.0, 100.0),
        #     scale=0.01,
        # )

        # ------------------------------Actions 动作------------------------------
        # 动作缩放:
        # - 髋关节幅度更小(0.125)，减少横摆过猛导致的不稳定
        # - 其余关节幅度0.25，保留足够摆动能力
        # 这里的 key 是正则，按关节名匹配后应用对应 scale
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        # 动作裁剪区间（非常宽），主要作为安全兜底防止异常值爆炸
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        # 限定动作控制关节集合，顺序与 joint_names 对齐
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events 随机化事件------------------------------
        self.events.randomize_rigid_body_material.params["asset_cfg"].body_names = [self.foot_link_name]
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (1.2, 2.0)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.8, 1.1)
        # reset 时随机化机身位姿与速度:
        # - pose_range: 初始位置/姿态扰动范围
        # - velocity_range: 初始线速度/角速度扰动范围
        # 目的: 提升鲁棒性，减少对单一起始状态过拟合
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-1.0, 1.0),
                "pitch": (-1.0, 1.0),
                "yaw": (-0.0, 0.0),
            },
            "velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (-1.0, 0.3),
                "roll": (-0.8, 0.8),
                "pitch": (-0.8, 0.8),
                "yaw": (-0.8, 0.8),
            },
        }
        # 仅随机化 base 的质量
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        # 随机化除 base 外其他刚体质量（负向前瞻正则: 不匹配 base）
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        # 仅随机化 base 质心位置
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        # 外力/外力矩扰动施加到 base，模拟推搡/干扰
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-20.0, 20.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-8.0, 8.0)
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.9, 1.1)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.9, 1.1)

        # ------------------------------Rewards 奖励函数------------------------------
        # 约定:
        # - 正权重: 鼓励该行为（reward）-> 底层返回正值，策略争取高分
        # - 负权重: 惩罚该行为（penalty）-> 底层返回误差/能耗，策略避免扣分
        # - 权重为0: 本项不参与训练（后面可统一 disable）

        # General
        # 触地不再终止回合，因此无需终止奖励。
        self.rewards.is_terminated.weight = 0.0

        # Root penalties
        # 惩罚机身 z 方向线速度，抑制“跳跃/颠簸”。
        # 绝对值增大 -> 更追求贴地平稳；过大可能抑制跨越障碍能力。
        self.rewards.lin_vel_z_l2.weight = -12.0
        # 惩罚机身 x/y 角速度（roll/pitch 旋转速度），降低侧翻和点头抖动。
        self.rewards.ang_vel_xy_l2.weight = -1.5
        # 完整重力方向误差在侧倒和倒置时仍持续惩罚姿态偏离。
        self.rewards.flat_orientation_l2.func = dog2_rewards.gravity_orientation_l2
        self.rewards.flat_orientation_l2.weight = -6.0
        # 指数高度惩罚在 0.1 m 误差时约为最大值的 63%，并在大误差时保持有界。
        self.rewards.base_height_l2.func = dog2_rewards.base_height_exp_penalty
        self.rewards.base_height_l2.weight = -2.0
        self.rewards.base_height_l2.params["std"] = 0.1
        # 目标机身高度（单位 m）。
        self.rewards.base_height_l2.params["target_height"] = 0.30
        # 指定用 base 刚体计算该项（避免多 body 统计带来歧义）。
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        # 惩罚机身线加速度（平滑机身受力/运动）
        # 该项常用于减小真实机器冲击与传感器饱和风险。
        self.rewards.body_lin_acc_l2.weight = -0.01
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        # 力矩 L2 惩罚，控制能耗并抑制“暴力驱动”。
        # 通常与 joint_power 配合，一个约束幅值，一个约束功率。
        self.rewards.joint_torques_l2.weight = -2.5e-5
        # 关节速度 L2 惩罚，抑制关节甩动
        self.rewards.joint_vel_l2.weight = -1.0e-5
        # 关节加速度 L2 惩罚，鼓励动作更平滑、降低冲击。
        self.rewards.joint_acc_l2.weight = -1.0e-5
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])
        # 关节限位惩罚: 接近/触发关节上下限时强惩罚，防止打限位。
        self.rewards.joint_pos_limits.weight = -5.0
        # 关节速度上限惩罚，当前关闭（可在硬件部署前再打开做保守化）。
        self.rewards.joint_vel_limits.weight = 0
        # 关节功率惩罚（约束机械功输出），降低发热与电池消耗。
        self.rewards.joint_power.weight = -2e-5
        # 静止命令下站立惩罚项（鼓励“该停就停”）。
        # 权重绝对值越大，零速命令时越倾向快速收敛到稳态。
        self.rewards.stand_still.weight = -8.0
        # 关节位置正则惩罚（通常相对默认姿态/安全姿态），抑制异常构型。
        self.rewards.joint_pos_penalty.weight = -2.5
        # 镜像对称惩罚: 约束对角腿运动统计相近，减少“偏腿”步态。
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.func = dog2_rewards.joint_mirror
        # 指定镜像关节对:
        # - FR 对 RL
        # - FL 对 RR
        # 有助于形成更自然的对角步态模式。
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        # Action penalties
        # 动作变化率惩罚，抑制相邻时刻动作突变，提升控制平滑性与可部署性。
        self.rewards.action_rate_l2.weight = -0.7

        # Contact sensor
        # 非足端接触持续扣分，包括机身严重倾斜或翻倒后的触地。
        self.rewards.undesired_contacts.func = dog2_rewards.undesired_contacts_unconditional
        self.rewards.undesired_contacts.weight = -20.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        # 足端接触力惩罚，避免落脚冲击过大。
        # 过大可能导致“轻触地”倾向，影响抓地与推进效率。
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        # 线速度追踪主奖励（xy 平面，指数型）。
        # 常为 locomotion 核心驱动项，值越大越优先“跟得上命令”。
        self.rewards.track_lin_vel_xy_exp.weight = 30.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.6
        # 偏航角速度追踪奖励（绕 z 转向），支持转向命令执行。
        self.rewards.track_ang_vel_z_exp.weight = 10.0

        # Others
        # 足端腾空时间奖励: 鼓励形成明确摆动相，避免拖脚。
        self.rewards.feet_air_time.weight = 2.0
        self.rewards.feet_air_time.params["threshold"] = 0.2
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        # 腾空时间方差惩罚: 抑制四腿步态节律差异过大，提升步态均匀性。
        self.rewards.feet_air_time_variance.weight = -30.0
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        # 足接触奖励（可用于鼓励稳定支撑），当前关闭。
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        # 无速度命令时的足接触奖励: 鼓励静止时脚不乱抬，站姿更稳。
        self.rewards.feet_contact_without_cmd.weight = 0.5
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        # 绊脚/碰撞惩罚，当前关闭（可按地形难度逐步启用）。
        self.rewards.feet_stumble.weight = 0.0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        # 足端滑动惩罚: 脚着地后相对地面滑移越大，惩罚越大。
        self.rewards.feet_slide.weight = -10.0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        # 摆动足端的世界高度目标。
        self.rewards.feet_height.weight = -20.0
        self.rewards.feet_height.params["target_height"] = 0.15
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        # 机身目标高度 0.30 m、摆动足端目标高度 0.15 m，对应机身坐标系下 -0.15 m。
        self.rewards.feet_height_body.weight = -5.0
        self.rewards.feet_height_body.params["target_height"] = -0.15
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        # 步态同步奖励: 鼓励对角腿成对同步（trot 风格）。
        self.rewards.feet_gait.weight = 3.0
        self.rewards.feet_gait.params["std"] = 0.05
        self.rewards.feet_gait.params["max_err"] = 0.3
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        # 机身“向上”姿态奖励（保持重力反方向对齐），提升整体直立稳定性。
        self.rewards.upward.weight = 5.0

        self.rewards.feet_distance_y_exp.weight = 0.8
        self.rewards.feet_distance_y_exp.params["stance_width"] = 0.34
        self.rewards.feet_distance_y_exp.params["std"] = 0.15
        self.rewards.feet_distance_y_exp.params["asset_cfg"].body_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

        # 将权重为0的奖励项禁用，减少无效计算与配置噪声
        if self.__class__.__name__ == "ATDogDog2RoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations 终止条件------------------------------
        self.terminations.illegal_contact = None

        # ------------------------------Curriculums 课程学习------------------------------
        # self.curriculum.command_levels.params["range_multiplier"] = (0.2, 1.0)
        # 关闭命令课程，直接使用固定命令范围
        self.curriculum.command_levels = None

        # ------------------------------Commands 命令范围------------------------------
        # 如需限制速度指令范围，可取消下方注释并按需调整
        # self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
