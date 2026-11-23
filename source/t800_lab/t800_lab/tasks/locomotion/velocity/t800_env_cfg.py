"""T800 的速度跟踪环境配置（G1/H1 官方 reward 风格）。

======================================================================
这份配置改了什么
======================================================================
奖励不再自己手调，而是照搬 Isaac Lab 官方 Unitree G1/H1 人形配置：
    - 摔倒惩罚 -200（非常重）
    - 线速度/角速度跟踪用 yaw frame / world frame
    - feet_air_time_positive_biped：只奖励单脚支撑时的步幅，短步不会被扣分
    - 髋/臂/躯干的关节偏离惩罚
    - 踝关节限位惩罚
    - 关闭推力、质量、质心随机化，reset 关节不缩放
这和之前那套“自定义 foot_clearance + 单脚/双支撑惩罚”完全不同，
更接近官方能训出人形步态的配置。

任务 id 仍然是 Isaac-T800-Locomotion-v0。
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

# 官方双足步态奖励（G1/H1 用的就是它）
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from t800_lab.assets.t800 import T800_CFG

# 只让策略控制 12 个腿关节；手臂/躯干/头保持默认角。
# 官方 G1/H1 控制全身，但 T800 手臂对早期平衡干扰大，先只控腿。
LEG_JOINTS = [
    "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L", "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R", "J07_HIP_ROLL_R", "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
]

# T800 的脚底接触在 LINK_FOOT_L/R（之前调试确认过）
FEET = ["LINK_FOOT_L", "LINK_FOOT_R"]


@configclass
class T800LocomotionEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ==============================================================
        # 1. 机器人
        # ==============================================================
        self.scene.robot = T800_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ==============================================================
        # 2. 平地，不用高度扫描
        # ==============================================================
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            terrain_generator=None,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )
        if hasattr(self.scene, "height_scanner"):
            self.scene.height_scanner = None
        if hasattr(self.observations.policy, "height_scan"):
            self.observations.policy.height_scan = None
        if hasattr(self.observations, "critic") and hasattr(self.observations.critic, "height_scan"):
            self.observations.critic.height_scan = None
        if hasattr(self.curriculum, "terrain_levels"):
            self.curriculum.terrain_levels = None

        # ==============================================================
        # 3. 速度指令（照搬 G1/H1）
        # ==============================================================
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # 官方默认 heading_command=True，保留；不需要时再关
        self.commands.base_velocity.rel_standing_envs = 0.02

        # ==============================================================
        # 4. 动作
        # ==============================================================
        # 官方人形 action scale 是 0.25
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.use_default_offset = True
        self.actions.joint_pos.joint_names = LEG_JOINTS

        # ==============================================================
        # 5. 奖励：官方 G1/H1 风格
        # ==============================================================
        # 摔倒惩罚非常重：-200
        self.rewards.termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

        # 线速度跟踪：用 yaw frame，官方 G1/H1 都是这个
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_lin_vel_xy_exp.params = {
            "command_name": "base_velocity",
            "std": 0.5,
        }

        # 角速度跟踪：用 world frame
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_world_exp
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.params = {
            "command_name": "base_velocity",
            "std": 0.5,
        }

        # 双足步态奖励：单脚支撑 + command>0.1 才有分，短步不会被扣
        self.rewards.feet_air_time.func = mdp.feet_air_time_positive_biped
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time.params = {
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "threshold": 0.4,
        }

        # 足端滑动惩罚（基类没有这项，新建）
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.1,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
                "asset_cfg": SceneEntityCfg("robot", body_names=FEET),
            },
        )

        # 姿态/平滑/省力
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.lin_vel_z_l2 = None
        self.rewards.undesired_contacts = None
        if hasattr(self.rewards, "base_height_l2"):
            self.rewards.base_height_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.ang_vel_xy_l2.weight = -0.05

        # 力矩/加速度惩罚只作用于腿
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params = {
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*HIP_.*", ".*KNEE_.*", ".*ANKLE_.*"]
            )
        }
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*HIP_.*", ".*KNEE_.*"])
        }

        # 踝关节限位
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.dof_pos_limits.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*ANKLE_.*"])
        }

        # 关节偏离惩罚：髋 yaw/roll、手臂、躯干
        self.rewards.joint_deviation_hip = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HIP_YAW.*", ".*HIP_ROLL.*"])},
        )
        self.rewards.joint_deviation_arms = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*SHOULDER.*", ".*ELBOW.*"])},
        )
        self.rewards.joint_deviation_torso = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*TORSO.*"])},
        )

        # ==============================================================
        # 6. 随机化：照搬 G1/H1，训练早期先关掉
        # ==============================================================
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["LINK_BASE"]

        # ==============================================================
        # 7. 终止条件：基座着地
        # ==============================================================
        for term_name in ("illegal_contact", "base_contact"):
            if hasattr(self.terminations, term_name):
                term = getattr(self.terminations, term_name)
                if "sensor_cfg" in term.params:
                    term.params["sensor_cfg"].body_names = ["LINK_BASE"]
                if "threshold" in term.params:
                    term.params["threshold"] = 1.0
        if hasattr(self.terminations, "bad_orientation"):
            self.terminations.bad_orientation.params["limit_angle"] = 0.8

        self.episode_length_s = 20.0
