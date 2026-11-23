"""T800 速度跟踪（走/跑）环境配置。

继承 Isaac Lab 自带的 LocomotionVelocityRoughEnvCfg，改成：
  - T800 机器人
  - 平地
  - 去掉高度扫描
  - 速度指令范围 -0.5 ~ 2.5 m/s（低速走、高速跑）
  - 足端接触/滑动/腾空时间奖励

任务 id：Isaac-T800-Locomotion-v0
"""

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from t800_lab.assets.t800 import T800_CFG


def _set_weight(rewards, name: str, value: float):
    """安全地改奖励权重：不同 Isaac Lab 版本可能没有某一项。"""
    if hasattr(rewards, name):
        getattr(rewards, name).weight = value


@configclass
class T800LocomotionEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ---- 机器人 ----
        self.scene.robot = T800_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ---- 平地，不用高度扫描 ----
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
        if hasattr(self.observations.critic, "height_scan"):
            self.observations.critic.height_scan = None
        if hasattr(self.curriculum, "terrain_levels"):
            self.curriculum.terrain_levels = None

        # ---- 速度指令：负速度后退，高速就是跑 ----
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # ---- 动作：关节位置增量 ----
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.use_default_offset = True

        # ---- 奖励权重 ----
        _set_weight(self.rewards, "track_lin_vel_xy_exp", 3.0)
        _set_weight(self.rewards, "track_ang_vel_z_exp", 1.5)
        _set_weight(self.rewards, "lin_vel_z_l2", -2.0)
        _set_weight(self.rewards, "ang_vel_xy_l2", -0.05)
        _set_weight(self.rewards, "flat_orientation_l2", -5.0)
        _set_weight(self.rewards, "joint_torques_l2", -2.5e-5)
        _set_weight(self.rewards, "joint_acc_l2", -1e-8)
        _set_weight(self.rewards, "action_rate_l2", -0.02)
        _set_weight(self.rewards, "feet_air_time", 5.0)
        _set_weight(self.rewards, "feet_slide", -0.05)
        _set_weight(self.rewards, "undesired_contacts", -0.5)
        _set_weight(self.rewards, "is_terminated", -2.0)
        # 不用高度扫描，base_height 这项关掉，避免依赖地形射线
        _set_weight(self.rewards, "base_height_l2", 0.0)

        # 足端 body 名字
        feet = ["LINK_FOOT_L", "LINK_FOOT_R"]
        if hasattr(self.rewards, "feet_air_time"):
            self.rewards.feet_air_time.params["threshold"] = 0.5
            if "sensor_cfg" in self.rewards.feet_air_time.params:
                self.rewards.feet_air_time.params["sensor_cfg"].body_names = feet
        if hasattr(self.rewards, "feet_slide"):
            if "sensor_cfg" in self.rewards.feet_slide.params:
                self.rewards.feet_slide.params["sensor_cfg"].body_names = feet
            if "asset_cfg" in self.rewards.feet_slide.params:
                self.rewards.feet_slide.params["asset_cfg"].body_names = feet
        if hasattr(self.rewards, "undesired_contacts"):
            if "sensor_cfg" in self.rewards.undesired_contacts.params:
                self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["LINK_BASE"]

        # ---- 终止条件：基座着地、身体太歪 ----
        for term_name in ("illegal_contact", "base_contact"):
            if hasattr(self.terminations, term_name):
                term = getattr(self.terminations, term_name)
                if "sensor_cfg" in term.params:
                    term.params["sensor_cfg"].body_names = ["LINK_BASE"]
                if "threshold" in term.params:
                    term.params["threshold"] = 1.0
        if hasattr(self.terminations, "bad_orientation"):
            self.terminations.bad_orientation.params["limit_angle"] = 0.8

        # 每个 episode 20 秒
        self.episode_length_s = 20.0
