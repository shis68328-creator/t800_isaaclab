"""T800 的速度跟踪环境配置（走 + 跑）。

======================================================================
这份文件在干什么
======================================================================
Isaac Lab 的 ManagerBasedRLEnv 用“配置类”描述环境：
    - 场景里有什么（地面、机器人、传感器）
    - 观测有哪些、动作是什么
    - 奖励怎么算、什么时候回合结束
    - 课程学习、随机化事件

我们继承 Isaac Lab 官方自带的
    isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg
里的 LocomotionVelocityRoughEnvCfg（崎岖地形速度跟踪），
然后把机器人换成 T800、地形改成平地、调奖励。

======================================================================
用到的包 / 类
======================================================================
sim_utils：
    Isaac Lab 对 Isaac Sim 的封装，这里用 RigidBodyMaterialCfg 配地面摩擦。
TerrainImporterCfg：
    地形配置。terrain_type="plane" 表示一块无限大的平地。
configclass：
    Isaac Lab 的配置类装饰器，类似 dataclass，但支持嵌套配置、
    to_dict()、replace()，Isaac Lab 的环境配置都靠它。
LocomotionVelocityRoughEnvCfg：
    Isaac Lab 官方“速度跟踪”环境的基类。它已经写好了：
      - 场景、观测、动作、奖励、终止、事件、课程
    我们只覆盖需要改的部分。
SceneEntityCfg / RewTerm / ObsTerm：
    基类里用到的“场景实体引用 / 奖励项 / 观测项”，
    我们通过 .params 修改具体参数（比如足端 body 名字）。
"""

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from t800_lab.assets.t800 import T800_CFG


def _set_weight(rewards, name: str, value: float):
    """安全地修改某个奖励项的权重。

    Isaac Lab 不同版本里奖励项名字可能略有差异，直接
    `self.rewards.feet_air_time.weight = ...` 在缺项时会报 AttributeError。
    这里先 hasattr 判断，缺了就跳过，保证配置在多个版本上都能加载。
    """
    if hasattr(rewards, name):
        getattr(rewards, name).weight = value


@configclass
class T800LocomotionEnvCfg(LocomotionVelocityRoughEnvCfg):
    """T800 平地速度跟踪环境。

    继承基类后，只需要在 __post_init__ 里覆盖：
      - 机器人
      - 地形 / 高度扫描
      - 速度指令范围
      - 动作缩放
      - 奖励权重和足端 body
      - 终止条件
    """

    def __post_init__(self):
        # 必须先调用父类，让基类把所有默认配置建好
        super().__post_init__()

        # ==============================================================
        # 1. 机器人
        # ==============================================================
        # 基类里的 scene.robot 是一个占位（MISSING），必须替换成 T800。
        # replace(prim_path=...) 会复制一份配置，并指定机器人在每个环境里的路径。
        # {ENV_REGEX_NS} 是 Isaac Lab 的占位符，会展开成 /World/envs/env_0 等。
        self.scene.robot = T800_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ==============================================================
        # 2. 地形：改成平地
        # ==============================================================
        # 基类默认是崎岖地形生成器。这里换成一块无限大的平面。
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            terrain_generator=None,
            collision_group=-1,
            # 地面摩擦：静摩擦 1.0、动摩擦 1.0，脚不容易打滑
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )

        # 平地不需要高度扫描传感器。
        # 用 hasattr 是因为不同 Isaac Lab 版本字段可能不同。
        if hasattr(self.scene, "height_scanner"):
            self.scene.height_scanner = None
        if hasattr(self.observations.policy, "height_scan"):
            self.observations.policy.height_scan = None
        if hasattr(self.observations.critic, "height_scan"):
            self.observations.critic.height_scan = None
        if hasattr(self.curriculum, "terrain_levels"):
            self.curriculum.terrain_levels = None

        # ==============================================================
        # 3. 速度指令
        # ==============================================================
        # 每个 episode 会随机采样一个目标速度，写进观测的 command 项。
        # vx 范围 -0.5 ~ 2.5：负值后退，0 站立，1 左右走路，2 以上跑。
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # ==============================================================
        # 4. 动作
        # ==============================================================
        # 动作用“关节位置增量”：策略输出 [-1,1]，乘 scale 加到默认关节角上。
        # scale=0.25 表示一个动作单位最多让关节偏 0.25 rad。
        self.actions.joint_pos.scale = 0.25
        # 以 init_state 的默认关节角为基准（而不是 URDF 的零位）
        self.actions.joint_pos.use_default_offset = True

        # ==============================================================
        # 5. 奖励
        # ==============================================================
        # 速度跟踪：实际速度越接近命令速度，exp 奖励越接近 1
        _set_weight(self.rewards, "track_lin_vel_xy_exp", 3.0)
        _set_weight(self.rewards, "track_ang_vel_z_exp", 1.5)
        # 抑制上下跳、横滚/俯仰角速度
        _set_weight(self.rewards, "lin_vel_z_l2", -2.0)
        _set_weight(self.rewards, "ang_vel_xy_l2", -0.05)
        # 机身保持水平
        _set_weight(self.rewards, "flat_orientation_l2", -5.0)
        # 省力、平滑
        _set_weight(self.rewards, "joint_torques_l2", -2.5e-5)
        _set_weight(self.rewards, "joint_acc_l2", -1e-8)
        _set_weight(self.rewards, "action_rate_l2", -0.02)
        # 足端腾空时间：鼓励迈步；跑步时自然鼓励腾空
        _set_weight(self.rewards, "feet_air_time", 5.0)
        # 足端滑动：脚在地上蹭要扣分
        _set_weight(self.rewards, "feet_slide", -0.05)
        # 非足端着地（比如膝盖、基座）
        _set_weight(self.rewards, "undesired_contacts", -0.5)
        # 回合终止惩罚
        _set_weight(self.rewards, "is_terminated", -2.0)
        # 平地没有高度扫描，base_height_l2 依赖地形射线，先关掉
        _set_weight(self.rewards, "base_height_l2", 0.0)

        # ==============================================================
        # 6. 把奖励项绑定到 T800 的足端
        # ==============================================================
        # feet_air_time / feet_slide 在基类里默认引用 "foot" 之类的 body，
        # T800 的足端 body 名字是 LINK_FOOT_L / LINK_FOOT_R，要改过来。
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

        # ==============================================================
        # 7. 终止条件
        # ==============================================================
        # 基类里有 illegal_contact / base_contact（不同版本名字不同）。
        # 只要基座着地，就认为摔倒，结束本回合。
        for term_name in ("illegal_contact", "base_contact"):
            if hasattr(self.terminations, term_name):
                term = getattr(self.terminations, term_name)
                if "sensor_cfg" in term.params:
                    term.params["sensor_cfg"].body_names = ["LINK_BASE"]
                if "threshold" in term.params:
                    term.params["threshold"] = 1.0
        # 身体倾斜超过 0.8 rad（约 46°）也终止
        if hasattr(self.terminations, "bad_orientation"):
            self.terminations.bad_orientation.params["limit_angle"] = 0.8

        # ==============================================================
        # 8. 回合长度
        # ==============================================================
        self.episode_length_s = 20.0
