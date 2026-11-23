# ==============================================================================
# 本文件用到的库（小例子见 docs/LIBRARIES.md）
#   configclass: 配置类装饰器
#   ManagerBasedRLEnvCfg: 管理器式 RL 环境
#   ObservationGroupCfg / ObsTerm: 观测组/观测项
#   RewTerm / DoneTerm / EventTerm: 奖励/终止/事件
#   ContactSensorCfg: 接触传感器
# ==============================================================================

"""T800 参考步态模仿环境（Mimic）。

==============================================================================
先用大白话解释：这个任务在干什么
==============================================================================
我们不再让机器人从零“自己发现走路”，而是：
    1. 先用 MNN 走路策略录一段参考步态（walk_ref.npz）
    2. 让 Isaac Lab 里的策略跟着参考走
    3. 策略只输出“在参考上的小修正”（residual）

比喻：
    参考轨迹 = 老师写好的草稿
    策略     = 学生，只需要在草稿上改几个字
    PPO      = 老师，根据改得好不好给分

==============================================================================
这个文件包含什么
==============================================================================
    MimicSceneCfg      : 场景里有什么（机器人、地面、接触传感器、灯）
    PolicyCfg          : 策略能看到哪些观测
    ActionsCfg         : 动作怎么定义（residual 关节位置）
    RewardsCfg         : 什么行为加分，什么行为扣分
    TerminationsCfg    : 什么时候结束回合（摔倒、膝盖着地、超时）
    EventsCfg          : reset 时怎么摆机器人
    T800MimicEnvCfg    : 把上面这些组装成一个完整环境

==============================================================================
关键名词（详细解释见 docs/GLOSSARY.md）
==============================================================================
USD          : Isaac Sim 使用的场景文件格式
Articulation : 带关节的机器人
DoF          : 自由度，T800 有 25 个驱动关节 + 1 个浮动基座
PD 控制      : 力矩 = kp × (目标角 - 当前角) - kd × 当前角速度
观测         : 策略能看到的数字
动作         : 策略输出的数字
奖励         : 每一步的分数，强化学习唯一的老师
Episode      : 从 reset 到摔倒/超时的一整段过程
"""

from pathlib import Path

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from t800_lab.assets.t800 import T800_MIMIC_CFG
from t800_lab.tasks.locomotion.mimic.mdp import (
    RefJointPositionActionCfg,
    ref_joint_pos,
    ref_phase,
    track_base_vel,
    track_ref_joint_l2,
    track_ref_joint_pos,
    reset_joints_to_reference,
)

# 参考轨迹的位置：
# 当前文件在 t800_lab/tasks/locomotion/mimic/ 下，
# parents[3] 就是 t800_lab 包目录。
REF_PATH = str(Path(__file__).resolve().parents[3] / "assets/reference/walk_ref.npz")


@configclass
class MimicSceneCfg(InteractiveSceneCfg):
    """场景配置：机器人、地面、接触传感器、灯。

    InteractiveSceneCfg 是 Isaac Lab 的“场景模板”。
    并行环境里，每个环境都会复制一份这里的配置。
    """

    # 机器人。replace(prim_path=...) 指定它在每个环境里的路径。
    # {ENV_REGEX_NS} 会展开成 /World/envs/env_0 等。
    robot: ArticulationCfg = T800_MIMIC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 平地。terrain_type="plane" 表示无限大的平面。
    terrain = TerrainImporterCfg(
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

    # 接触传感器：测量脚/膝盖/肩膀有没有碰到地面。
    # track_air_time=True 会记录“离地多久”，用于步态奖励。
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # 灯光。没有灯，画面会是黑的；headless 训练不影响物理。
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            color=(1.0, 1.0, 1.0),
        ),
    )


@configclass
class PolicyCfg(ObservationGroupCfg):
    """策略观测：策略每一步能看到的数字。

    观测 = 机器人的“感官”。观测里没有的东西，策略不可能知道。
    """

    # 基座线速度 (3)：前后、左右、上下
    base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel)
    # 基座角速度 (3)：绕 x/y/z 转多快
    base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel)
    # 投影重力 (3)：机身相对地面有多歪
    projected_gravity = ObsTerm(func=base_mdp.projected_gravity)
    # 25 个关节角，相对默认角
    joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
    # 25 个关节速度
    joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)
    # 上一步动作 (25)
    actions = ObsTerm(func=base_mdp.last_action)
    # 当前参考关节角 (25)：老师草稿
    ref_joint_pos = ObsTerm(func=ref_joint_pos)
    # 步态相位 (2)：当前走到周期的哪里
    ref_phase = ObsTerm(func=ref_phase)

    def __post_init__(self):
        # 训练时是否给观测加噪声。
        # 这里先不加，方便调试；真机部署前应该加噪声提高鲁棒性。
        self.enable_corruption = False


@configclass
class ObservationsCfg:
    """观测组集合。目前只有一个 policy 组。"""

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """动作配置。

    动作不是绝对关节角，而是：
        target = 参考关节角 + action * 0.25
    这就是 residual action（残差动作）。
    """

    joint_pos: RefJointPositionActionCfg = RefJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],   # 25 个关节全控
        scale=0.25,           # 最多在参考上修正 0.25 弧度
        ref_path=REF_PATH,
    )


@configclass
class RewardsCfg:
    """奖励配置。

    奖励 = 强化学习唯一的老师。
    这里只围绕三件事：
        1. 关节角跟参考像不像
        2. 有没有往前走
        3. 有没有摔倒/跪地/乱晃
    """

    # 跟踪参考关节角：指数奖励，误差越小越接近 1
    track_joint = RewTerm(func=track_ref_joint_pos, weight=5.0, params={"std": 0.5})
    # 参考关节角的 L2 误差：给连续梯度，防止指数奖励在误差大时没梯度
    track_joint_l2 = RewTerm(func=track_ref_joint_l2, weight=-1.0)

    # 跟踪前进速度：目标 0.5 m/s
    track_vel = RewTerm(func=track_base_vel, weight=2.0, params={"target": 0.5, "std": 0.5})

    # 机身保持水平
    flat_orientation = RewTerm(func=base_mdp.flat_orientation_l2, weight=-1.0)
    # 动作变化不要太剧烈
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    # 关节力矩不要太大
    joint_torques = RewTerm(func=base_mdp.joint_torques_l2, weight=-1e-6)

    # 非脚着地惩罚：膝盖、肩膀、髋、基座着地都扣分
    undesired_contacts = RewTerm(
        func=base_mdp.undesired_contacts,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["LINK_BASE", "LINK_KNEE.*", "LINK_SHOULDER.*", "LINK_HIP.*"],
            ),
            "threshold": 1.0,
        },
    )

    # 摔倒惩罚
    termination = RewTerm(func=base_mdp.is_terminated, weight=-20.0)


@configclass
class TerminationsCfg:
    """终止条件：什么时候结束这个回合。"""

    # 超时：到 10 秒正常结束，不算失败
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)

    # 基座着地：摔倒
    base_contact = DoneTerm(
        func=base_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["LINK_BASE"]),
            "threshold": 1.0,
        },
    )

    # 膝盖/肩膀着地：也算失败，防止“跪着撑到超时”
    non_foot_contact = DoneTerm(
        func=base_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["LINK_KNEE.*", "LINK_SHOULDER.*"]
            ),
            "threshold": 1.0,
        },
    )


@configclass
class EventsCfg:
    """reset 事件：每个回合开始时怎么摆机器人。"""

    # 基座 reset：位置小幅随机，速度全 0
    reset_base = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.2, 0.2)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # 关节 reset：直接写成参考轨迹第 0 帧
    reset_joints = EventTerm(
        func=reset_joints_to_reference,
        mode="reset",
        params={},
    )


@configclass
class T800MimicEnvCfg(ManagerBasedRLEnvCfg):
    """把场景、观测、动作、奖励、终止、事件组装成完整环境。"""

    scene: MimicSceneCfg = MimicSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        super().__post_init__()
        # decimation=4 表示：物理每 0.005 秒走一步，
        # 每 4 步才让策略输出一次动作。
        # 所以控制频率 = 1 / (0.005 × 4) = 50 Hz。
        self.decimation = 4
        # 每个回合 10 秒，正好对应 10 秒参考轨迹
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
