"""T800 参考步态模仿环境（Mimic）。

动作：target = 参考关节角 + action * 0.25
观测：本体状态 + 当前参考关节角 + 步态相位
奖励：跟踪参考关节角 + 跟踪前进速度 + 姿态/平滑正则
终止：基座着地、超时
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

from t800_lab.assets.t800 import T800_CFG
from t800_lab.tasks.locomotion.mimic.mdp import (
    RefJointPositionActionCfg,
    ref_joint_pos,
    ref_phase,
    track_base_vel,
    track_ref_joint_pos,
)

# t800_lab/assets/reference/walk_ref.npz
REF_PATH = str(Path(__file__).resolve().parents[3] / "assets/reference/walk_ref.npz")


@configclass
class MimicSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = T800_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            color=(1.0, 1.0, 1.0),
        ),
    )


@configclass
class PolicyCfg(ObservationGroupCfg):
    base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=base_mdp.projected_gravity)
    joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)
    actions = ObsTerm(func=base_mdp.last_action)
    ref_joint_pos = ObsTerm(func=ref_joint_pos)
    ref_phase = ObsTerm(func=ref_phase)

    def __post_init__(self):
        self.enable_corruption = False


@configclass
class ObservationsCfg:
    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    joint_pos: RefJointPositionActionCfg = RefJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        ref_path=REF_PATH,
    )


@configclass
class RewardsCfg:
    # 跟踪参考关节角
    track_joint = RewTerm(func=track_ref_joint_pos, weight=3.0, params={"std": 0.3})
    # 跟踪参考前进速度
    track_vel = RewTerm(func=track_base_vel, weight=2.0, params={"target": 0.5, "std": 0.5})
    # 姿态/平滑/省力
    flat_orientation = RewTerm(func=base_mdp.flat_orientation_l2, weight=-1.0)
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    joint_torques = RewTerm(func=base_mdp.joint_torques_l2, weight=-1e-6)
    # 摔倒惩罚
    termination = RewTerm(func=base_mdp.is_terminated, weight=-20.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=base_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["LINK_BASE"]),
            "threshold": 1.0,
        },
    )


@configclass
class EventsCfg:
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

    reset_joints = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )


@configclass
class T800MimicEnvCfg(ManagerBasedRLEnvCfg):
    scene: MimicSceneCfg = MimicSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4       # 50 Hz 控制
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
