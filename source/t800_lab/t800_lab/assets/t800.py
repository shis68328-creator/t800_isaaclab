"""T800 在 Isaac Lab 中的 ArticulationCfg。

USD 文件由 scripts/convert_t800.py 从 URDF 生成，默认输出到
本文件同目录下的 t800.usd。
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# 转换脚本生成的 USD 路径
T800_USD_PATH = str(Path(__file__).resolve().parent / "t800.usd")

# 25 个关节的默认站立角（来自 SDK rl_walking_example）
DEFAULT_JOINT_POS = {
    "J00_HIP_PITCH_L": -0.06,
    "J01_HIP_ROLL_L": 0.0,
    "J02_HIP_YAW_L": 0.0,
    "J03_KNEE_PITCH_L": 0.12,
    "J04_ANKLE_PITCH_L": -0.06,
    "J05_ANKLE_ROLL_L": 0.0,
    "J06_HIP_PITCH_R": -0.06,
    "J07_HIP_ROLL_R": 0.0,
    "J08_HIP_YAW_R": 0.0,
    "J09_KNEE_PITCH_R": 0.12,
    "J10_ANKLE_PITCH_R": -0.06,
    "J11_ANKLE_ROLL_R": 0.0,
    "J12_TORSO_YAW": 0.0,
    "J13_SHOULDER_PITCH_L": 0.0,
    "J14_SHOULDER_ROLL_L": 0.15,
    "J15_SHOULDER_YAW_L": 0.0,
    "J16_ELBOW_PITCH_L": -0.25,
    "J17_ELBOW_YAW_L": 0.0,
    "J18_SHOULDER_PITCH_R": 0.0,
    "J19_SHOULDER_ROLL_R": -0.15,
    "J20_SHOULDER_YAW_R": 0.0,
    "J21_ELBOW_PITCH_R": -0.25,
    "J22_ELBOW_YAW_R": 0.0,
    "J23_HEAD_PITCH": 0.0,
    "J24_HEAD_YAW": 0.0,
}

T800_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=T800_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.03),
        joint_pos=DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        # 髋 pitch、膝 pitch：承重关节，增益高一点
        "hip_pitch_knee": ImplicitActuatorCfg(
            joint_names_expr=[".*HIP_PITCH.*", ".*KNEE_PITCH.*"],
            stiffness=180.0,
            damping=5.0,
        ),
        # 髋 roll / yaw
        "hip_roll_yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*HIP_ROLL.*", ".*HIP_YAW.*"],
            stiffness=100.0,
            damping=3.0,
        ),
        # 踝关节：软一点，方便脚掌贴地
        "ankle": ImplicitActuatorCfg(
            joint_names_expr=[".*ANKLE.*"],
            stiffness=40.0,
            damping=0.3,
        ),
        # 手臂
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*SHOULDER.*", ".*ELBOW.*"],
            stiffness=60.0,
            damping=0.5,
        ),
        # 躯干和头
        "torso_head": ImplicitActuatorCfg(
            joint_names_expr=[".*TORSO.*", ".*HEAD.*"],
            stiffness=100.0,
            damping=1.0,
        ),
    },
)
