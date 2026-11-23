"""T800 在 Isaac Lab 里的“机器人配置”。

======================================================================
先理解三个词
======================================================================
【USD】
    Universal Scene Description，NVIDIA Omniverse / Isaac Sim 用的场景格式。
    Isaac Lab 不能直接加载 MuJoCo 的 MJCF 或 URDF，必须先把 URDF 转成 .usd。
    这个文件里的 T800_USD_PATH 就是 scripts/convert_t800.py 生成的那个 USD。

【Articulation（关节体）】
    Isaac Lab 对“带关节的机器人”的称呼。一个 Articulation 由很多刚体（link）
    和关节（joint）组成，可以读取关节角、设置关节目标、施加力矩。

【ArticulationCfg】
    描述“怎么把机器人加载进仿真”的配置类，包含：
      - spawn：USD 文件路径、碰撞、重力、求解器迭代次数
      - init_state：初始位置、初始关节角、初始速度
      - actuators：每个关节用什么 PD 增益、力矩上限
    它本身不做仿真，只是“一份配置”；Isaac Lab 的 ManagerBasedRLEnv
    会读取它并在场景里创建机器人。

======================================================================
用到的包
======================================================================
sim_utils：
    Isaac Lab 对 Isaac Sim 的封装，提供 UsdFileCfg、RigidBodyPropertiesCfg、
    ArticulationRootPropertiesCfg 等“生成/物理属性”配置。
ImplicitActuatorCfg：
    Isaac Lab 的“隐式 PD 执行器”配置。
    隐式 = 由 PhysX 底层直接算 PD，不需要 Python 每个控制步算力矩。
    stiffness 对应 kp，damping 对应 kd。
ArticulationCfg：
    机器人整体配置类（见上）。
Path：
    pathlib，用来拼 USD 的绝对路径。
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# ----------------------------------------------------------------------
# 1. USD 路径
# ----------------------------------------------------------------------
# __file__ 是当前文件；resolve() 转绝对路径；parent 是 assets 目录。
# 转换脚本会把 t800.usd 生成到同目录。
T800_USD_PATH = str(Path(__file__).resolve().parent / "t800.usd")

# ----------------------------------------------------------------------
# 2. 25 个关节的默认站立角（弧度）
# ----------------------------------------------------------------------
# 这是 Isaac Lab 的“初始关节角”，不是动作。训练开始时机器人会摆成这个姿势。
# 数值来自 SDK 的 rl_walking_example，能让 T800 自然站立。
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

# ----------------------------------------------------------------------
# 3. 机器人整体配置
# ----------------------------------------------------------------------
# T800_CFG 是一个 ArticulationCfg 对象，后面 t800_env_cfg.py 会把它放进场景：
#     self.scene.robot = T800_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
# ISaac Lab 会在每个并行环境里复制一份机器人。
T800_CFG = ArticulationCfg(
    # ---- 3.1 从 USD 文件生成机器人 ----
    spawn=sim_utils.UsdFileCfg(
        usd_path=T800_USD_PATH,   # 转换脚本生成的 USD
        # 打开接触传感器：足端接触奖励、腾空时间都要用它
        activate_contact_sensors=True,
        # 刚体物理属性
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,        # 受重力影响
            retain_accelerations=False,   # 不保留加速度（省内存）
            linear_damping=0.0,           # 线性阻尼
            angular_damping=0.0,          # 角阻尼
            max_linear_velocity=1000.0,   # 最大线速度
            max_angular_velocity=1000.0,  # 最大角速度
            max_depenetration_velocity=1.0,  # 穿模后最多用多快的速度弹开
        ),
        # 关节体属性
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,       # 关闭自碰撞（提高性能）
            solver_position_iteration_count=4,   # 位置约束迭代次数
            solver_velocity_iteration_count=1,   # 速度约束迭代次数
        ),
    ),

    # ---- 3.2 初始状态 ----
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.03),  # 基座初始位置：离地 1.03 m
        joint_pos=DEFAULT_JOINT_POS,  # 25 个关节的初始角
        joint_vel={".*": 0.0},        # 所有关节初始速度为 0
    ),

    # 关节软限位系数：0.95 表示只用关节范围的 95%，避免顶到硬限位
    soft_joint_pos_limit_factor=0.95,

    # ---- 3.3 执行器（关节 PD 控制） ----
    # 每个 ImplicitActuatorCfg 是一组关节：
    #   joint_names_expr：正则表达式，匹配关节名
    #   stiffness：位置增益 kp
    #   damping：速度增益 kd
    # Isaac Lab 用 PhysX 的隐式 PD，不需要我们在 Python 里手写 tau。
    actuators={
        # 髋 pitch、膝 pitch：承重关节，增益高
        "hip_pitch_knee": ImplicitActuatorCfg(
            joint_names_expr=[".*HIP_PITCH.*", ".*KNEE_PITCH.*"],
            stiffness=180.0,
            damping=5.0,
        ),
        # 髋 roll / yaw：中等增益
        "hip_roll_yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*HIP_ROLL.*", ".*HIP_YAW.*"],
            stiffness=100.0,
            damping=3.0,
        ),
        # 踝关节：软一点，脚掌更容易贴地
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
