"""把 T800 速度跟踪任务注册到 Gym。

======================================================================
gym.register 在干什么
======================================================================
Gym / Gymnasium 用“任务 id”管理环境：
    gym.make("Isaac-T800-Locomotion-v0", cfg=env_cfg)

必须先注册，make 才知道这个 id 对应哪个环境类、哪个配置。
register 的参数：
    id：任务字符串
    entry_point：环境类。Isaac Lab 用 ManagerBasedRLEnv
    kwargs：
        env_cfg_entry_point   -> 环境配置类
        rsl_rl_cfg_entry_point -> PPO 配置类

字符串路径格式：
    "模块.文件:类名"
比如：
    t800_lab.tasks.locomotion.velocity.t800_env_cfg:T800LocomotionEnvCfg
"""

import gymnasium as gym

# 导入配置模块，让类被 Python 加载
from . import t800_env_cfg  # noqa: F401
from .agents import rsl_rl_ppo_cfg  # noqa: F401

gym.register(
    id="Isaac-T800-Locomotion-v0",
    # Isaac Lab 的环境类，负责场景、观测、奖励、终止的管理
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    # 关闭 Gym 自带的参数检查，Isaac Lab 不需要
    disable_env_checker=True,
    kwargs={
        # 环境配置：T800 平地速度跟踪
        "env_cfg_entry_point": f"{__name__}.t800_env_cfg:T800LocomotionEnvCfg",
        # PPO 配置：网络结构、学习率、迭代次数
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:T800PPORunnerCfg",
    },
)
