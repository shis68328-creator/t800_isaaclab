"""T800 速度跟踪任务注册。"""

import gymnasium as gym

from . import t800_env_cfg  # noqa: F401
from .agents import rsl_rl_ppo_cfg  # noqa: F401

gym.register(
    id="Isaac-T800-Locomotion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.t800_env_cfg:T800LocomotionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:T800PPORunnerCfg",
    },
)
