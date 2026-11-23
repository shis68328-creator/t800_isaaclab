# ==============================================================================
# 本文件用到的库（小例子见 docs/LIBRARIES.md）
#   gymnasium.register: 把 Mimic 任务注册到 Gym
#       gym.register(id="Isaac-T800-Mimic-v0", entry_point=..., kwargs={...})
# ==============================================================================

"""T800 参考步态模仿任务。"""

import gymnasium as gym

from . import mimic_env_cfg  # noqa: F401
from .agents import rsl_rl_ppo_cfg  # noqa: F401

gym.register(
    id="Isaac-T800-Mimic-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mimic_env_cfg:T800MimicEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:T800MimicPPORunnerCfg",
    },
)
