"""参考步态 residual 动作项。

target_joint = reference_joint + action * scale

这样策略一开始只需要输出接近 0 的动作，就能跟着参考步态走；
PPO 只负责微调平衡，不用从零探索步态。
"""

import math
from pathlib import Path

import numpy as np
import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class RefJointPositionAction(ActionTerm):
    cfg: "RefJointPositionActionCfg"

    def __init__(self, cfg: "RefJointPositionActionCfg", env):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)

        # 参考轨迹：npz 里是 MuJoCo 关节顺序，按名字映射到 Isaac 顺序
        data = np.load(cfg.ref_path)
        ref_names = [str(n) for n in data["joint_names"]]
        order = [ref_names.index(name) for name in self._joint_names]
        q = data["joint_pos"][:, order].astype(np.float32)

        # npz 是 100 Hz，控制是 50 Hz，隔点采样
        step = max(1, int(round(env.step_dt / float(data["dt"]))))
        self._ref = torch.tensor(q[::step], device=env.device, dtype=torch.float32)
        self._ref_dt = step * float(data["dt"])
        self._ref_len = self._ref.shape[0]

        self._raw_actions = torch.zeros(
            (env.num_envs, len(self._joint_ids)), device=env.device, dtype=torch.float32
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return len(self._joint_ids)

    @property
    def joint_ids(self):
        return self._joint_ids

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _ref_index(self) -> torch.Tensor:
        t = (self._env.episode_length_buf.to(torch.float32) * self._env.step_dt) / self._ref_dt
        return t.long() % self._ref_len

    def get_reference(self) -> torch.Tensor:
        """返回当前时刻的参考关节角，形状 (num_envs, num_joints)。"""
        return self._ref[self._ref_index()]

    def get_phase(self) -> torch.Tensor:
        """返回 [sin(phase), cos(phase)]，形状 (num_envs, 2)。"""
        t = self._ref_index().to(torch.float32)
        angle = 2.0 * math.pi * t / self._ref_len
        return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions = self.get_reference() + actions * self.cfg.scale

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)


@configclass
class RefJointPositionActionCfg(ActionTermCfg):
    class_type: type = RefJointPositionAction
    joint_names: list[str] = [".*"]
    scale: float = 0.25
    # 参考 npz 的路径，在 env cfg 里填绝对路径
    ref_path: str = ""
