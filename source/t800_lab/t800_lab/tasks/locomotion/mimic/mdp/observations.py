"""Mimic 任务的自定义观测项。"""

import torch


def ref_joint_pos(env) -> torch.Tensor:
    """当前参考关节角，形状 (num_envs, 25)。"""
    term = env.action_manager.get_term("joint_pos")
    return term.get_reference()


def ref_phase(env) -> torch.Tensor:
    """参考步态相位 [sin, cos]，形状 (num_envs, 2)。"""
    term = env.action_manager.get_term("joint_pos")
    return term.get_phase()
