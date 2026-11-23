"""Mimic 任务的自定义奖励项。"""

import torch


def track_ref_joint_pos(env, std: float = 0.3) -> torch.Tensor:
    """跟踪参考关节角：误差越小奖励越接近 1。"""
    robot = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    ref = term.get_reference()
    err = torch.sum((robot.data.joint_pos - ref) ** 2, dim=1)
    return torch.exp(-err / (std ** 2))


def track_base_vel(env, target: float = 0.5, std: float = 0.5) -> torch.Tensor:
    """跟踪参考前进速度（基座 x 方向）。"""
    robot = env.scene["robot"]
    vx = robot.data.root_lin_vel_b[:, 0]
    return torch.exp(-((vx - target) ** 2) / (std ** 2))


def track_ref_joint_l2(env) -> torch.Tensor:
    """参考关节角的 L2 误差（作为负奖励，提供连续梯度）。"""
    robot = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    ref = term.get_reference()
    return torch.sum((robot.data.joint_pos - ref) ** 2, dim=1)
