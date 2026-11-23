# ==============================================================================
# 本文件用到的库（小例子见 docs/LIBRARIES.md）
#   torch: exp / sum / 张量运算
#   robot.data.joint_pos: 当前关节角
#   robot.data.root_lin_vel_b: 基座线速度
# ==============================================================================

"""Mimic 任务的自定义奖励项。

==============================================================================
奖励是什么
==============================================================================
奖励是强化学习唯一的“老师”。策略通过奖励判断：
    这个动作是好还是坏？

Mimic 任务的奖励核心只有一句话：
    “关节角越接近参考，分越高；能往前走，分更高；摔倒/跪地，扣分。”
"""

import torch


def track_ref_joint_pos(env, std: float = 0.3) -> torch.Tensor:
    """跟踪参考关节角（指数奖励）。

    参数：
        env：环境
        std：容忍度。std 越小，要求越严格。

    返回：
        形状 (num_envs,)，范围 (0, 1]：
            误差 0      -> 1.0
            误差很大    -> 接近 0

    公式：
        exp(-误差平方 / std²)

    为什么用 exp：
        它把“误差”变成 0~1 的分数，误差越大分数越接近 0，
        而且处处可导，适合梯度下降。
    """
    robot = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    ref = term.get_reference()

    # 25 个关节的误差平方和
    err = torch.sum((robot.data.joint_pos - ref) ** 2, dim=1)
    return torch.exp(-err / (std ** 2))


def track_ref_joint_l2(env) -> torch.Tensor:
    """参考关节角的 L2 误差（作为负奖励）。

    指数奖励在误差很大时几乎为 0，梯度也几乎为 0，学不动。
    L2 误差是“误差平方和”，误差越大惩罚越大，
    可以给策略一个连续的“往回拉”的力。

    返回：
        形状 (num_envs,)，误差平方和，越小越好。
    """
    robot = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    ref = term.get_reference()
    return torch.sum((robot.data.joint_pos - ref) ** 2, dim=1)


def track_base_vel(env, target: float = 0.5, std: float = 0.5) -> torch.Tensor:
    """跟踪前进速度。

    参数：
        target：希望的前进速度，单位 m/s
        std：容忍度

    返回：
        形状 (num_envs,)，速度越接近 target，越接近 1。

    注意：
        这里只看基座 x 方向速度，不看横向和转向。
        所以训练出来的策略可能走不直。
    """
    robot = env.scene["robot"]
    vx = robot.data.root_lin_vel_b[:, 0]
    return torch.exp(-((vx - target) ** 2) / (std ** 2))
