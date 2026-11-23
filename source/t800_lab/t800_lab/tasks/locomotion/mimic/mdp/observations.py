"""Mimic 任务的自定义观测项。

==============================================================================
观测是什么
==============================================================================
观测就是“策略能看到的数字”。策略只能根据观测做决定，
观测里没有的信息，它不可能知道。

对 Mimic 任务来说，最关键的两个观测是：
    1. ref_joint_pos：当前参考关节角（老师草稿）
    2. ref_phase：当前走到步态的哪个相位

没有参考关节角，策略不知道“下一步该去哪”，只能瞎猜。
"""

import torch


def ref_joint_pos(env) -> torch.Tensor:
    """当前参考关节角，形状 (num_envs, 25)。

    参数：
        env：Isaac Lab 环境

    返回：
        tensor，每一行是一个环境当前应该跟踪的 25 个关节角。

    实现：
        动作项里已经加载了参考 npz，这里直接问它要当前帧，
        避免观测和动作使用不同的参考帧。
    """
    term = env.action_manager.get_term("joint_pos")
    return term.get_reference()


def ref_phase(env) -> torch.Tensor:
    """参考步态相位 [sin, cos]，形状 (num_envs, 2)。

    相位可以理解成“当前走到一个周期的百分之几”：
        0%   -> sin=0, cos=1
        25%  -> sin=1, cos=0
        50%  -> sin=0, cos=-1

    用 sin/cos 而不是 0~1 的小数，是为了让相位在循环处连续，
    神经网络更容易学。
    """
    term = env.action_manager.get_term("joint_pos")
    return term.get_phase()
