"""Mimic 任务的 mdp 函数集合。

本文件用到的库：无（只做 re-export）。

mdp = Markov Decision Process（马尔可夫决策过程）。
在这里可以简单理解成“强化学习的各种零件”：
    观测、奖励、终止、事件、动作。

这个文件把各文件的函数导出，env cfg 可以统一 import：

    from t800_lab.tasks.locomotion.mimic.mdp import (
        RefJointPositionActionCfg,
        ref_joint_pos,
        ref_phase,
        track_ref_joint_pos,
        track_ref_joint_l2,
        track_base_vel,
        reset_joints_to_reference,
    )
"""

from .actions import RefJointPositionAction, RefJointPositionActionCfg
from .events import reset_joints_to_reference
from .observations import ref_joint_pos, ref_phase
from .rewards import track_base_vel, track_ref_joint_l2, track_ref_joint_pos
