"""Mimic 任务的 mdp 函数集合。

mdp = Markov Decision Process（马尔可夫决策过程），
在这里可以简单理解成“强化学习的各种零件”：
观测、奖励、终止、事件、动作。
"""

from .actions import RefJointPositionAction, RefJointPositionActionCfg
from .events import reset_joints_to_reference
from .observations import ref_joint_pos, ref_phase
from .rewards import track_base_vel, track_ref_joint_l2, track_ref_joint_pos
