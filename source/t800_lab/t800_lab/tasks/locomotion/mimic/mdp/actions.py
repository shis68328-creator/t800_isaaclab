# ==============================================================================
# 本文件用到的库（小例子见 docs/LIBRARIES.md）
#   ActionTerm: 动作项基类
#       class MyAction(ActionTerm): ...
#   ActionTermCfg: 动作项配置
#   torch: 张量、索引、sin/cos
#   numpy: 读 npz 参考轨迹
# ==============================================================================

"""参考步态 residual 动作项。

==============================================================================
先用大白话解释：这个文件在干什么
==============================================================================
普通走路策略要直接输出 25 个关节的目标角，相当于“从白纸开始写作文”。

我们改成：
    最终目标角 = 参考轨迹的关节角 + 策略输出 × 0.25

参考轨迹就是提前录好的“老师草稿”。策略只需要在草稿上改几个字，
所以学习难度低很多。这种“在参考上做小修正”的做法叫 residual action
（残差动作）。

==============================================================================
关键名词
==============================================================================
ActionTerm：
    Isaac Lab 里“动作项”的基类。它负责把神经网络输出的数字，
    翻译成仿真里真正要执行的指令（比如关节目标角）。

joint（关节）：
    机器人身上能转动的部位，比如膝盖、髋、踝。

qpos / joint_pos：
    关节当前的角度，单位是弧度（rad）。

residual：
    残差，也就是“在参考值上再加一点修正”。

==============================================================================
数据流
==============================================================================
walk_ref.npz
    ↓ 按关节名重新排序
参考关节角 (T, 25)
    ↓ 用 episode_length_buf 找到当前第几帧
ref[t]
    ↓ target = ref[t] + action * scale
PD 控制
"""

import math
from pathlib import Path

import numpy as np
import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class RefJointPositionAction(ActionTerm):
    """参考关节角 + residual 的动作项。

    父类 ActionTerm 规定了三个必须实现的东西：
      1. action_dim：动作有几个数
      2. process_actions(actions)：把网络输出变成目标角
      3. apply_actions()：把目标角写进仿真
    """

    cfg: "RefJointPositionActionCfg"

    def __init__(self, cfg: "RefJointPositionActionCfg", env):
        """初始化：读参考轨迹，建立关节映射。

        参数：
            cfg：动作项配置，里面有 npz 路径、缩放系数、关节名
            env：Isaac Lab 环境，能拿到机器人、当前回合长度、控制周期
        """
        super().__init__(cfg, env)
        # 从场景里取出机器人（T800）
        self._asset = env.scene[cfg.asset_name]

        # find_joints 会返回：
        #   joint_ids：这些关节在机器人里的下标
        #   joint_names：按 Isaac Lab 顺序排列的关节名
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)

        # ---- 读参考轨迹 ----
        # npz 是 numpy 的压缩文件，里面存着数组
        data = np.load(cfg.ref_path)
        ref_names = [str(n) for n in data["joint_names"]]

        # MuJoCo 的关节顺序和 Isaac Lab 不一定一样，必须按“名字”对齐。
        # order 的意思是：Isaac 的第 i 个关节，对应 npz 的第几列。
        order = [ref_names.index(name) for name in self._joint_names]
        q = data["joint_pos"][:, order].astype(np.float32)

        # 参考是 100 Hz 录的，Isaac Lab 控制是 50 Hz，
        # 所以隔一帧取一帧，避免步态播放速度不对。
        step = max(1, int(round(env.step_dt / float(data["dt"]))))
        self._ref = torch.tensor(q[::step], device=env.device, dtype=torch.float32)
        self._ref_dt = step * float(data["dt"])
        self._ref_len = self._ref.shape[0]

        # 策略输出缓存。形状 (环境数, 关节数)
        self._raw_actions = torch.zeros(
            (env.num_envs, len(self._joint_ids)), device=env.device, dtype=torch.float32
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        """动作维度：T800 是 25 个关节。"""
        return len(self._joint_ids)

    @property
    def joint_ids(self):
        """这些关节在机器人里的下标，reset 时也会用到。"""
        return self._joint_ids

    @property
    def raw_actions(self) -> torch.Tensor:
        """策略原始输出，形状 (num_envs, 25)。"""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """加上参考之后的最终目标角，形状 (num_envs, 25)。"""
        return self._processed_actions

    def _ref_index(self) -> torch.Tensor:
        """算出每个环境当前应该取参考的第几帧。

        episode_length_buf 是“这个回合已经走了多少控制步”。
        控制周期 0.02 秒，参考周期 0.02 秒，所以索引基本就是步数。
        """
        t = (self._env.episode_length_buf.to(torch.float32) * self._env.step_dt) / self._ref_dt
        return t.long() % self._ref_len

    def get_reference(self) -> torch.Tensor:
        """返回当前参考关节角，形状 (num_envs, 25)。

        观测和奖励都会调用它，保证大家用的是同一帧参考。
        """
        return self._ref[self._ref_index()]

    def get_phase(self) -> torch.Tensor:
        """返回步态相位 [sin, cos]，形状 (num_envs, 2)。

        相位可以理解成“当前走到一个步态周期的百分之几”。
        用 sin/cos 表示，神经网络更容易学。
        """
        t = self._ref_index().to(torch.float32)
        angle = 2.0 * math.pi * t / self._ref_len
        return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)

    def process_actions(self, actions: torch.Tensor):
        """把策略输出变成最终关节目标角。

        参数：
            actions：形状 (num_envs, 25)，范围大约 [-1, 1]

        最终：
            target = 参考角 + actions * scale
        """
        self._raw_actions[:] = actions
        self._processed_actions = self.get_reference() + actions * self.cfg.scale

    def apply_actions(self):
        """把目标角写进仿真，PD 控制器会据此算力矩。"""
        self._asset.set_joint_position_target(
            self._processed_actions, joint_ids=self._joint_ids
        )


@configclass
class RefJointPositionActionCfg(ActionTermCfg):
    """动作项配置。

    属性：
        joint_names：控制哪些关节，[".*"] 表示全部
        scale：residual 缩放，0.25 表示最多修正 0.25 弧度
        ref_path：参考 npz 的路径
    """

    class_type: type = RefJointPositionAction
    joint_names: list[str] = [".*"]
    scale: float = 0.25
    ref_path: str = ""
