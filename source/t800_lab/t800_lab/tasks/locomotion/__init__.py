"""Locomotion（移动）任务包。

导入 velocity 后，里面的 gym.register 会执行，
任务 id 就被注册到 Gym 了。
"""

from . import velocity  # noqa: F401

from . import mimic  # noqa: F401
