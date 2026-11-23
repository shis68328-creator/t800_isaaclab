"""T800 任务包。

import t800_lab 时会走到这里，进而导入 locomotion，
最终执行 velocity/__init__.py 里的 gym.register。
"""

from . import locomotion  # noqa: F401
