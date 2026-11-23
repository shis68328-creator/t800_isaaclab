"""T800 Isaac Lab 扩展入口。

只要 import t800_lab，就会：
    1. 导入 tasks 包
    2. 注册 Isaac-T800-Locomotion-v0
所以训练脚本里写 `import t800_lab` 就够了。
"""

from . import tasks  # noqa: F401
