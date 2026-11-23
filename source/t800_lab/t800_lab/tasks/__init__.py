"""T800 任务包。

==============================================================================
这个文件干什么
==============================================================================
导入 locomotion 子包，从而执行里面的 gym.register。

Python 的包机制：
    一个目录里有 __init__.py，它就是一个“包”。
    import t800_lab.tasks 会执行这个文件。

这里不需要写注册代码，注册在：
    tasks/locomotion/velocity/__init__.py
    tasks/locomotion/mimic/__init__.py
"""

from . import locomotion  # noqa: F401
