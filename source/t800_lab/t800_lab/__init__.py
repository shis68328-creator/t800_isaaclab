"""T800 Isaac Lab 扩展入口。

==============================================================================
这个文件干什么
==============================================================================
只要 `import t800_lab`，就会：
    1. 导入 tasks 包
    2. 执行 tasks/locomotion/velocity/__init__.py 和 mimic/__init__.py
    3. 里面的 gym.register 把任务注册到 Gym

所以训练脚本里只要写：

    import t800_lab

就不用管任务是怎么注册的。

注意：
    这个 import 必须在 AppLauncher 启动 Isaac Sim 之后执行，
    因为它会间接导入 isaaclab / omni / carb。
"""

from . import tasks  # noqa: F401
