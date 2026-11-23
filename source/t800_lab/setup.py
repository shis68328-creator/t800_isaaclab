# ==============================================================================
# 本文件用到的库（小例子见 docs/LIBRARIES.md）
#   setuptools.setup: 安装入口
#       setup(name="t800_lab", packages=find_packages())
#   setuptools.find_packages: 自动找子包
#   toml.load: 读 config/extension.toml
#       meta = toml.load("config/extension.toml")
# ==============================================================================

"""安装 t800_lab 扩展。

======================================================================
这个文件什么时候执行
======================================================================
`pip install -e source/t800_lab` 时，pip 会运行 setup.py。

-e 表示 editable（可编辑安装）：
    不会把代码复制到 site-packages，而是创建一个指向当前源码目录的链接。
    以后改 source/ 里的代码，Python 会直接读到新版本，不用重装。

======================================================================
用到的包
======================================================================
setuptools.setup：
    打包/安装入口。
setuptools.find_packages：
    自动找出所有带 __init__.py 的子包（t800_lab、t800_lab.assets 等）。
toml：
    读取 config/extension.toml 里的版本、作者、描述。
os：
    拼路径。
"""

import os

import toml
from setuptools import find_packages, setup

# 当前文件所在目录：source/t800_lab
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))

# 读取 Isaac Lab 扩展元数据（版本、作者、仓库等）
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

setup(
    name="t800_lab",
    # find_packages() 会自动包含 t800_lab、t800_lab.assets、t800_lab.tasks...
    packages=find_packages(),
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    # 只装额外依赖：Isaac Lab 环境里已经有 numpy / torch / rsl_rl，
    # 这里不要再 pin 它们，避免把 Isaac Sim 的版本搞乱。
    install_requires=["psutil", "toml"],
    license="Apache License 2.0",
    include_package_data=True,
    python_requires=">=3.10",
    zip_safe=False,
)
