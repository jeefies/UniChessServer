"""定位 import 根（~/UniChess）：Kit / ResNet / Transformer / SSM 都是它的顶层包。

扁平布局（2026-09-25 重构）后，各仓库根目录**就是**包本身，import 根是它们的共同父目录；
kit 仍然没有 pip 安装进 conda 环境，所以这里把 import 根追加进 sys.path，
之后 ``import Kit`` / ``import Transformer`` 才可用。可用环境变量 ``UNICHESS_IMPORT_ROOT``
（兼容旧名 ``UNICHESS_KIT_ROOT``）。追加到 sys.path 末尾而不是开头：
Kit 仓库下也有 ``tests`` 包，放在前面会遮蔽 Server 自己的同名模块。

Server 的 ``models/<name>`` 指向各引擎仓库（符号链接），各引擎的 ``engine.py`` 自己会再挂一次
import 根——两边都追加不重复，顺序不受影响。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

IMPORT_ROOT = Path(os.environ.get("UNICHESS_IMPORT_ROOT")
                   or os.environ.get("UNICHESS_KIT_ROOT")
                   or Path(__file__).resolve().parent.parent).resolve()

if str(IMPORT_ROOT) not in sys.path:
    sys.path.append(str(IMPORT_ROOT))

# 兼容旧名：别的地方（含远端脚本）可能还在 import 这两个符号
KIT_ROOT = IMPORT_ROOT


def subprocess_env(extra: dict | None = None) -> dict:
    """kit job 子进程的环境：PYTHONPATH 带上 import 根，输出统一 UTF-8。"""
    env = dict(os.environ)
    parts = [str(IMPORT_ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(parts))
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(extra or {})
    return env
