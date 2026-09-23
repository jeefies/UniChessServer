"""定位 UniChessKit（共享管线库）。

kit 没有 pip 安装进 conda 环境：默认用与 Server 同级的 ``../Kit``（远端 ``~/UniChess/Kit``），
可用环境变量 ``UNICHESS_KIT_ROOT`` 覆盖。追加到 sys.path 末尾而不是开头：
kit 仓库根目录下也有 ``tests`` 包，放在前面会遮蔽 Server 自己的同名模块。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

KIT_ROOT = Path(os.environ.get("UNICHESS_KIT_ROOT")
                or Path(__file__).resolve().parent.parent / "Kit").resolve()

if str(KIT_ROOT) not in sys.path:
    sys.path.append(str(KIT_ROOT))


def subprocess_env(extra: dict | None = None) -> dict:
    """kit job 子进程的环境：PYTHONPATH 带上 kit，输出统一 UTF-8。"""
    env = dict(os.environ)
    parts = [str(KIT_ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(parts))
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(extra or {})
    return env
