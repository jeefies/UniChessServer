"""ResNet 模型的 GameEngine 占位实现。

TODO（由 UniChess/ResNet 项目负责的 agent 迁移填充）：
接入 UniChess/ResNet/engine/engine.py 里的 UniChessEngine，实现下面的
GameEngine 契约（见 Server/README.md 完整说明）。

类属性 IMPLEMENTED = False 表示当前仅为占位实现，不可用于对局。

契约：

    class GameEngine:
        def __init__(self, **kwargs):
            \"\"\"kwargs 来自 config.json 中被选中的 <arg_name> 那组参数。\"\"\"

        def setup(self, fen: str | None = None) -> dict:
            \"\"\"初始化/重置局面。fen 为 None 时使用标准初始局面。
            返回当前局面状态（建议至少含 fen、turn）。\"\"\"

        def human_move(self, uci: str) -> dict:
            \"\"\"应用一步人类走法（服务层已用 python-chess 校验过合法性），
            不触发引擎思考，只更新内部局面状态。返回当前局面状态。\"\"\"

        def engine_move(self) -> dict:
            \"\"\"不接受走法参数，引擎按当前局面自行决定并走一步。
            返回值必须包含（强制字段，服务层依据它推进权威 Board）：
                {"engine_move": "e7e5", "fen": "...", "done": false, ...}
            缺失/为空时服务层直接报错，绝不静默不错位。\"\"\"

        def state(self) -> dict:
            \"\"\"返回当前局面状态快照。\"\"\"

        def undo(self) -> dict:
            \"\"\"悔棋（一般悔双方各一步，回到人类走棋前），返回局面状态。\"\"\"

        def cleanup(self) -> None:
            \"\"\"释放本实例占用的资源（GPU 显存、搜索树/cache 等）。
            滑动窗口淘汰最旧实例时会调用，必须实现（哪怕是空实现），
            未实现会导致 /api/new 拒绝加载该模型。\"\"\"
"""
from __future__ import annotations


class GameEngine:
    IMPLEMENTED = False
    NOT_IMPLEMENTED_REASON = (
        "ResNet 引擎尚未接入 GameEngine 契约，暂不可对局。接入指南见 Server/README.md。"
    )

    def __init__(self, **kwargs):
        raise NotImplementedError("ResNet 引擎尚未接入，暂不可对局。")

    def setup(self, fen: str | None = None) -> dict:
        raise NotImplementedError

    def human_move(self, uci: str) -> dict:
        raise NotImplementedError

    def engine_move(self) -> dict:
        raise NotImplementedError

    def state(self) -> dict:
        raise NotImplementedError

    def undo(self) -> dict:
        raise NotImplementedError

    def cleanup(self) -> None:
        raise NotImplementedError
