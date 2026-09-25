"""对局会话管理：滑动窗口生命周期 + 合法性校验权威判定。

设计要点（对应用户澄清的需求）：
- 每一局棋对应一个长期存活的 GameEngine 实例（同一局内复用同一个实例，
  这样 MCTS/搜索树、R cache 等状态才能在同一局内正确复用/推进）。
- 服务进程内最多同时持有 MAX_SESSIONS 个已注册对局（滑动窗口，按创建/访问顺序）。
  超过时强制淘汰最旧的一个，调用其 GameEngine.cleanup() 释放资源。
  cleanup() 在管理器锁之外执行，避免释放 GPU/搜索树的耗时阻塞其它对局。
- 并发在建引擎数同样受 MAX_SESSIONS 限制（信号量）：否则 N 个并发 /api/new
  会同时加载 N 份模型权重，瞬时突破显存预算。注册+在建的瞬时实例数最多约
  2×MAX_SESSIONS，真实引擎接入时如需更严苛的显存预算可在此基础上再加限制。
- 走法合法性判定的权威在服务层：每个会话额外维护一份 python-chess 的
  Board，人类走子先经它校验合法，再转发给 GameEngine.human_move()。
  GameEngine 内部不需要（也不应被信任）做合法性判断。
- engine_white 决定引擎执哪方色，由服务层（本文件）决定调度时序：
  轮到引擎时调用 GameEngine.engine_move()（新局引擎执白时也在这里
  触发开局第一步），轮到人类时才接受 human_move(uci)。GameEngine
  本身不需要知道自己执哪方。
- 每个会话有独立的 _op_lock：FastAPI 同步端点在线程池并发执行，同一
  session_id 的并发请求（双 Tab、重复提交）必须串行推进 Board 与引擎，
  否则共享的搜索树/cache 会被并发写坏。

失败语义（审查修复）：
- human_move 先推服务层 Board 再转给引擎；引擎抛异常时回滚 Board，
  保证任一侧失败双方停在同一局面，不会永久错位。
- create() 中引擎构建或 setup（含引擎执白的开局第一步）失败时，立即
  调用 cleanup() 释放已建引擎，不会因未注册而泄漏。
- engine_move() 返回值必须包含非空 "engine_move"（UCI）字段（强制，
  不再是建议）：缺失即报错，绝不允许静默不错位。
- state() 的 fen 始终取服务层权威 Board，引擎返回值不覆盖。
- 终局判定用 UniChessKit 的 classify（claim_draw 语义：三次重复 / 50 回合一旦可申请即终局），
  与批量对弈、观战、各引擎的训练评测口径一致。
"""
from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import chess

import kit_env  # noqa: F401  挂好 UniChessKit 路径
import models as model_registry
from Kit.rules import classify

logger = logging.getLogger('unichess_server.session_manager')

MAX_SESSIONS = 4


class SessionError(Exception):
    pass


class SessionNotFoundError(SessionError):
    pass


class IllegalMoveError(SessionError):
    """走法未通过服务层 python-chess 合法性校验。"""


class InvalidFenError(SessionError):
    """请求携带的 fen 无法被 python-chess 解析（客户端输入错误，映射 400）。"""


@dataclass
class GameSession:
    session_id: str
    model_name: str
    arg_name: str | None
    engine_white: bool
    engine: Any
    board: chess.Board = field(default_factory=chess.Board)
    # 每会话操作锁：setup/human_move/undo/state/cleanup 全程持有，
    # 保证同一对局的请求串行推进（管理器锁只保护字典本身，不保护会话状态）。
    _op_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def _push_engine_result(self, result: dict) -> None:
        """把引擎返回的 engine_move 同步进服务层权威 Board。

        "engine_move" 字段是强制约定：引擎内部已走子，服务层 Board 必须
        同步推进；缺失/为空/不合法一律报错，绝不静默跳过（那会造成
        "服务层局面落后于引擎"的隐蔽错位）。
        """
        engine_move = result.get('engine_move') if isinstance(result, dict) else None
        if not engine_move or not isinstance(engine_move, str):
            raise SessionError(
                'engine_move() 必须返回包含非空 "engine_move"（UCI 字符串）字段的 dict，'
                f'实际返回: {result!r}'
            )
        try:
            move = chess.Move.from_uci(engine_move)
        except (ValueError, AssertionError) as e:
            raise SessionError(
                f'引擎返回的走法 "{engine_move}" 无法解析为 UCI: {e}'
            ) from e
        if move not in self.board.legal_moves:
            raise SessionError(
                f'引擎返回的走法 "{engine_move}" 在服务层当前局面下不合法'
            )
        self.board.push(move)

    def _is_over(self) -> bool:
        return classify(self.board) is not None

    def _is_engine_turn(self) -> bool:
        return (self.board.turn == chess.WHITE) == self.engine_white

    def setup(self, fen: str | None = None) -> dict:
        with self._op_lock:
            if fen:
                try:
                    self.board = chess.Board(fen)
                except ValueError as e:
                    raise InvalidFenError(f'无法解析的 FEN "{fen}": {e}') from e
            else:
                self.board = chess.Board()
            result = self.engine.setup(self.board.fen())

            # 引擎执白且新局面轮到白方时，服务层没有人类走法可等，
            # 由引擎直接走开局第一步。
            if not self._is_over() and self._is_engine_turn():
                engine_result = self.engine.engine_move()
                self._push_engine_result(engine_result)
                return engine_result

            return result

    def human_move(self, uci: str) -> dict:
        """校验并应用人类走法，若接下来轮到引擎则让引擎应答一步。"""
        with self._op_lock:
            if self._is_over():
                # claim_draw 语义下可申请和棋即终局，此时仍有合法着法，必须显式拒绝
                raise IllegalMoveError('对局已结束，不能继续走子')
            if self._is_engine_turn():
                raise IllegalMoveError('当前轮到引擎走子，不能提交人类走法')

            try:
                move = chess.Move.from_uci(uci)
            except ValueError as e:
                raise IllegalMoveError(f'无法解析的 UCI 走法 "{uci}": {e}') from e

            if move not in self.board.legal_moves:
                raise IllegalMoveError(f'走法 "{uci}" 在当前局面下不合法')

            self.board.push(move)
            try:
                result = self.engine.human_move(uci)
            except Exception:
                # 引擎应用失败：回滚服务层 Board，避免"服务层领先引擎"的永久错位
                if self.board.move_stack:
                    self.board.pop()
                raise

            if not self._is_over() and self._is_engine_turn():
                engine_result = self.engine.engine_move()
                self._push_engine_result(engine_result)
                return engine_result

            return result

    def state(self) -> dict:
        with self._op_lock:
            engine_state = self.engine.state()
            merged = dict(engine_state) if isinstance(engine_state, dict) else {}
            # fen 以服务层权威 Board 为准：legal_moves/is_game_over 也由它
            # 计算，三者必须来自同一局面，引擎返回值不得覆盖。
            merged['fen'] = self.board.fen()
            merged['legal_moves'] = [m.uci() for m in self.board.legal_moves]
            merged['is_game_over'] = self._is_over()
            merged['engine_white'] = self.engine_white
            return merged

    def undo(self) -> dict:
        with self._op_lock:
            result = self.engine.undo()
            # 悔棋回退人类的一步和引擎应答的一步（若存在），服务层 Board 同步回退。
            for _ in range(2):
                if self.board.move_stack:
                    self.board.pop()
            # 回退后若轮到引擎（如引擎执白），必须重新触发引擎走子，
            # 否则没有任何请求会再调用 engine_move()，对局永久卡死在
            # "引擎思考中"（人类提交又会被 IllegalMoveError 拒绝）。
            if not self._is_over() and self._is_engine_turn():
                engine_result = self.engine.engine_move()
                self._push_engine_result(engine_result)
                return engine_result
            return result

    def cleanup(self) -> None:
        with self._op_lock:
            self.engine.cleanup()


class SessionManager:
    """线程安全的滑动窗口会话管理器，最多持有 MAX_SESSIONS 个活跃对局。"""

    def __init__(self, max_sessions: int = MAX_SESSIONS):
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, GameSession] = OrderedDict()
        self._lock = threading.RLock()
        # 并发在建引擎数上限：防止并发 /api/new 同时加载多份模型权重
        self._build_slots = threading.BoundedSemaphore(max_sessions)

    def create(
        self,
        model_name: str,
        arg_name: str | None = None,
        fen: str | None = None,
        engine_white: bool = False,
    ) -> GameSession:
        session_id = uuid.uuid4().hex
        with self._build_slots:
            engine = model_registry.create_engine(model_name, arg_name)
            session = GameSession(
                session_id=session_id,
                model_name=model_name,
                arg_name=arg_name,
                engine_white=engine_white,
                engine=engine,
            )
            try:
                session.setup(fen)
            except BaseException:
                # 引擎已构建但 setup（或开局第一步）失败：会话永远不会注册，
                # 没有任何 eviction/close 会来 cleanup，必须在这里释放。
                try:
                    engine.cleanup()
                except Exception:
                    logger.exception('setup 失败后释放引擎也失败（model=%s）', model_name)
                raise

        with self._lock:
            victims = self._evict_if_full_locked()
            self._sessions[session_id] = session
        # 淘汰在锁外执行 cleanup：真实引擎释放 GPU/搜索树可达秒级，
        # 持锁做会阻塞其它所有对局的 get/close/list。
        # cleanup 失败只记录不抛出：新会话已注册成功，不该被旧会话的
        # 释放失败连累成 500。
        for victim in victims:
            try:
                victim.cleanup()
            except Exception:
                logger.exception('淘汰会话 %s 的引擎 cleanup 失败', victim.session_id)
        return session

    def get(self, session_id: str) -> GameSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(f'对局 "{session_id}" 不存在或已被淘汰')
            # 访问即刷新为最新（LRU：最久未访问的最先被淘汰）。
            self._sessions.move_to_end(session_id)
            return session

    def close(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            session.cleanup()

    def list_sessions(self) -> list[dict]:
        with self._lock:
            return [
                {
                    'session_id': sid,
                    'model_name': s.model_name,
                    'arg_name': s.arg_name,
                    'engine_white': s.engine_white,
                }
                for sid, s in self._sessions.items()
            ]

    def _evict_if_full_locked(self) -> list[GameSession]:
        """在持有 self._lock 的前提下调用：满了就摘出最旧的会话。

        返回受害者列表，由调用方在释放锁后逐个 cleanup()。
        """
        victims: list[GameSession] = []
        while len(self._sessions) >= self._max_sessions:
            _, oldest = self._sessions.popitem(last=False)
            victims.append(oldest)
        return victims


# 进程级单例：整个 FastAPI 服务共用一个会话管理器。
manager = SessionManager()
