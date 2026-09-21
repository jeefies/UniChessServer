"""竞技场会话管理：双引擎（White vs Black）对弈与持久化。

设计要点：
- 管理 Arena 对局（双引擎：White Engine 和 Black Engine）。
- 最大并发 Arena 会话数受 MAX_ARENA_SESSIONS = 2 限制，防止 GPU 显存超限。
- 采用与 SessionManager 类似的滑动窗口与构建信号量控制，LRU 淘汰旧会话并释放双引擎。
- 走法合法性权威在服务层 python-chess Board。
- 对局终局或手动关闭时自动将对局记录入库并释放引擎。
"""
from __future__ import annotations

import datetime
import logging
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import chess

import arena_storage
import models as model_registry
from session_manager import IllegalMoveError, InvalidFenError, SessionError, SessionNotFoundError

logger = logging.getLogger("unichess_server.arena_manager")

MAX_ARENA_SESSIONS = 2


class ArenaError(SessionError):
    pass


class ArenaNotFoundError(ArenaError):
    pass


@dataclass
class ArenaSession:
    arena_id: str
    white_model: str
    white_arg: str | None
    black_model: str
    black_arg: str | None
    white_engine: Any
    black_engine: Any
    board: chess.Board = field(default_factory=chess.Board)
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    is_cleaned: bool = False
    _op_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def setup(self, fen: str | None = None) -> None:
        with self._op_lock:
            if fen:
                try:
                    self.board = chess.Board(fen)
                except ValueError as e:
                    raise InvalidFenError(f'无法解析的 FEN "{fen}": {e}') from e
            else:
                self.board = chess.Board()

            fen_str = self.board.fen()
            self.white_engine.setup(fen_str)
            self.black_engine.setup(fen_str)

    def step(self) -> dict[str, Any]:
        """执行单步对弈。

        根据 board.turn 调度执棋引擎进行 move，将合法着法推入 board 并同步给对手。
        若对局终局，自动保存记录并 cleanup() 释放引擎。
        """
        with self._op_lock:
            if self.board.is_game_over():
                return self._build_game_over_result()

            turn_is_white = self.board.turn == chess.WHITE
            active_engine = self.white_engine if turn_is_white else self.black_engine
            opponent_engine = self.black_engine if turn_is_white else self.white_engine

            start_t = time.perf_counter()
            try:
                move_res = active_engine.engine_move()
            except Exception as e:
                logger.exception("引擎 engine_move 异常 (turn=%s)", "white" if turn_is_white else "black")
                raise ArenaError(f"引擎走子执行失败: {e}") from e
            engine_ms = int((time.perf_counter() - start_t) * 1000)

            if not isinstance(move_res, dict) or "engine_move" not in move_res or not move_res["engine_move"]:
                raise ArenaError(f'引擎未返回有效的 "engine_move": {move_res!r}')

            uci = move_res["engine_move"]
            try:
                move = chess.Move.from_uci(uci)
            except (ValueError, AssertionError) as e:
                raise ArenaError(f'引擎返回无法解析的 UCI 走法 "{uci}": {e}') from e

            if move not in self.board.legal_moves:
                raise ArenaError(f'引擎返回的走法 "{uci}" 在当前局面下不合法')

            san = self.board.san(move)
            self.board.push(move)

            # 同步给对手引擎
            try:
                opponent_engine.human_move(uci)
            except Exception as e:
                # 引擎同步失败时回滚服务层 board
                self.board.pop()
                logger.exception("对手引擎 human_move 同步失败: %s", uci)
                raise ArenaError(f"对手引擎同步走法失败: {e}") from e

            is_over = self.board.is_game_over()
            result, winner, termination_reason = self._evaluate_outcome()

            step_data = {
                "move": uci,
                "san": san,
                "fen": self.board.fen(),
                "turn": "white" if self.board.turn == chess.WHITE else "black",
                "is_game_over": is_over,
                "result": result,
                "winner": winner,
                "ply_count": len(self.board.move_stack),
                "engine_ms": engine_ms,
                "eval": move_res.get("eval") if isinstance(move_res, dict) else None,
                "engine_details": {k: v for k, v in move_res.items() if k not in ("engine_move", "fen")} if isinstance(move_res, dict) else {},
            }

            if is_over:
                self._save_record_to_db(result=result, winner=winner, termination_reason=termination_reason)
                self.cleanup()

            return step_data

    def state(self) -> dict[str, Any]:
        with self._op_lock:
            result, winner, termination_reason = self._evaluate_outcome()
            return {
                "arena_id": self.arena_id,
                "created_at": self.created_at,
                "white_model": self.white_model,
                "white_arg": self.white_arg,
                "black_model": self.black_model,
                "black_arg": self.black_arg,
                "fen": self.board.fen(),
                "turn": "white" if self.board.turn == chess.WHITE else "black",
                "legal_moves": [m.uci() for m in self.board.legal_moves],
                "is_game_over": self.board.is_game_over(),
                "result": result,
                "winner": winner,
                "termination_reason": termination_reason,
                "ply_count": len(self.board.move_stack),
                "moves": " ".join([m.uci() for m in self.board.move_stack]),
            }

    def _evaluate_outcome(self) -> tuple[str, str | None, str | None]:
        """判定对局当前胜负结果、赢家和终局原因。"""
        if not self.board.is_game_over():
            return "*", None, None

        outcome = self.board.outcome()
        if outcome is None:
            return "*", None, None

        res_str = outcome.result()
        if outcome.winner == chess.WHITE:
            winner = "white"
        elif outcome.winner == chess.BLACK:
            winner = "black"
        else:
            winner = "draw"

        reason = outcome.termination.name.lower()
        return res_str, winner, reason

    def _build_game_over_result(self) -> dict[str, Any]:
        result, winner, reason = self._evaluate_outcome()
        return {
            "move": None,
            "san": None,
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "is_game_over": True,
            "result": result,
            "winner": winner,
            "ply_count": len(self.board.move_stack),
            "engine_ms": 0,
            "eval": None,
            "engine_details": {},
        }

    def _save_record_to_db(self, result: str, winner: str | None, termination_reason: str | None) -> None:
        try:
            record = {
                "id": self.arena_id,
                "created_at": self.created_at,
                "end_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "white_model": self.white_model,
                "white_arg": self.white_arg,
                "black_model": self.black_model,
                "black_arg": self.black_arg,
                "moves": " ".join([m.uci() for m in self.board.move_stack]),
                "ply_count": len(self.board.move_stack),
                "result": result,
                "winner": winner,
                "termination_reason": termination_reason,
            }
            arena_storage.storage.save_record(record)
        except Exception:
            logger.exception("保存竞技场记录到数据库失败 (arena_id=%s)", self.arena_id)

    def cleanup(self) -> None:
        with self._op_lock:
            if self.is_cleaned:
                return
            self.is_cleaned = True

            # 若尚未对局结束，由外部或LRU淘汰/关闭，标记为 stopped 存库
            if not self.board.is_game_over():
                self._save_record_to_db(result="*", winner="stopped", termination_reason="stopped")

            for role, eng in (("white", self.white_engine), ("black", self.black_engine)):
                try:
                    eng.cleanup()
                except Exception:
                    logger.exception("竞技场清理 %s 引擎失败", role)


class ArenaManager:
    """线程安全的双引擎竞技场管理器，限制最多 MAX_ARENA_SESSIONS。"""

    def __init__(self, max_sessions: int = MAX_ARENA_SESSIONS, storage: arena_storage.ArenaStorage | None = None):
        self._max_sessions = max_sessions
        self._storage = storage or arena_storage.storage
        self._sessions: OrderedDict[str, ArenaSession] = OrderedDict()
        self._lock = threading.RLock()
        self._build_slots = threading.BoundedSemaphore(max_sessions)

    def create(
        self,
        white_model: str,
        white_arg: str | None = None,
        black_model: str = "",
        black_arg: str | None = None,
        fen: str | None = None,
    ) -> ArenaSession:
        arena_id = uuid.uuid4().hex
        with self._build_slots:
            white_engine = model_registry.create_engine(white_model, white_arg)
            try:
                black_engine = model_registry.create_engine(black_model, black_arg)
            except BaseException:
                try:
                    white_engine.cleanup()
                except Exception:
                    pass
                raise

            session = ArenaSession(
                arena_id=arena_id,
                white_model=white_model,
                white_arg=white_arg,
                black_model=black_model,
                black_arg=black_arg,
                white_engine=white_engine,
                black_engine=black_engine,
            )

            try:
                session.setup(fen)
            except BaseException:
                try:
                    white_engine.cleanup()
                except Exception:
                    pass
                try:
                    black_engine.cleanup()
                except Exception:
                    pass
                raise

        with self._lock:
            victims = self._evict_if_full_locked()
            self._sessions[arena_id] = session

        for victim in victims:
            try:
                victim.cleanup()
            except Exception:
                logger.exception("淘汰竞技场会话 %s 失败", victim.arena_id)

        return session

    def get(self, arena_id: str) -> ArenaSession:
        with self._lock:
            session = self._sessions.get(arena_id)
            if session is None:
                raise ArenaNotFoundError(f'竞技场对局 "{arena_id}" 不存在或已被清理')
            self._sessions.move_to_end(arena_id)
            return session

    def close(self, arena_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(arena_id, None)
        if session is not None:
            session.cleanup()

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "arena_id": aid,
                    "white_model": s.white_model,
                    "white_arg": s.white_arg,
                    "black_model": s.black_model,
                    "black_arg": s.black_arg,
                    "is_cleaned": s.is_cleaned,
                }
                for aid, s in self._sessions.items()
            ]

    def _evict_if_full_locked(self) -> list[ArenaSession]:
        victims: list[ArenaSession] = []
        while len(self._sessions) >= self._max_sessions:
            _, oldest = self._sessions.popitem(last=False)
            victims.append(oldest)
        return victims


# 单例管理
manager = ArenaManager()
