"""批量对弈自动运行器（BatchRunner）。

设计要点：
- 进程级单例，整个 FastAPI 服务共用；同一时刻只有**一个**批量批次在跑，
  所有打开 /arena/batch 的用户看到的是同一批 4 路 worker 的实时状态。
- 固定 4 个 worker 线程；每轮由全局 round 计数器分发给空闲 worker，
  同一轮内颜色按 round 序号交替（奇数轮互换白黑）。
- worker 失败/停止时不保存未完成的残局，只记录自然终局或步数上限和棋。
- 对局记录自动带 batch_id 落入 arena_storage，/arena 的历史记录面板可查看。
"""
from __future__ import annotations

import datetime
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import chess

import arena_manager as am
import arena_storage
import models as model_registry

logger = logging.getLogger("unichess_server.batch_runner")

BATCH_WORKERS = 4
DEFAULT_MAX_PLIES = 400


class BatchError(Exception):
    pass


class BatchAlreadyRunningError(BatchError):
    pass


@dataclass
class BatchConfig:
    white_model: str
    white_arg: str | None = None
    black_model: str = ""
    black_arg: str | None = None
    rounds: int = 8
    max_plies: int = DEFAULT_MAX_PLIES


@dataclass
class BatchWorkerState:
    index: int
    status: str = "idle"
    arena_id: str | None = None
    round_index: int | None = None
    white_model: str = ""
    white_arg: str | None = None
    black_model: str = ""
    black_arg: str | None = None
    fen: str = chess.STARTING_FEN
    last_move: str | None = None
    san_history: list[str] = field(default_factory=list)
    ply_count: int = 0
    turn: str = "white"
    is_game_over: bool = False
    result: str | None = None
    winner: str | None = None
    termination_reason: str | None = None
    engine_ms: int | None = None
    error: str | None = None


@dataclass
class BatchRound:
    index: int
    white_model: str
    white_arg: str | None
    black_model: str
    black_arg: str | None


class _RoundGetter:
    """线程安全的 round 分发器：每次调用返回一个 BatchRound 或 None。"""

    def __init__(self, rounds: list[BatchRound]):
        self._rounds = rounds
        self._idx = 0
        self._lock = threading.Lock()

    def __call__(self) -> BatchRound | None:
        with self._lock:
            if self._idx >= len(self._rounds):
                return None
            r = self._rounds[self._idx]
            self._idx += 1
            return r


class BatchWorker(threading.Thread):
    """单个 worker 线程：循环取轮次、对弈、更新 runner 统计。"""

    def __init__(self, runner, index: int, round_getter, config: BatchConfig):
        self.runner = runner
        self.index = index
        self.round_getter = round_getter
        self.config = config
        self.state = BatchWorkerState(index=index)
        self._state_lock = threading.Lock()
        super().__init__(daemon=True)

    def run(self) -> None:
        while True:
            if self.runner._shutdown.is_set():
                break
            round_info = self.round_getter()
            if round_info is None:
                break
            try:
                self._play_round(round_info)
            except Exception as exc:
                logger.exception("BatchWorker %d round %d error", self.index, round_info.index)
                self._set_state(status="error", error=str(exc))
        self._set_state(status="idle")

    def _play_round(self, round_info: BatchRound) -> None:
        w_model = round_info.white_model
        w_arg = round_info.white_arg
        b_model = round_info.black_model
        b_arg = round_info.black_arg

        self.runner._on_round_start()
        self._set_state(status="building", round_index=round_info.index,
                        white_model=w_model, white_arg=w_arg,
                        black_model=b_model, black_arg=b_arg, error=None)

        white_engine = None
        black_engine = None
        session = None
        try:
            white_engine = model_registry.create_engine(w_model, w_arg)
            black_engine = model_registry.create_engine(b_model, b_arg)
            session = am.ArenaSession(
                arena_id=uuid.uuid4().hex,
                white_model=w_model,
                white_arg=w_arg,
                black_model=b_model,
                black_arg=b_arg,
                white_engine=white_engine,
                black_engine=black_engine,
                batch_id=self.runner.batch_id,
                storage=self.runner._storage,
            )
            session.setup()
            self._set_state(
                status="playing", arena_id=session.arena_id,
                fen=session.board.fen(), turn="white",
                is_game_over=False, result=None, winner=None,
                termination_reason=None, last_move=None, san_history=[], ply_count=0,
            )

            finished = False
            while True:
                if session.board.is_game_over():
                    break
                if len(session.board.move_stack) >= self.config.max_plies:
                    session.finish_as_draw("max_plies")
                    self._set_state(
                        is_game_over=True, result="1/2-1/2", winner="draw",
                        termination_reason="max_plies", ply_count=len(session.board.move_stack),
                    )
                    self.runner._on_game_complete(self.index, {
                        "winner": "draw", "result": "1/2-1/2", "termination_reason": "max_plies",
                    })
                    finished = True
                    break
                step = session.step()
                self._set_state(
                    fen=step["fen"], last_move=step["move"],
                    san_history=step.get("san_history", [])[-48:],
                    ply_count=step["ply_count"], turn=step["turn"],
                    is_game_over=step["is_game_over"], result=step["result"],
                    winner=step["winner"], termination_reason=step.get("termination_reason"),
                    engine_ms=step.get("engine_ms"),
                )
                if step["is_game_over"]:
                    self.runner._on_game_complete(self.index, step)
                    finished = True
                    break

            if not finished and session is not None and not session.is_cleaned:
                session.cleanup(save_stopped=False)

        except Exception as exc:
            logger.exception("BatchWorker %d round %d error", self.index, round_index)
            self._set_state(status="error", error=str(exc))
            if session is not None and not getattr(session, "is_cleaned", False):
                try:
                    session.cleanup(save_stopped=False)
                except Exception:
                    pass
            self.runner._on_worker_error(self.index, exc)
        finally:
            if white_engine is not None:
                try:
                    white_engine.cleanup()
                except Exception:
                    pass
            if black_engine is not None:
                try:
                    black_engine.cleanup()
                except Exception:
                    pass

    def _set_state(self, **kw):
        with self._state_lock:
            for k, v in kw.items():
                setattr(self.state, k, v)


class BatchRunner:
    """批量对弈进程级单例。

    用法：
        runner = BatchRunner.get()
        runner.start(BatchConfig(...))  # 返回 batch snapshot
        state = runner.snapshot()
    注意：批次一旦启动将自动跑完，不提供手动停止接口。
    """

    _instance: BatchRunner | None = None
    _init_lock = threading.Lock()

    def __init__(self, storage=None):
        self._storage = storage or arena_storage.storage
        # 服务重启后把遗留的 running 批次标记为 interrupted
        self._storage.mark_stale_running_batches()

        self._lock = threading.RLock()
        self._batch: dict[str, Any] | None = None
        self._workers: list[BatchWorker] = []
        self._tally: dict[str, int] = {"white": 0, "black": 0, "draw": 0}
        self._tally_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._started_rounds = 0

    @classmethod
    def get(cls, storage=None):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls(storage=storage)
        return cls._instance

    @classmethod
    def reset(cls):
        with cls._init_lock:
            if cls._instance is not None:
                cls._instance._shutdown.set()
                for w in list(cls._instance._workers):
                    if w.is_alive():
                        w.join(timeout=2)
                cls._instance = None

    @property
    def batch_id(self) -> str | None:
        return self._batch.get("id") if self._batch else None

    @property
    def is_running(self) -> bool:
        return self._batch is not None and self._batch.get("status") == "running"

    def start(self, config: BatchConfig) -> dict[str, Any]:
        with self._lock:
            if self.is_running:
                raise BatchAlreadyRunningError("批量对弈已在运行中")

            if config.rounds % 2 != 0:
                raise BatchError("批量对弈轮数必须为偶数")

            self._tally = {"white": 0, "black": 0, "draw": 0}
            self._started_rounds = 0
            batch_id = uuid.uuid4().hex
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            self._batch = {
                "id": batch_id,
                "created_at": now,
                "end_time": None,
                "white_model": config.white_model,
                "white_arg": config.white_arg,
                "black_model": config.black_model,
                "black_arg": config.black_arg,
                "rounds_planned": config.rounds,
                "rounds_completed": 0,
                "status": "running",
                "tally": dict(self._tally),
                "error": None,
            }
            self._storage.save_batch(self._batch)

            half = config.rounds // 2
            a_rounds = [
                BatchRound(i, config.white_model, config.white_arg, config.black_model, config.black_arg)
                for i in range(half)
            ]
            b_rounds = [
                BatchRound(half + i, config.black_model, config.black_arg, config.white_model, config.white_arg)
                for i in range(half)
            ]
            a_getter = _RoundGetter(a_rounds)
            b_getter = _RoundGetter(b_rounds)

            self._workers = [
                BatchWorker(self, i, a_getter if i < 2 else b_getter, config)
                for i in range(BATCH_WORKERS)
            ]
            for w in self._workers:
                w.start()

            workers = []
            for w in self._workers:
                with w._state_lock:
                    workers.append(dict(w.state.__dict__))

            return {
                "batch": dict(self._batch),
                "workers": workers,
                "is_running": self.is_running,
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            batch = dict(self._batch) if self._batch else None

        # 收集 worker 状态
        workers = []
        for w in self._workers:
            with w._state_lock:
                workers.append(dict(w.state.__dict__))

        # 自动收尾：所有 worker 都停下来且轮次已发完 → completed
        if batch and batch.get("status") == "running":
            any_active = any(w["status"] in ("building", "playing") for w in workers)
            started = self._started_rounds
            if not any_active and started >= batch.get("rounds_planned", 0):
                with self._lock:
                    self._batch = dict(self._batch)
                    self._batch["status"] = "completed"
                    self._batch["end_time"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    self._storage.save_batch(self._batch)
                    batch = dict(self._batch)

        return {
            "batch": batch,
            "workers": workers,
            "is_running": self.is_running,
        }

    def _on_round_start(self):
        with self._tally_lock:
            self._started_rounds += 1

    def _on_game_complete(self, worker_index: int, step_data: dict | None) -> None:
        with self._tally_lock:
            if step_data:
                w = step_data.get("winner")
                if w == "white":
                    self._tally["white"] += 1
                elif w == "black":
                    self._tally["black"] += 1
                elif w == "draw":
                    self._tally["draw"] += 1
            if self._batch:
                self._batch["rounds_completed"] += 1
                self._batch["tally"] = dict(self._tally)
                self._storage.save_batch(self._batch)

    def _on_worker_error(self, worker_index: int, exc: Exception) -> None:
        with self._tally_lock:
            if self._batch:
                self._batch["error"] = str(exc)
                self._batch["status"] = "error"
                self._batch["end_time"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                self._storage.save_batch(self._batch)
