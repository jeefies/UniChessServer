"""批量对弈自动运行器（BatchRunner）。

设计要点：
- 进程级单例，整个 FastAPI 服务共用；同一时刻只有**一个**批量批次在跑，
  所有打开 /arena/batch 的用户看到的是同一批 4 路 worker 的实时状态。
- 固定 4 个 worker 线程，从同一个轮次队列取任务。轮次按"开局对"组织：
  第 i 对的两局使用同一开局、A/B 互换执白，消除开局与先手偏差，
  也避免确定性引擎把同一盘棋逐字节重复下 N 遍。
- 统计按模型 A/B 计数（a_win / b_win / draw），与执色无关。
- 任一 worker 出错即停止整批：其余 worker 在当前着法后放弃残局退出，
  最后一个退出的 worker 负责把批次收尾为 completed / error。
- worker 失败/停止时不保存未完成的残局，只记录自然终局或步数上限和棋。
- 对局记录自动带 batch_id 落入 arena_storage，/arena 的历史记录面板可查看。

P2 将由 UniChessKit 的 MatchJob 取代本模块。
"""
from __future__ import annotations

import datetime
import logging
import pathlib
import random
import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

import chess

import arena_manager as am
import arena_storage
import models as model_registry

logger = logging.getLogger("unichess_server.batch_runner")

BATCH_WORKERS = 4
DEFAULT_MAX_PLIES = 400
MAX_ROUNDS = 200
MAX_PLIES_CAP = 1000
# 启动前要求的 GPU 空闲显存（MiB）；训练占卡时拒绝启动。无 nvidia-smi 时跳过检查。
MIN_FREE_GPU_MIB = 4096

OPENINGS_PATH = pathlib.Path(__file__).resolve().parent / "openings.txt"


class BatchError(Exception):
    pass


class BatchAlreadyRunningError(BatchError):
    pass


class GpuBusyError(BatchError):
    pass


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_openings(path: pathlib.Path = OPENINGS_PATH) -> list[list[str]]:
    """读取开局库：每行一条 UCI 着法序列，# 开头为注释。"""
    lines: list[list[str]] = []
    if not path.is_file():
        return lines
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.split("#", 1)[0].strip()
        if text:
            lines.append(text.split())
    return lines


def gpu_free_mib() -> int | None:
    """返回第 0 张卡的空闲显存（MiB）；拿不到时返回 None。"""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        logger.warning("nvidia-smi 查询失败，跳过 GPU 空闲检查", exc_info=True)
        return None


@dataclass
class BatchConfig:
    white_model: str
    white_arg: str | None = None
    black_model: str = ""
    black_arg: str | None = None
    rounds: int = 8
    max_plies: int = DEFAULT_MAX_PLIES
    use_openings: bool = True
    min_free_gpu_mib: int = MIN_FREE_GPU_MIB


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
    opening: str | None = None
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
    a_is_white: bool
    opening: list[str] = field(default_factory=list)


def build_rounds(config: BatchConfig, openings: list[list[str]], rng: random.Random) -> list[BatchRound]:
    """按开局对生成轮次：第 i 对 = (A 执白, B 执白)，两局共用一条开局。"""
    pairs = config.rounds // 2
    order = list(range(len(openings)))
    rng.shuffle(order)
    rounds: list[BatchRound] = []
    for i in range(pairs):
        opening = list(openings[order[i % len(order)]]) if order else []
        rounds.append(BatchRound(2 * i, config.white_model, config.white_arg,
                                 config.black_model, config.black_arg, True, opening))
        rounds.append(BatchRound(2 * i + 1, config.black_model, config.black_arg,
                                 config.white_model, config.white_arg, False, opening))
    return rounds


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


class _BatchStopped(Exception):
    """整批被叫停（其它 worker 出错），当前残局直接放弃。"""


class BatchWorker(threading.Thread):
    """单个 worker 线程：循环取轮次、对弈、把结果回报给 runner。"""

    def __init__(self, runner: BatchRunner, index: int, round_getter: _RoundGetter,
                 config: BatchConfig, stop: threading.Event):
        self.runner = runner
        self.index = index
        self.round_getter = round_getter
        self.config = config
        self.stop = stop
        self.state = BatchWorkerState(index=index)
        self._state_lock = threading.Lock()
        super().__init__(daemon=True)

    def run(self) -> None:
        try:
            while not self.stop.is_set():
                round_info = self.round_getter()
                if round_info is None:
                    break
                try:
                    self._play_round(round_info)
                except _BatchStopped:
                    break
                except Exception as exc:
                    logger.exception("BatchWorker %d round %d error", self.index, round_info.index)
                    self._set_state(status="error", error=str(exc))
                    self.runner._on_worker_error(self.index, exc)
                    return
            self._set_state(status="idle")
        finally:
            self.runner._on_worker_exit(self.index)

    def _play_round(self, round_info: BatchRound) -> None:
        w_model, w_arg = round_info.white_model, round_info.white_arg
        b_model, b_arg = round_info.black_model, round_info.black_arg

        self._set_state(status="building", round_index=round_info.index,
                        white_model=w_model, white_arg=w_arg,
                        black_model=b_model, black_arg=b_arg,
                        opening=" ".join(round_info.opening) or None, error=None)

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
            session.play_opening(round_info.opening)
            self._set_state(
                status="playing", arena_id=session.arena_id,
                fen=session.board.fen(),
                turn="white" if session.board.turn == chess.WHITE else "black",
                is_game_over=False, result=None, winner=None,
                termination_reason=None,
                last_move=session.board.move_stack[-1].uci() if session.board.move_stack else None,
                san_history=list(session.san_history), ply_count=len(session.board.move_stack),
            )

            while True:
                if self.stop.is_set():
                    raise _BatchStopped()
                if len(session.board.move_stack) >= self.config.max_plies:
                    session.finish_as_draw("max_plies")
                    self._set_state(
                        is_game_over=True, result="1/2-1/2", winner="draw",
                        termination_reason="max_plies", ply_count=len(session.board.move_stack),
                    )
                    self.runner._on_game_complete(round_info, "draw")
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
                    self.runner._on_game_complete(round_info, step["winner"])
                    break
        finally:
            if session is not None and not session.is_cleaned:
                try:
                    session.cleanup(save_stopped=False)
                except Exception:
                    logger.exception("BatchWorker %d 清理会话失败", self.index)
            # session.cleanup 已释放引擎；会话未建成时在这里兜底（cleanup 需幂等）
            if session is None:
                for eng in (white_engine, black_engine):
                    if eng is not None:
                        try:
                            eng.cleanup()
                        except Exception:
                            pass

    def _set_state(self, **kw):
        with self._state_lock:
            for k, v in kw.items():
                setattr(self.state, k, v)

    def snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            return dict(self.state.__dict__)


class BatchRunner:
    """批量对弈进程级单例。

    用法：
        runner = BatchRunner.get()
        runner.start(BatchConfig(...))  # 返回 batch snapshot
        state = runner.snapshot()
    注意：批次一旦启动将自动跑完（或因 worker 出错整批停止），不提供手动停止接口。
    """

    _instance: BatchRunner | None = None
    _init_lock = threading.Lock()

    def __init__(self, storage=None, openings: list[list[str]] | None = None):
        self._storage = storage or arena_storage.storage
        # 服务重启后把遗留的 running 批次标记为 interrupted
        self._storage.mark_stale_running_batches()
        self._openings = load_openings() if openings is None else openings

        self._lock = threading.RLock()
        self._batch: dict[str, Any] | None = None
        self._workers: list[BatchWorker] = []
        self._active_workers = 0
        self._stop = threading.Event()

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
                cls._instance._stop.set()
                for w in list(cls._instance._workers):
                    if w.is_alive():
                        w.join(timeout=5)
                cls._instance = None

    @property
    def batch_id(self) -> str | None:
        with self._lock:
            return self._batch.get("id") if self._batch else None

    @property
    def is_running(self) -> bool:
        """只要还有 worker 线程在跑就算运行中（即便批次已因出错进入 error 收尾）。"""
        with self._lock:
            return self._active_workers > 0

    def start(self, config: BatchConfig) -> dict[str, Any]:
        with self._lock:
            if self.is_running:
                raise BatchAlreadyRunningError("批量对弈已在运行中")
            if config.rounds < 2 or config.rounds % 2 != 0:
                raise BatchError("批量对弈轮数必须为不小于 2 的偶数")
            if config.rounds > MAX_ROUNDS:
                raise BatchError(f"批量对弈轮数不能超过 {MAX_ROUNDS}")
            if not 1 <= config.max_plies <= MAX_PLIES_CAP:
                raise BatchError(f"单局步数上限需在 1..{MAX_PLIES_CAP} 之间")
            if config.min_free_gpu_mib > 0:
                free = gpu_free_mib()
                if free is not None and free < config.min_free_gpu_mib:
                    raise GpuBusyError(
                        f"GPU 空闲显存 {free} MiB 低于 {config.min_free_gpu_mib} MiB，"
                        "可能有训练任务在跑，拒绝启动批量对弈")

            batch_id = uuid.uuid4().hex
            self._batch = {
                "id": batch_id,
                "created_at": _now(),
                "end_time": None,
                "white_model": config.white_model,
                "white_arg": config.white_arg,
                "black_model": config.black_model,
                "black_arg": config.black_arg,
                "rounds_planned": config.rounds,
                "rounds_completed": 0,
                "status": "running",
                "tally": {"a_win": 0, "b_win": 0, "draw": 0},
                "error": None,
            }
            self._storage.save_batch(self._batch)

            openings = self._openings if config.use_openings else []
            rounds = build_rounds(config, openings, random.Random())
            getter = _RoundGetter(rounds)
            self._stop = threading.Event()
            self._workers = [
                BatchWorker(self, i, getter, config, self._stop)
                for i in range(BATCH_WORKERS)
            ]
            self._active_workers = len(self._workers)
            for w in self._workers:
                w.start()

            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        """只读快照；状态迁移全部发生在 worker 回调里。"""
        with self._lock:
            batch = dict(self._batch) if self._batch else None
            if batch:
                batch["tally"] = dict(batch["tally"])
            workers = list(self._workers)
            running = self._active_workers > 0
        return {
            "batch": batch,
            "workers": [w.snapshot() for w in workers],
            "is_running": running,
        }

    def _on_game_complete(self, round_info: BatchRound, winner: str | None) -> None:
        with self._lock:
            if not self._batch:
                return
            tally = self._batch["tally"]
            if winner == "draw":
                tally["draw"] += 1
            elif winner in ("white", "black"):
                a_won = (winner == "white") == round_info.a_is_white
                tally["a_win" if a_won else "b_win"] += 1
            self._batch["rounds_completed"] += 1
            self._storage.save_batch(self._batch)

    def _on_worker_error(self, worker_index: int, exc: Exception) -> None:
        with self._lock:
            self._stop.set()
            if self._batch and self._batch.get("error") is None:
                self._batch["error"] = f"worker {worker_index + 1}: {exc}"
                self._storage.save_batch(self._batch)

    def _on_worker_exit(self, worker_index: int) -> None:
        with self._lock:
            self._active_workers -= 1
            if self._active_workers > 0 or not self._batch:
                return
            if self._batch["status"] == "running":
                if self._batch.get("error"):
                    self._batch["status"] = "error"
                elif self._stop.is_set():
                    self._batch["status"] = "stopped"  # 仅 reset()（测试/关停）会走到
                else:
                    self._batch["status"] = "completed"
                self._batch["end_time"] = _now()
                self._storage.save_batch(self._batch)
