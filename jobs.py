"""批量对弈与网页观战：都交给 UniChessKit 的后台 job（独立进程组 + GPU 租约）。

取代旧的 batch_runner.py（进程内 4 线程）与 arena_manager.py（进程内双引擎单步）：

- 引擎解析：GameEngine 声明了 ``KIT_FACTORY`` 的走 kit 原生 Player（跨局攒批，1 进程 8 并发）；
  否则用 ``unichess_kit.serving:game_engine_player_factory`` 包装六方法引擎（T 的 C++ MCTS、M6），
  每个 worker 进程一局，4 进程并行。
- job 目录 ``data/jobs/<id>/``：job.json / status.json / live.json / results.jsonl / job.log。
  Server 只读这些文件，不与 job 进程通信；服务重启后凭目录接管或收尾（进程已死 = interrupted）。
- 批量对弈：仅管理员可启停（app.require_admin）；统计按模型 A/B，Elo 等来自 kit 汇总。
  完成的局按 ``<batch_id>-g<n>`` 写进 arena_storage，历史面板可按批次筛选。
- 观战：一局的 game job 在后台连续下完，``step`` 按顺序逐步揭示 live.json 里的着法；
  引擎还没走出下一步时等待至多 STEP_WAIT_S 秒，仍没有就返回 pending（前端继续轮询）。
  最多同时 MAX_ARENA_SESSIONS 局，超出淘汰最旧的一局（停进程、记为 stopped）。
- 终局裁决统一用 kit 的 classify（claim_draw 语义）；超步数上限记 truncated，按和棋计。
"""
from __future__ import annotations

import datetime
import logging
import secrets
import shutil
import sys
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess

import arena_storage
import kit_env
import models as model_registry
from session_manager import InvalidFenError, SessionError, SessionNotFoundError

from unichess_kit.jobs import STATES_FINAL, JobHandle  # noqa: E402（kit_env 先挂好路径）
from unichess_kit.rules import classify  # noqa: E402

logger = logging.getLogger("unichess_server.jobs")

JOBS_DIR = Path(__file__).resolve().parent / "data" / "jobs"

MAX_ROUNDS = 200
MAX_PLIES_CAP = 1000
DEFAULT_MAX_PLIES = 400
ARENA_MAX_PLIES = 600

BATCH_GPU_MIB = 4096          # 批量对弈（最多 4 个引擎进程）申请的显存预算
GAME_GPU_MIB = 2048           # 观战一局
WRAPPED_WORKERS = 4           # 六方法引擎：同步阻塞，靠多进程并行
NATIVE_CONCURRENCY = 8        # kit 原生引擎：单进程多局协程，跨局攒批
START_WAIT_S = 5.0            # 启动后等 job 越过 GPU 租约检查的时间
STEP_WAIT_S = 20.0
MAX_ARENA_SESSIONS = 2
SAN_TAIL = 48

WRAPPED_FACTORY = "unichess_kit.serving:game_engine_player_factory"


class JobError(Exception):
    pass


class InvalidBatchConfigError(JobError):
    pass


class BatchAlreadyRunningError(JobError):
    pass


class GpuBusyError(JobError):
    pass


class ArenaError(SessionError):
    pass


class ArenaNotFoundError(SessionNotFoundError, ArenaError):
    pass


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).isoformat()


# ====================================================================== 引擎解析

@dataclass(frozen=True)
class EngineRef:
    model: str
    arg: str | None
    spec: dict
    native: bool

    @property
    def label(self) -> str:
        return f"{self.model}({self.arg})" if self.arg else self.model


def resolve_engine(model: str, arg: str | None) -> EngineRef:
    """<model>/<arg> → kit EngineSpec。模型名校验、契约校验、预设查表都复用 models 注册表。"""
    engine_cls = model_registry.engine_class(model)
    kwargs = model_registry.resolve_kwargs(model, arg)
    arg = arg or None
    model_dir = (model_registry.MODELS_DIR / model).resolve()
    label = f"{model}({arg})" if arg else model
    kit_factory = getattr(engine_cls, "KIT_FACTORY", None)
    if kit_factory:
        spec = {"factory": str(kit_factory), "root": str(model_dir),
                "kwargs": {"preset": arg} if arg else {}, "label": label}
    else:
        spec = {"factory": WRAPPED_FACTORY, "root": None, "label": label,
                "kwargs": {"engine_path": str(model_dir / "engine.py"),
                           "engine_kwargs": kwargs, "name": label}}
    return EngineRef(model, arg, spec, bool(kit_factory))


def _winner(result: str) -> str:
    return {"1-0": "white", "0-1": "black"}.get(result, "draw")


def _replay(start_fen: str | None, ucis) -> tuple[chess.Board, list[str]]:
    board = chess.Board(start_fen) if start_fen else chess.Board()
    sans = []
    for uci in ucis:
        move = chess.Move.from_uci(uci)
        sans.append(board.san(move))
        board.push(move)
    return board, sans


def _eval_of(detail: dict | None, mover_white: bool):
    """前端评估条用白方视角：六方法引擎自带 eval 原样透传；kit 搜索给行棋方视角的 q。"""
    info = (detail or {}).get("info") or {}
    if info.get("eval") is not None:
        return info["eval"]
    q = info.get("q")
    if isinstance(q, (int, float)):
        return q if mover_white else -q
    return None


class _JobLauncher:
    def __init__(self, jobs_dir: Path | None, python: str | None, env_extra: dict | None,
                 lease_dir: str | None):
        self.jobs_dir = Path(jobs_dir or JOBS_DIR)
        self.python = python or sys.executable
        self.env_extra = dict(env_extra or {})
        self.lease_dir = lease_dir

    def submit(self, job_id: str, job: dict) -> JobHandle:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        if self.lease_dir:
            job = {**job, "lease_dir": self.lease_dir}
        return JobHandle.submit(self.jobs_dir / job_id, job, python=self.python,
                                env=kit_env.subprocess_env(self.env_extra))


# ====================================================================== 批量对弈

class BatchService:
    """同一时刻最多一个批次；所有访问 /arena/batch 的人看到同一个批次。"""

    def __init__(self, storage: arena_storage.ArenaStorage | None = None, *,
                 jobs_dir: Path | None = None, python: str | None = None,
                 env_extra: dict | None = None, gpu_mib: int = BATCH_GPU_MIB,
                 lease_dir: str | None = None):
        self._storage = storage or arena_storage.storage
        self._launcher = _JobLauncher(jobs_dir, python, env_extra, lease_dir)
        self.gpu_mib = gpu_mib
        self._lock = threading.RLock()
        self._current: dict | None = None
        self._recover()

    # ---------------------------------------------------------------- 生命周期

    def _recover(self) -> None:
        """接管上次服务进程留下的 running 批次：job 目录在就按其状态收尾，否则记为 interrupted。"""
        running = [b for b in self._storage.list_batches(limit=50) if b.get("status") == "running"]
        for i, row in enumerate(running):
            job_dir = self._launcher.jobs_dir / row["id"]
            if i == 0 and (job_dir / "job.json").is_file():
                self._current = self._attach(row, JobHandle(job_dir))
                self.refresh()
            else:
                row.update(status="interrupted", end_time=row.get("end_time") or _now())
                self._storage.save_batch(row)

    @staticmethod
    def _attach(row: dict, handle: JobHandle) -> dict:
        return {"row": row, "handle": handle, "ingested": set(),
                "a": (row["white_model"], row.get("white_arg")),
                "b": (row["black_model"], row.get("black_arg"))}

    def start(self, white_model: str, white_arg: str | None, black_model: str,
              black_arg: str | None, rounds: int = 8,
              max_plies: int = DEFAULT_MAX_PLIES) -> dict:
        if rounds < 2 or rounds % 2:
            raise InvalidBatchConfigError("批量对弈轮数必须为不小于 2 的偶数")
        if rounds > MAX_ROUNDS:
            raise InvalidBatchConfigError(f"批量对弈轮数不能超过 {MAX_ROUNDS}")
        if not 1 <= max_plies <= MAX_PLIES_CAP:
            raise InvalidBatchConfigError(f"单局步数上限需在 1..{MAX_PLIES_CAP} 之间")
        a = resolve_engine(white_model, white_arg)
        b = resolve_engine(black_model, black_arg)
        native = a.native and b.native
        job = {"kind": "match", "a": a.spec, "b": b.spec, "names": {"A": a.label, "B": b.label},
               "match": {"pairs": rounds // 2, "seed": secrets.randbits(31),
                         "max_plies": max_plies, "openings": "bundled",
                         "concurrency": NATIVE_CONCURRENCY if native else 1,
                         "workers": 1 if native else WRAPPED_WORKERS},
               "gpu_mib": self.gpu_mib}
        with self._lock:
            self.refresh()
            if self.is_running:
                raise BatchAlreadyRunningError("批量对弈已在运行中")
            batch_id = uuid.uuid4().hex
            row = {"id": batch_id, "created_at": _now(), "end_time": None,
                   "white_model": a.model, "white_arg": a.arg,
                   "black_model": b.model, "black_arg": b.arg,
                   "rounds_planned": rounds, "rounds_completed": 0, "status": "running",
                   "tally": {"a_win": 0, "b_win": 0, "draw": 0}, "error": None}
            handle = self._launcher.submit(batch_id, job)
            self._storage.save_batch(row)
            self._current = self._attach(row, handle)
        state = handle.wait_started(START_WAIT_S)
        self.refresh()
        if state == "gpu_busy":
            raise GpuBusyError(handle.status().get("error") or "GPU 显存不足，拒绝启动批量对弈")
        return self.snapshot()

    def stop(self) -> dict:
        with self._lock:
            cur = self._current
        if cur is not None and cur["row"]["status"] == "running":
            cur["handle"].stop(grace_s=10)
            cur["handle"].wait(timeout=15)
        self.refresh()
        return self.snapshot()

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._current is not None and self._current["row"]["status"] == "running"

    # ---------------------------------------------------------------- 状态同步

    def refresh(self) -> None:
        """把 job 目录的进度同步进 arena_storage（幂等；轮询与后台监视线程都会调用）。"""
        with self._lock:
            cur = self._current
            if cur is None:
                return
            handle, row = cur["handle"], cur["row"]
            if row["status"] != "running" and self._storage.get_batch(row["id"]) is None:
                self._current = None          # 已结束的批次被运维从库里删了：不再展示、也不回写
                return
            before = dict(row, tally=dict(row["tally"]))
            records = handle.records()
            tally = {"a_win": 0, "b_win": 0, "draw": 0}
            for rec in records:
                score = rec.get("a_score")
                tally["a_win" if score == 1.0 else "b_win" if score == 0.0 else "draw"] += 1
                if rec["game"] not in cur["ingested"]:
                    self._save_record(rec, row, cur)
                    cur["ingested"].add(rec["game"])
            row["tally"] = tally
            row["rounds_completed"] = len(records)
            if row["status"] == "running":
                state = handle.state()
                if state in STATES_FINAL or state == "died":
                    status = handle.status()
                    row["status"] = "interrupted" if state == "died" else state
                    row["error"] = status.get("error") or (
                        "job 进程意外退出（被杀或服务重启）" if state == "died" else None)
                    row["end_time"] = _now()
            if row != before:
                self._storage.save_batch(row)

    def _save_record(self, rec: dict, row: dict, cur: dict) -> None:
        a, b = cur["a"], cur["b"]
        (wm, wa), (bm, ba) = (a, b) if rec["white"] == "A" else (b, a)
        end = time.time()
        self._storage.save_record({
            "id": f"{row['id']}-g{rec['game']}",
            "created_at": _iso(end - float(rec.get("elapsed_s") or 0)), "end_time": _iso(end),
            "white_model": wm, "white_arg": wa, "black_model": bm, "black_arg": ba,
            "moves": list(rec.get("opening", [])) + list(rec["moves"]),
            "ply_count": rec["plies"], "result": rec["result"], "winner": _winner(rec["result"]),
            "termination_reason": rec["termination"], "batch_id": row["id"],
        })

    def snapshot(self) -> dict[str, Any]:
        self.refresh()
        with self._lock:
            cur = self._current
            if cur is None:
                latest = self._storage.list_batches(limit=1)
                return {"batch": latest[0] if latest else None, "workers": [], "is_running": False}
            batch = dict(cur["row"], tally=dict(cur["row"]["tally"]))
            handle = cur["handle"]
            running = batch["status"] == "running"
        status = handle.status()
        batch["summary"] = status.get("summary")
        batch["job_state"] = status.get("state")
        workers = self._workers(handle, cur) if running else []
        return {"batch": batch, "workers": workers, "is_running": running}

    @staticmethod
    def _workers(handle: JobHandle, cur: dict) -> list[dict]:
        out = []
        games = sorted(handle.live().get("games", {}).values(), key=lambda g: g["game"])
        for i, g in enumerate(games):
            try:
                board, sans = _replay(g.get("fen"), list(g.get("opening", [])) + g["moves"])
            except ValueError:
                continue                      # 快照与着法不一致（不应发生）：跳过这一格
            a_white = g.get("white") == "A"
            (wm, wa), (bm, ba) = (cur["a"], cur["b"]) if a_white else (cur["b"], cur["a"])
            details = g.get("details") or []
            out.append({
                "index": i, "round_index": g["game"], "status": "playing",
                "white_model": wm, "white_arg": wa, "black_model": bm, "black_arg": ba,
                "opening": " ".join(g.get("opening", [])) or None,
                "fen": board.fen(), "turn": "white" if board.turn else "black",
                "last_move": board.move_stack[-1].uci() if board.move_stack else None,
                "san_history": sans[-SAN_TAIL:], "ply_count": len(board.move_stack),
                "in_check": board.is_check(),
                "engine_ms": details[-1].get("ms") if details else None,
            })
        return out


# ====================================================================== 观战

@dataclass
class ArenaWatch:
    arena_id: str
    white: EngineRef
    black: EngineRef
    handle: JobHandle
    start_fen: str | None
    created_at: str = field(default_factory=_now)
    board: chess.Board = field(default_factory=chess.Board)
    san_history: list[str] = field(default_factory=list)
    saved: bool = False
    stopped: bool = False
    final: dict | None = None            # {"result", "winner", "termination_reason"}
    storage: arena_storage.ArenaStorage | None = None
    _op_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # ---------------------------------------------------------------- 读 job

    def _source(self) -> tuple[list[str], list[dict], dict | None]:
        """(着法, 每步细节, 终局记录)。job 已写结果时以 results.jsonl 为准（live.json 最后一次落盘可能稍晚）。"""
        live = self.handle.live().get("games", {}).get("0") or {}
        moves, details = list(live.get("moves", [])), list(live.get("details", []))
        records = self.handle.records()
        if records:
            moves = list(records[0]["moves"])
            return moves, details, records[0]
        return moves, details, None

    def _outcome(self, record: dict | None) -> dict | None:
        verdict = classify(self.board)
        if verdict is not None:
            return {"result": verdict.result, "winner": _winner(verdict.result),
                    "termination_reason": verdict.termination}
        if record is not None and len(self.board.move_stack) >= len(record["moves"]):
            return {"result": record["result"], "winner": _winner(record["result"]),
                    "termination_reason": record["termination"]}
        return None

    def _reveal(self, uci: str, detail: dict | None) -> dict:
        move = chess.Move.from_uci(uci)
        if move not in self.board.legal_moves:
            raise ArenaError(f'job 给出的着法 "{uci}" 在当前局面下不合法')
        mover_white = self.board.turn == chess.WHITE
        san = self.board.san(move)
        self.board.push(move)
        self.san_history.append(san)
        info = (detail or {}).get("info") or {}
        return {"move": uci, "san": san, "engine_ms": (detail or {}).get("ms"),
                "eval": _eval_of(detail, mover_white),
                "engine_details": {"source": (detail or {}).get("source"), **info},
                "turn": "white" if mover_white else "black"}

    def step(self, wait_s: float = STEP_WAIT_S) -> dict[str, Any]:
        with self._op_lock:
            deadline = time.monotonic() + wait_s
            while True:
                if self.final is not None:
                    return self._step_payload(None, over=True)
                moves, details, record = self._source()
                ply = len(self.board.move_stack)
                if ply < len(moves):
                    data = self._reveal(moves[ply], details[ply] if ply < len(details) else None)
                    self.final = self._outcome(record)
                    if self.final is not None:
                        self._save()
                    return self._step_payload(data, over=self.final is not None)
                self.final = self._outcome(record)
                if self.final is not None:
                    self._save()
                    continue
                if self.stopped:
                    raise ArenaError("对局已终止")
                state = self.handle.state()
                if state in ("error", "gpu_busy", "stopped", "died"):
                    err = self.handle.status().get("error") or state
                    raise ArenaError(f"对局进程已结束（{state}）：{err}")
                if time.monotonic() >= deadline:
                    return self._step_payload(None, over=False, pending=True)
                time.sleep(0.1)

    def _step_payload(self, data: dict | None, *, over: bool, pending: bool = False) -> dict:
        final = self.final or {}
        out = {"move": None, "san": None, "engine_ms": 0, "eval": None, "engine_details": {},
               **(data or {}),
               "fen": self.board.fen(),
               "is_game_over": over, "pending": pending,
               "result": final.get("result", "*"), "winner": final.get("winner"),
               "termination_reason": final.get("termination_reason"),
               "ply_count": len(self.board.move_stack),
               "last_move": self.board.move_stack[-1].uci() if self.board.move_stack else None,
               "in_check": self.board.is_check(), "san_history": list(self.san_history)}
        if data is None:
            out["turn"] = "white" if self.board.turn else "black"
        return out

    def state(self) -> dict[str, Any]:
        with self._op_lock:
            final = self.final or {}
            moves, _, _ = self._source() if self.final is None and not self.stopped else ([], [], None)
            return {
                "arena_id": self.arena_id, "created_at": self.created_at,
                "white_model": self.white.model, "white_arg": self.white.arg,
                "black_model": self.black.model, "black_arg": self.black.arg,
                "fen": self.board.fen(), "turn": "white" if self.board.turn else "black",
                "legal_moves": [m.uci() for m in self.board.legal_moves],
                "is_game_over": self.final is not None or self.stopped,
                "result": final.get("result", "*"),
                "winner": final.get("winner", "stopped" if self.stopped else None),
                "termination_reason": final.get("termination_reason",
                                                 "stopped" if self.stopped else None),
                "ply_count": len(self.board.move_stack),
                "moves": " ".join(m.uci() for m in self.board.move_stack),
                "batch_id": None,
                "last_move": self.board.move_stack[-1].uci() if self.board.move_stack else None,
                "in_check": self.board.is_check(), "san_history": list(self.san_history),
                "pending_moves": max(0, len(moves) - len(self.board.move_stack)),
                "job_state": self.handle.state(),
            }

    # ---------------------------------------------------------------- 收尾

    def finish_unwatched(self) -> None:
        """没人继续单步时，job 下完就把整局入库（监视线程调用）。"""
        with self._op_lock:
            if self.saved or self.stopped:
                return
            moves, details, record = self._source()
            if record is None:
                return
            for i in range(len(self.board.move_stack), len(moves)):
                self._reveal(moves[i], details[i] if i < len(details) else None)
            self.final = self._outcome(record)
            if self.final is not None:
                self._save()

    def close(self) -> None:
        with self._op_lock:
            if not self.saved and self.final is None:
                self.stopped = True
                self._save(stopped=True)
            self.handle.stop(grace_s=5)
            self.handle.wait(timeout=10)
        shutil.rmtree(self.handle.dir, ignore_errors=True)

    def _save(self, stopped: bool = False) -> None:
        if self.saved:
            return
        self.saved = True
        final = self.final or {}
        record = {
            "id": self.arena_id, "created_at": self.created_at, "end_time": _now(),
            "white_model": self.white.model, "white_arg": self.white.arg,
            "black_model": self.black.model, "black_arg": self.black.arg,
            "moves": [m.uci() for m in self.board.move_stack],
            "ply_count": len(self.board.move_stack),
            "result": "*" if stopped else final.get("result", "*"),
            "winner": "stopped" if stopped else final.get("winner"),
            "termination_reason": "stopped" if stopped else final.get("termination_reason"),
            "batch_id": None,
        }
        try:
            (self.storage or arena_storage.storage).save_record(record)
        except Exception:
            logger.exception("保存观战记录失败 (arena_id=%s)", self.arena_id)


class ArenaService:
    """观战会话：每局一个 game job，最多 MAX_ARENA_SESSIONS 局（LRU 淘汰）。"""

    def __init__(self, storage: arena_storage.ArenaStorage | None = None, *,
                 jobs_dir: Path | None = None, python: str | None = None,
                 env_extra: dict | None = None, gpu_mib: int = GAME_GPU_MIB,
                 lease_dir: str | None = None, max_sessions: int = MAX_ARENA_SESSIONS,
                 max_plies: int = ARENA_MAX_PLIES):
        self._storage = storage or arena_storage.storage
        self._launcher = _JobLauncher(jobs_dir, python, env_extra, lease_dir)
        self.gpu_mib = gpu_mib
        self.max_plies = max_plies
        self._max = max_sessions
        self._sessions: OrderedDict[str, ArenaWatch] = OrderedDict()
        self._lock = threading.RLock()

    def create(self, white_model: str, white_arg: str | None = None, black_model: str = "",
               black_arg: str | None = None, fen: str | None = None) -> ArenaWatch:
        start_fen = None
        if fen:
            try:
                board = chess.Board(fen)
            except ValueError as e:
                raise InvalidFenError(f'无法解析的 FEN "{fen}": {e}') from e
            if board.fen() != chess.STARTING_FEN:
                start_fen = board.fen()
        white = resolve_engine(white_model, white_arg)
        black = resolve_engine(black_model, black_arg)
        arena_id = uuid.uuid4().hex
        job = {"kind": "game", "a": white.spec, "b": black.spec,
               "names": {"A": white.label, "B": black.label},
               "game": {"max_plies": self.max_plies, "seed": secrets.randbits(31),
                        "opening": [], "fen": start_fen},
               "gpu_mib": self.gpu_mib}
        with self._lock:
            victims = []
            while len(self._sessions) >= self._max:
                victims.append(self._sessions.popitem(last=False)[1])
        for v in victims:                     # 先放掉旧局的 GPU 租约，新局才申请得到
            self._close_quietly(v)
        handle = self._launcher.submit(arena_id, job)
        watch = ArenaWatch(arena_id, white, black, handle, start_fen,
                           board=chess.Board(start_fen) if start_fen else chess.Board(),
                           storage=self._storage)
        state = handle.wait_started(START_WAIT_S)
        if state in ("gpu_busy", "error", "died"):
            err = handle.status().get("error") or state
            handle.wait(timeout=10)
            shutil.rmtree(handle.dir, ignore_errors=True)
            if state == "gpu_busy":
                raise GpuBusyError(err)
            raise ArenaError(f"观战对局启动失败：{err}")
        with self._lock:
            self._sessions[arena_id] = watch
        return watch

    def get(self, arena_id: str) -> ArenaWatch:
        with self._lock:
            watch = self._sessions.get(arena_id)
            if watch is None:
                raise ArenaNotFoundError(f'竞技场对局 "{arena_id}" 不存在或已被清理')
            self._sessions.move_to_end(arena_id)
            return watch

    def close(self, arena_id: str) -> None:
        with self._lock:
            watch = self._sessions.pop(arena_id, None)
        if watch is None:
            raise ArenaNotFoundError(f'竞技场对局 "{arena_id}" 不存在或已被清理')
        watch.close()

    def refresh(self) -> None:
        with self._lock:
            watches = list(self._sessions.values())
        for w in watches:
            try:
                w.finish_unwatched()
            except Exception:
                logger.exception("观战对局 %s 收尾失败", w.arena_id)

    def close_all(self) -> None:
        with self._lock:
            watches = list(self._sessions.values())
            self._sessions.clear()
        for w in watches:
            self._close_quietly(w)

    @staticmethod
    def _close_quietly(watch: ArenaWatch) -> None:
        try:
            watch.close()
        except Exception:
            logger.exception("关闭观战对局 %s 失败", watch.arena_id)


# ====================================================================== 后台监视

class Monitor(threading.Thread):
    """定期同步批次进度、把没人看的观战对局收尾入库；无人轮询时也能按时入库。"""

    def __init__(self, batch: BatchService, arena: ArenaService, interval: float = 2.0):
        super().__init__(daemon=True, name="unichess-job-monitor")
        self.batch, self.arena, self.interval = batch, arena, interval
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(self.interval):
            for fn in (self.batch.refresh, self.arena.refresh):
                try:
                    fn()
                except Exception:
                    logger.exception("job 监视线程出错")

    def stop(self) -> None:
        self._stop.set()


_services: dict[str, Any] = {}
_services_lock = threading.Lock()


def batch_service() -> BatchService:
    with _services_lock:
        if "batch" not in _services:
            _services["batch"] = BatchService()
        return _services["batch"]


def arena_service() -> ArenaService:
    with _services_lock:
        if "arena" not in _services:
            _services["arena"] = ArenaService()
        return _services["arena"]
