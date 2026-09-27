"""UniChessServer 适配层：M2 = 上游 chess_ai neural v2.0.0（策略网 + 价值网 + negamax）。

包是上游冻结产物，本文件只做「接线」：把上游的 ChessCNN / ResidualValueModel /
NeuralSearchV4 接到 UniChessServer 的 GameEngine 六方法契约上，不改上游任何逻辑。

与 M3（src/chess_ai 命名空间包）的关键差别：上游 v2.0.0 是一堆**裸顶层模块**
（model_cnn / search_engine / neural_fast ...），靠自身目录在 sys.path 上才能互相
import；而且 search_engine.policy_order 里有函数内惰性 import，sys.path 必须一直
保留该目录，不能 import 完就撤。

线程安全约定（踩过坑）：FastEvaluator 复用输入缓冲区、且文档明言
"not safe for concurrent calls"，因此**每个会话一份 TracedEvaluator + 一份搜索**；
策略网与价值网只做只读前向，进程级共享（与 M3 的「共享权重、每会话搜索树」同构）。

线程数（踩过坑）：本机 20 逻辑核，torch 默认开满会让 19×8×8 的微型 CNN 慢 450 倍
（评估 0.119ms → 54.5ms，2 秒只够走完 depth 1）。默认压到 4 线程，与上游
play_v4.py 的做法一致；这是进程级全局设置，服务进程里其余模型都跑 GPU、不受影响。
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

import chess
import torch

# 包根：默认 <import 根>/M2；测试/异地部署可用 UNICHESS_M2_ROOT 覆盖。
# 本文件在 <import 根>/Server/models/M2/engine.py，故 import 根是 parents[3]。
_HERE = Path(__file__).resolve()
_IMPORT_ROOT = _HERE.parents[3]


def _package_root() -> Path:
    import os
    env = os.environ.get('UNICHESS_M2_ROOT')
    return Path(env).resolve() if env else (_IMPORT_ROOT / 'M2')


_PACKAGE_ROOT = _package_root()
_SRC = str(_PACKAGE_ROOT)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from model_cnn import ChessCNN  # noqa: E402
from model_residual_value import ResidualValueModel  # noqa: E402
from neural_inference_v3 import TracedEvaluator  # noqa: E402
from neural_search_v4 import NeuralSearchV4  # noqa: E402

POLICY_WEIGHTS = 'chess_model_balanced.pt'
POLICY_SHA256 = '60ab3e89ebf853380f67777825645c12c3716221f58f3b311a713ffa6d60ce00'
VALUE_GLOB = 'value_full_runs/*/value_full_epoch2.pt'
VALUE_SHA256 = '4e44421b422c8467a32c06c85a31a5591b8397ae24c2f54fdd39c778fe7dd4a0'

# 这台机器 20 逻辑核，torch 默认开满 20 线程跑 19×8×8 的微型 CNN 会是灾难：
# 实测价值评估从 0.119ms（8 线程）劣化到 54.5ms（20 线程），搜索从 15000 n/s
# 掉到 47 n/s、2 秒只能走完 depth 1。上游 play_v4.py 自己也写了
# set_num_threads(4)/set_num_interop_threads(1)，这里沿用。
# 注意 set_num_threads 是进程级全局设置：服务进程里 M3/T/R/S 都跑 GPU，
# 不用 CPU 线程池，因此不受影响（已记入 Server/AGENTS.md）。
DEFAULT_THREADS = 4

_threads_applied = False
_threads_lock = threading.Lock()


def _apply_threads(n: int | None) -> None:
    """设置 torch 线程数（进程级，幂等）。n 为 None 表示不动全局设置。"""
    global _threads_applied
    with _threads_lock:
        if _threads_applied:
            return
        if n is None:
            return
        try:
            torch.set_num_threads(int(n))
        except Exception:
            return
        try:
            # 只能在并行工作启动前设置一次，失败（已初始化）就跳过
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        _threads_applied = True

_model_cache: dict[str, Any] = {}
_cache_lock = threading.Lock()
_verified: set[str] = set()


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _verify_once(path: Path, want: str) -> None:
    key = str(path)
    if key in _verified:
        return
    if not path.is_file():
        raise FileNotFoundError(f'缺少 M2 包内文件: {path}')
    got = _sha256(path)
    if got != want:
        raise RuntimeError(
            f'{path.name} 校验和不符：期望 {want}，实际 {got}（包被改动过？）'
        )
    _verified.add(key)


def _value_weights() -> Path:
    found = sorted(_PACKAGE_ROOT.glob(VALUE_GLOB))
    if not found:
        raise FileNotFoundError(
            f'在 {_PACKAGE_ROOT} 下找不到价值网（{VALUE_GLOB}）'
        )
    return found[-1]


def _device() -> torch.device:
    return torch.device('cpu')


def _shared_models():
    """进程级共享：策略网与价值网各加载一次（只读前向，可跨会话共用）。"""
    key = f'{_PACKAGE_ROOT}|cpu'
    with _cache_lock:
        cached = _model_cache.get(key)
        if cached is not None:
            return cached
        device = _device()
        policy_path = _PACKAGE_ROOT / POLICY_WEIGHTS
        value_path = _value_weights()
        _verify_once(policy_path, POLICY_SHA256)
        _verify_once(value_path, VALUE_SHA256)

        policy = ChessCNN().to(device)
        policy.load_state_dict(
            torch.load(policy_path, map_location=device, weights_only=True)
        )
        policy.eval()
        value = ResidualValueModel().to(device)
        value.load_state_dict(
            torch.load(value_path, map_location=device, weights_only=True)
        )
        value.eval()
        _model_cache[key] = (policy, value)
        return _model_cache[key]


def _new_evaluator(value_model) -> TracedEvaluator:
    """每会话一份评估器：FastEvaluator 复用输入缓冲区，不可并发共用。"""
    evaluator = TracedEvaluator(value_model)
    evaluator.validate([chess.Board()])
    return evaluator


class GameEngine:
    """六方法契约。search_* 参数与原上游 CLI 一一对应。"""

    IMPLEMENTED: bool = True
    NOT_IMPLEMENTED_REASON: str = ''

    def __init__(self, **kwargs: Any) -> None:
        allowed = {'seconds', 'depth', 'qdepth', 'claim_draw', 'threads'}
        unknown = sorted(set(kwargs) - allowed)
        if unknown:
            raise TypeError(f'unsupported M2 engine kwargs: {unknown}')
        self.seconds = float(kwargs.get('seconds', 2.0))
        self.depth = int(kwargs.get('depth', 5))
        self.qdepth = int(kwargs.get('qdepth', 6))
        self.claim_draw = bool(kwargs.get('claim_draw', True))
        self.threads = kwargs.get('threads', DEFAULT_THREADS)
        if self.threads is not None and not 1 <= int(self.threads) <= 16:
            raise ValueError('threads 必须在 1～16 内（或 None 表示不动全局线程设置）')
        if not 0 < self.seconds <= 300:
            raise ValueError('seconds 必须在 (0, 300] 内')
        if not 1 <= self.depth <= 10:
            raise ValueError('depth 必须在 1～10 内')

        _apply_threads(self.threads)
        self._device = _device()
        self._policy, self._value = _shared_models()
        self._evaluator = _new_evaluator(self._value)
        self._search = NeuralSearchV4(
            self._evaluator,
            self.seconds,
            self.depth,
            qdepth=self.qdepth,
            claim_draw=self.claim_draw,
        )
        self._board = chess.Board()
        self._san_history: list[str] = []
        self._move_records: list[dict[str, Any]] = []
        self._state_eval: dict[str, float] | None = None
        self._lock = threading.RLock()
        self.setup()

    # ---- 内部 ----
    def _policy_order(self):
        from search_engine import policy_order
        return policy_order(self._board, self._policy, self._device)

    def _refresh_eval(self) -> None:
        """缓存当前局面的原生评估（行棋方视角 tanh），供 UI 取用。"""
        if self._evaluator is None:
            self._state_eval = None
            return
        value = float(self._evaluator(self._board))
        value = max(-1.0, min(1.0, value))
        self._state_eval = value

    def _white_eval(self) -> float | None:
        if self._state_eval is None:
            return None
        return self._state_eval if self._board.turn == chess.WHITE else -self._state_eval

    # ---- 六方法 ----
    def setup(self, fen: str | None = None):
        with self._lock:
            if fen is None:
                board = chess.Board()
            elif isinstance(fen, str):
                board = chess.Board(fen)
                if not board.is_valid():
                    raise ValueError(f'invalid FEN position: {fen!r}')
            else:
                raise TypeError(f'fen must be str or None, got {type(fen).__name__}')
            self._board = board
            self._search = NeuralSearchV4(
                self._evaluator,
                self.seconds,
                self.depth,
                qdepth=self.qdepth,
                claim_draw=self.claim_draw,
            )
            self._san_history = []
            self._move_records = []
            self._refresh_eval()
            return self.state()

    def human_move(self, uci: str):
        with self._lock:
            if not isinstance(uci, str):
                raise TypeError(f'uci must be str, got {type(uci).__name__}')
            try:
                move = chess.Move.from_uci(uci)
            except ValueError as exc:
                raise ValueError(f'invalid UCI move: {uci!r}') from exc
            if move not in self._board.legal_moves:
                raise ValueError(f'illegal move {uci!r} for current position')
            san = self._board.san(move)
            self._board.push(move)
            self._san_history.append(san)
            self._move_records.append({'uci': move.uci(), 'san': san, 'actor': 'human'})
            self._refresh_eval()
            return self.state()

    def engine_move(self):
        with self._lock:
            outcome = self._board.outcome(claim_draw=True)
            if outcome is not None:
                raise ValueError('cannot choose an engine move from a terminal position')
            started = time.perf_counter()
            preferred = self._policy_order()
            move, is_mate, stats = self._search.choose(self._board, preferred)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if move is None:
                # claim_draw=True 时上游可能回报「建议申领和棋」而非走法。
                # 服务层 classify 已把可申领和棋判为终局，正常不会走到这里；
                # 万一走到，显式报错，绝不返回假走法。
                raise ValueError('M2 认为当前局面应申领和棋，未给出走法')
            if move not in self._board.legal_moves:
                raise RuntimeError(f'MCTS returned an illegal move: {move.uci()}')
            san = self._board.san(move)
            uci = move.uci()
            self._board.push(move)
            self._san_history.append(san)
            self._move_records.append({
                'uci': uci,
                'san': san,
                'actor': 'engine',
                'engine_ms': elapsed_ms,
                'nodes': int(stats.get('nodes', 0)),
                'depth': int(stats.get('depth', 0)),
            })
            self._refresh_eval()
            return {
                'engine_move': uci,
                'mate': bool(is_mate),
                'engine_ms': elapsed_ms,
                'nodes': int(stats.get('nodes', 0)),
                'depth': int(stats.get('depth', 0)),
                'state': self.state(),
            }

    def state(self):
        with self._lock:
            outcome = self._board.outcome(claim_draw=True)
            last = self._move_records[-1] if self._move_records else None
            engine_ms = None
            nodes = None
            depth = None
            if last is not None and last.get('actor') == 'engine':
                engine_ms = last.get('engine_ms')
                nodes = last.get('nodes')
                depth = last.get('depth')
            # 前端只渲染 WDL 字典；M2 只有一个 tanh 标量，不能伪造 WDL，
            # 因此 eval 置空（UI 显示中性条），原生分放在 value_tanh 里。
            return {
                'fen': self._board.fen(),
                'turn': 'white' if self._board.turn == chess.WHITE else 'black',
                'legal_moves': sorted(m.uci() for m in self._board.legal_moves),
                'history': [m.uci() for m in self._board.move_stack],
                'san_history': list(self._san_history),
                'last_move': None if last is None else last['uci'],
                'last_move_san': None if last is None else last['san'],
                'last_move_actor': None if last is None else last['actor'],
                'engine_ms': engine_ms,
                'nodes': nodes,
                'depth': depth,
                'value_tanh': self._white_eval(),
                'eval': None,
                'in_check': self._board.is_check(),
                'game_over': outcome is not None,
                'result': None if outcome is None else outcome.result(),
            }

    def undo(self):
        with self._lock:
            for _ in range(min(2, len(self._board.move_stack))):
                self._board.pop()
                if self._san_history:
                    self._san_history.pop()
                if self._move_records:
                    self._move_records.pop()
            self._refresh_eval()
            return self.state()

    def cleanup(self):
        with self._lock:
            self._move_records = []
            self._san_history = []
            self._state_eval = None
            return None


__all__ = ['GameEngine']
