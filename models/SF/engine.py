"""UniChessServer 适配层：SF = Stockfish 19（开源最强国际象棋引擎，UCI 子进程）。

定位是 **baseline**：把世界最强开源引擎接进同一套 GameEngine 六方法契约，
让 T/R/S/M2/M3/DS 能与它人机对弈、也能用批量对弈（`/api/arena/batch`）直接
量出 Elo 差距。它不追求胜率，追求的是"同一规则、同一裁决口径下的可复现标尺"。

接线方式与 M2 同构：**适配层在仓库里，引擎本体是第三方冻结二进制**。

- 二进制不进 git（GPL，且单文件 103MB）：默认路径 `Server/tools/stockfish`
  （Windows 上 `stockfish.exe`），可用环境变量 `UNICHESS_STOCKFISH_BIN` 或
  构造参数 `binary` 覆盖；安装/校验用 `tools/fetch_stockfish.py`。
- 引擎发现是**惰性**的：模块 import 不碰二进制，`/api/models` 因此恒可列出；
  没装二进制只是 `__init__` 报错（`/api/new` 返回 500 并带上安装提示），
  不会把整个服务的模型清单带红。
- 每个会话一个 UCI 子进程（`chess.engine.SimpleEngine` 包装），进程间零共享：
  Threads/Hash 默认保守（1 线程 / 16MB），因为进程内最多 4 个对局会话、
  批量对弈还有 4 个 worker 进程，每份都开满线程会跟同机的 GPU 训练抢核。

与 M2/M3 的 eval 口径差别：SF 自带 UCI_ShowWDL，回报的是引擎自己的
`wdl`（千分比，行棋方视角），适配层只做视角换算（行棋方 → 白方）与归一化，
**不伪造**；关掉 `show_wdl` 或引擎不回报时 `eval` 为 None，引擎原始分放
`score_cp` / `score_mate` / `pv` / `nodes` / `depth`，与 M2 的 value_tanh 同哲学。
"""
from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import chess
import chess.engine

# 本文件在 <Server>/models/SF/engine.py，故 parents[2] 就是 Server 仓库目录。
_SERVER_ROOT = Path(__file__).resolve().parents[2]

# SF 19 官方 release（sf_19）各平台资产。digest 与 GitHub release API 的
# sha256 一致（tools/fetch_stockfish.py 按这张表下载并校验，改版本要同步改）。
STOCKFISH_VERSION = '19'
BINARY_CANDIDATES = ('stockfish', 'stockfish.exe')

_allowed_keys = (
    'binary', 'movetime_ms', 'depth', 'nodes', 'skill_level', 'uci_elo',
    'hash_mb', 'threads', 'show_wdl', 'move_overhead_ms',
)

# 引擎身份（id name）按二进制路径缓存：多个会话复用同一进程级结论，
# state() 里原样回报，前端/日志能看出这一局背后是哪个引擎。
_identity_cache: dict[str, str] = {}
_identity_lock = threading.Lock()


def _looks_like_stockfish(path: Path) -> bool:
    # Windows 上无执行权限位，无法用 os.access(X_OK) 判断，统一用可读+后缀探测。
    return path.is_file() and (path.suffix.lower() == '.exe' or os.access(path, os.X_OK))


def default_binary() -> Path | str:
    """按优先级定位 Stockfish 可执行文件。

    1. 环境变量 UNICHESS_STOCKFISH_BIN（显式指定，含文件名）
    2. Server 仓库内 tools/stockfish|stockfish.exe（tools/fetch_stockfish.py 的落点）
    3. PATH 上的 stockfish（系统级安装）
    """
    env = os.environ.get('UNICHESS_STOCKFISH_BIN')
    if env:
        return Path(env)
    for name in BINARY_CANDIDATES:
        candidate = _SERVER_ROOT / 'tools' / name
        if _looks_like_stockfish(candidate):
            return candidate
    found = shutil.which('stockfish')
    if found:
        return Path(found)
    return _SERVER_ROOT / 'tools' / 'stockfish'      # 不存在，交给调用方报错


def _resolve_binary(binary: str | Path | None) -> Path | str:
    if binary is None:
        binary = default_binary()
    path = Path(binary)
    if _looks_like_stockfish(path) or path.is_file():
        return path
    which = shutil.which(str(binary))
    if which:
        return Path(which)
    raise FileNotFoundError(
        f'找不到 Stockfish 可执行文件: {binary}。'
        f'安装方式：在 Server 目录运行 '
        f'python tools/fetch_stockfish.py（下载官方 release 并校验 sha256），'
        f'或用环境变量 UNICHESS_STOCKFISH_BIN 指向已有二进制。'
    )


class GameEngine:
    """六方法契约。search_* 参数与 UCI 选项一一对应（见 _allowed_keys）。"""

    IMPLEMENTED: bool = True
    NOT_IMPLEMENTED_REASON: str = ''

    def __init__(self, **kwargs: Any) -> None:
        unknown = sorted(set(kwargs) - set(_allowed_keys))
        if unknown:
            raise TypeError(f'unsupported SF engine kwargs: {unknown}')

        self.movetime_ms = int(kwargs.get('movetime_ms', 1000))
        self.depth = kwargs.get('depth')
        self.nodes = kwargs.get('nodes')
        self.skill_level = kwargs.get('skill_level')
        self.uci_elo = kwargs.get('uci_elo')
        self.hash_mb = int(kwargs.get('hash_mb', 16))
        self.threads = int(kwargs.get('threads', 1))
        self.show_wdl = bool(kwargs.get('show_wdl', True))
        self.move_overhead_ms = int(kwargs.get('move_overhead_ms', 10))
        self._binary_arg = kwargs.get('binary')

        self.depth = None if self.depth is None else int(self.depth)
        self.nodes = None if self.nodes is None else int(self.nodes)
        self.skill_level = None if self.skill_level is None else int(self.skill_level)
        self.uci_elo = None if self.uci_elo is None else int(self.uci_elo)

        # 搜索预算三选一：depth > nodes > movetime_ms（都填时按此优先级生效）。
        if self.depth is None and self.nodes is None and self.movetime_ms <= 0:
            raise ValueError('depth / nodes / movetime_ms 至少给一个有效的搜索预算')
        if self.depth is not None and not 1 <= self.depth <= 100:
            raise ValueError('depth 必须在 1～100 内')
        if self.nodes is not None and not 1 <= self.nodes <= 100_000_000:
            raise ValueError('nodes 必须在 1～100000000 内')
        if not 10 <= self.movetime_ms <= 600_000:
            raise ValueError('movetime_ms 必须在 10～600000 内')
        if not 1 <= self.hash_mb <= 2048:
            raise ValueError('hash_mb 必须在 1～2048 内（进程内最多 4 会话 × 批量 4 worker）')
        if not 1 <= self.threads <= 64:
            raise ValueError('threads 必须在 1～64 内（默认 1：与同机 GPU 训练/其它会话共存）')
        if not 0 <= self.move_overhead_ms <= 5000:
            raise ValueError('move_overhead_ms 必须在 0～5000 内')
        if self.skill_level is not None and not 0 <= self.skill_level <= 20:
            raise ValueError('skill_level 必须在 0～20 内')
        if self.uci_elo is not None and not 1320 <= self.uci_elo <= 3190:
            raise ValueError('uci_elo 必须在 1320～3190 内')
        if self.skill_level is not None and self.uci_elo is not None:
            raise ValueError('skill_level 与 uci_elo 互斥：二选一（UCI_Elo 由引擎自行换算力度）')

        self._engine = self._spawn()
        self._board = chess.Board()
        self._san_history: list[str] = []
        self._move_records: list[dict[str, Any]] = []
        self._last: dict[str, Any] | None = None
        self._lock = threading.RLock()
        self.setup()

    # ---- 内部 ----

    def _spawn(self) -> chess.engine.SimpleEngine:
        binary = _resolve_binary(self._binary_arg)
        try:
            engine = chess.engine.SimpleEngine.popen_uci(binary)
        except Exception as exc:
            raise RuntimeError(f'无法启动 Stockfish 子进程 {binary}: {exc}') from exc
        identity = str(engine.id.get('name') or 'Stockfish')
        if STOCKFISH_VERSION and STOCKFISH_VERSION not in identity:
            # 只是提示而非拒绝：官方 release tag 可能领先/滞后，按能力而非版本号卡死。
            identity = f'{identity} (期望 Stockfish {STOCKFISH_VERSION})'
        with _identity_lock:
            _identity_cache[str(binary)] = identity
        self._identity = identity
        # 资源控制必须立住：这两项 configure 失败就该让会话建不起来，
        # 而不是默默开满线程去和同机的 GPU 训练抢核。
        engine.configure({'Threads': self.threads, 'Hash': self.hash_mb,
                          'Move Overhead': self.move_overhead_ms})
        if self.uci_elo is not None:
            engine.configure({'UCI_LimitStrength': True, 'UCI_Elo': self.uci_elo})
        elif self.skill_level is not None:
            engine.configure({'Skill Level': self.skill_level})
        if self.show_wdl:
            try:
                engine.configure({'UCI_ShowWDL': True})
            except chess.engine.EngineError:
                # 老版本引擎没有这个选项：退回"不汇报 WDL"，着法与分数不受影响。
                self.show_wdl = False
        return engine

    @property
    def identity(self) -> str:
        return getattr(self, '_identity', None) or _identity_cache.get(str(self._binary_arg or ''), 'Stockfish')

    def _limit(self) -> chess.engine.Limit:
        if self.depth is not None:
            return chess.engine.Limit(depth=self.depth)
        if self.nodes is not None:
            return chess.engine.Limit(nodes=self.nodes)
        return chess.engine.Limit(time=self.movetime_ms / 1000.0)

    def _record_eval(self, info: dict) -> dict[str, Any]:
        """把一次搜索的 info 换算成白方视角的展示字段。

        score/wdl 都是**行棋方视角**（PovScore/PovWdl.turn = 行棋方），
        这里统一翻成白方视角；wdl 用引擎自报的千分比，缺失即 None（不伪造）。
        """
        score = info.get('score')
        wdl = info.get('wdl')
        white = self._board.turn == chess.WHITE
        out: dict[str, Any] = {
            'score_cp': None,
            'score_mate': None,
            'eval': None,
            'pv': [m.uci() for m in (info.get('pv') or [])],
            'depth': info.get('depth'),
            'seldepth': info.get('seldepth'),
            'nodes': info.get('nodes'),
            'nps': info.get('nps'),
        }
        if score is not None:
            relative = score.relative
            cp = relative.score()
            mate = relative.mate()
            if mate is not None:
                out['score_mate'] = mate if white else -mate
            elif cp is not None:
                out['score_cp'] = cp if white else -cp
        if wdl is not None:
            relative = wdl.relative
            wins, draws, losses = relative.wins, relative.draws, relative.losses
            if not white:
                wins, losses = losses, wins
            total = wins + draws + losses
            if total > 0:
                out['eval'] = {
                    'win': wins / total,
                    'draw': draws / total,
                    'loss': losses / total,
                    'pov': 'white',
                }
        return out

    # ---- 六方法 ----

    def setup(self, fen: str | None = None):
        with self._lock:
            if fen is None:
                board = chess.Board()
            elif isinstance(fen, str):
                try:
                    board = chess.Board(fen)
                except ValueError as exc:
                    raise ValueError(f'invalid FEN position: {fen!r}') from exc
                if not board.is_valid():
                    raise ValueError(f'invalid FEN position: {fen!r}')
            else:
                raise TypeError(f'fen must be str or None, got {type(fen).__name__}')
            self._board = board
            self._san_history = []
            self._move_records = []
            self._last = None
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
            # 人类走子后局面变了，上一次搜索的分数/pv 已不属于当前局面：
            # 不像 M2 有廉价价值网可以重算，这里直接置空，等引擎应答再给分。
            self._last = None
            return self.state()

    def engine_move(self):
        with self._lock:
            outcome = self._board.outcome(claim_draw=True)
            if outcome is not None:
                raise ValueError('cannot choose an engine move from a terminal position')
            started = time.perf_counter()
            try:
                result = self._engine.play(
                    self._board.copy(), self._limit(), info=chess.engine.INFO_ALL
                )
            except chess.engine.EngineError as exc:
                raise RuntimeError(f'Stockfish 搜索失败: {exc}') from exc
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            move = result.move
            if move is None or move not in self._board.legal_moves:
                raise RuntimeError(f'Stockfish 返回了不可用着法: {move!r}')
            info = dict(result.info) if result.info else {}
            evaluated = self._record_eval(info)
            san = self._board.san(move)
            uci = move.uci()
            self._board.push(move)
            self._san_history.append(san)
            record = {
                'uci': uci,
                'san': san,
                'actor': 'engine',
                'engine_ms': elapsed_ms,
                'evaluated': evaluated,
            }
            self._move_records.append(record)
            self._last = record
            payload = {
                'engine_move': uci,
                'mate': evaluated['score_mate'] is not None,
                'engine_ms': elapsed_ms,
                'source': 'stockfish',
                'state': self.state(),
            }
            payload.update(evaluated)
            return payload

    def state(self):
        with self._lock:
            outcome = self._board.outcome(claim_draw=True)
            last = self._move_records[-1] if self._move_records else None
            evaluated = self._last['evaluated'] if (self._last is not None) else None
            engine_ms = None
            nodes = None
            depth = None
            if last is not None and last.get('actor') == 'engine':
                engine_ms = last.get('engine_ms')
            if evaluated is not None:
                nodes = evaluated.get('nodes')
                depth = evaluated.get('depth')
            return {
                'fen': self._board.fen(),
                'turn': 'white' if self._board.turn == chess.WHITE else 'black',
                'legal_moves': sorted(m.uci() for m in self._board.legal_moves),
                'history': [m.uci() for m in self._board.move_stack],
                'san_history': list(self._san_history),
                'last_move': None if last is None else last['uci'],
                'last_move_san': None if last is None else last['san'],
                'last_move_actor': None if last is None else last['actor'],
                'engine': self.identity,
                'engine_ms': engine_ms,
                'nodes': nodes,
                'depth': depth,
                'seldepth': None if evaluated is None else evaluated.get('seldepth'),
                'nps': None if evaluated is None else evaluated.get('nps'),
                'score_cp': None if evaluated is None else evaluated.get('score_cp'),
                'score_mate': None if evaluated is None else evaluated.get('score_mate'),
                'pv': [] if evaluated is None else evaluated.get('pv', []),
                'eval': None if evaluated is None else evaluated.get('eval'),
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
            self._last = None
            return self.state()

    def cleanup(self):
        with self._lock:
            engine, self._engine = self._engine, None
            self._move_records = []
            self._san_history = []
            self._last = None
        if engine is None:
            return None
        try:
            engine.quit()
        except Exception:
            # quit 失败（进程已死 / 卡死）时兜底强杀：SF 常驻线程必须回收，
            # 否则批量对弈反复建会话会把机器啃干净。
            try:
                engine.close()
            except Exception:
                pass
        return None

    def __del__(self):
        # 会话被淘汰时 session_manager 总归会调 cleanup()；这里只是进程被杀
        # （KeyboardInterrupt / os._exit）时的最后防线，异常一律吞掉。
        try:
            self.cleanup()
        except Exception:
            pass


__all__ = ['GameEngine', 'default_binary']
