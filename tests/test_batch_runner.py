"""批量对弈 BatchRunner 单元测试。"""
from __future__ import annotations

import pathlib
import random
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import chess  # noqa: E402
import models as model_registry  # noqa: E402

import arena_storage as a_storage  # noqa: E402
import batch_runner as batch_mod  # noqa: E402


class FakeBatchEngine:
    """契约完整的批量对弈假引擎：每步走当前局面第一个合法着。"""

    def __init__(self, **kwargs):
        self.board = chess.Board()
        self.cleaned = 0
        for k, v in kwargs.items():
            setattr(self, k, v)

    def setup(self, fen=None):
        self.board = chess.Board(fen) if fen else chess.Board()

    def human_move(self, uci):
        self.board.push(chess.Move.from_uci(uci))

    def engine_move(self):
        move = next(iter(self.board.legal_moves))
        self.board.push(move)
        return {"engine_move": move.uci(), "fen": self.board.fen()}

    def state(self):
        return {"fen": self.board.fen()}

    def undo(self):
        if self.board.move_stack:
            self.board.pop()

    def cleanup(self):
        self.cleaned += 1


class BlockingEngine(FakeBatchEngine):
    """engine_move 阻塞到 release 置位，用来让批次保持运行态。"""

    def __init__(self, release, **kwargs):
        super().__init__(**kwargs)
        self.release = release

    def engine_move(self):
        self.release.wait(timeout=30)
        return super().engine_move()


class FailingEngine(FakeBatchEngine):
    def engine_move(self):
        raise RuntimeError("boom")


def _make_runner(storage):
    batch_mod.BatchRunner.reset()
    return batch_mod.BatchRunner.get(storage=storage)


class TestBatchRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = pathlib.Path(self.tmp.name) / "test_batch.db"
        self.storage = a_storage.ArenaStorage(self.db)
        self.runner = _make_runner(self.storage)

    def tearDown(self):
        batch_mod.BatchRunner.reset()
        self.tmp.cleanup()

    def _start(self, rounds=2, max_plies=400, **kw):
        cfg = batch_mod.BatchConfig(
            white_model=kw.pop('white_model', 'A'),
            black_model=kw.pop('black_model', 'B'),
            rounds=rounds,
            max_plies=max_plies,
            min_free_gpu_mib=kw.pop('min_free_gpu_mib', 0),
            **kw,
        )
        return self.runner.start(cfg)

    def _wait_done(self, timeout=30):
        deadline = time.time() + timeout
        while time.time() < deadline:
            snap = self.runner.snapshot()
            if snap['batch'] and snap['batch'].get('status') in ('completed', 'error', 'stopped'):
                return snap
            time.sleep(0.1)
        return self.runner.snapshot()

    def test_start_returns_running_snapshot(self):
        with mock.patch.object(model_registry, 'create_engine', side_effect=lambda n, a: FakeBatchEngine()):
            snap = self._start(rounds=2)
        self.assertIsNotNone(snap['batch'])
        self.assertEqual(snap['batch']['status'], 'running')
        self.assertEqual(snap['batch']['rounds_planned'], 2)
        self.assertEqual(len(snap['workers']), batch_mod.BATCH_WORKERS)

    def test_double_start_raises(self):
        release = threading.Event()
        with mock.patch.object(model_registry, 'create_engine',
                               side_effect=lambda n, a: BlockingEngine(release)):
            self._start(rounds=2)
            try:
                with self.assertRaises(batch_mod.BatchAlreadyRunningError):
                    self._start(rounds=2)
            finally:
                release.set()

    def test_rounds_complete_and_tally(self):
        with mock.patch.object(model_registry, 'create_engine', side_effect=lambda n, a: FakeBatchEngine()):
            self._start(rounds=4)
            snap = self._wait_done(timeout=60)
        self.assertEqual(snap['batch']['status'], 'completed')
        self.assertEqual(snap['batch']['rounds_completed'], 4)
        tally = snap['batch']['tally']
        self.assertEqual(set(tally), {'a_win', 'b_win', 'draw'})
        total = tally['a_win'] + tally['b_win'] + tally['draw']
        self.assertEqual(total, 4)

    def test_color_alternation(self):
        with mock.patch.object(model_registry, 'create_engine', side_effect=lambda n, a: FakeBatchEngine()):
            self._start(rounds=4, white_model='A', black_model='B')
            snap = self._wait_done(timeout=60)
        recs = self.storage.list_records(batch_id=snap['batch']['id'])
        self.assertEqual(len(recs), 4)
        a_white = [r for r in recs if r['white_model'] == 'A' and r['black_model'] == 'B']
        b_white = [r for r in recs if r['white_model'] == 'B' and r['black_model'] == 'A']
        self.assertEqual(len(a_white), 2)
        self.assertEqual(len(b_white), 2)

    def test_ply_cap_draw(self):
        with mock.patch.object(model_registry, 'create_engine', side_effect=lambda n, a: FakeBatchEngine()):
            self._start(rounds=2, max_plies=2, use_openings=False)
            snap = self._wait_done(timeout=60)
        recs = self.storage.list_records(batch_id=snap['batch']['id'])
        self.assertEqual(len(recs), 2)
        for rec in recs:
            self.assertEqual(rec['result'], '1/2-1/2')
            self.assertEqual(rec['winner'], 'draw')
            self.assertEqual(rec['termination_reason'], 'max_plies')
            self.assertEqual(rec['ply_count'], 2)

    def test_records_have_batch_id(self):
        with mock.patch.object(model_registry, 'create_engine', side_effect=lambda n, a: FakeBatchEngine()):
            snap = self._start(rounds=2)
            snap = self._wait_done(timeout=60)
        recs = self.storage.list_records(batch_id=snap['batch']['id'])
        self.assertEqual(len(recs), 2)
        for rec in recs:
            self.assertEqual(rec['batch_id'], snap['batch']['id'])

    def test_tally_is_per_model_not_per_color(self):
        # A 永远赢：A 执白时白胜、A 执黑时黑胜，按模型计数应全部记给 A
        rounds = [
            batch_mod.BatchRound(0, 'A', None, 'B', None, True),
            batch_mod.BatchRound(1, 'B', None, 'A', None, False),
        ]
        self.runner._batch = {"id": "x", "created_at": "2026-01-01T00:00:00Z", "rounds_completed": 0,
                              "tally": {"a_win": 0, "b_win": 0, "draw": 0}}
        self.runner._on_game_complete(rounds[0], "white")
        self.runner._on_game_complete(rounds[1], "black")
        self.runner._on_game_complete(rounds[1], "draw")
        self.assertEqual(self.runner._batch["tally"], {"a_win": 2, "b_win": 0, "draw": 1})

    def test_opening_pairs_share_line_and_swap_colors(self):
        openings = [["e2e4", "e7e5"], ["d2d4", "d7d5"], ["c2c4"]]
        cfg = batch_mod.BatchConfig(white_model='A', black_model='B', rounds=6)
        rounds = batch_mod.build_rounds(cfg, openings, random.Random(0))
        self.assertEqual(len(rounds), 6)
        for first, second in zip(rounds[::2], rounds[1::2]):
            self.assertEqual(first.opening, second.opening)
            self.assertTrue(first.a_is_white)
            self.assertFalse(second.a_is_white)
            self.assertEqual((first.white_model, second.white_model), ('A', 'B'))
        self.assertEqual(len({tuple(r.opening) for r in rounds}), 3)

    def test_bundled_openings_are_legal(self):
        openings = batch_mod.load_openings()
        self.assertGreaterEqual(len(openings), 16)
        for line in openings:
            board = chess.Board()
            for uci in line:
                move = chess.Move.from_uci(uci)
                self.assertIn(move, board.legal_moves, f"{line}: {uci}")
                board.push(move)

    def test_records_start_from_opening(self):
        with mock.patch.object(model_registry, 'create_engine', side_effect=lambda n, a: FakeBatchEngine()):
            self._start(rounds=2)
            snap = self._wait_done(timeout=60)
        recs = self.storage.list_records(batch_id=snap['batch']['id'])
        self.assertEqual(len(recs), 2)
        prefixes = {" ".join(r['moves'].split()[:4]) for r in recs}
        self.assertEqual(len(prefixes), 1)  # 同一开局对
        line = prefixes.pop().split()
        self.assertTrue(any(o[:4] == line for o in batch_mod.load_openings()))

    def test_worker_error_stops_whole_batch(self):
        with mock.patch.object(model_registry, 'create_engine', side_effect=lambda n, a: FailingEngine()):
            self._start(rounds=20)
            snap = self._wait_done(timeout=60)
        self.assertEqual(snap['batch']['status'], 'error')
        self.assertIn('boom', snap['batch']['error'])
        # 等全部 worker 退出后才允许新批次
        deadline = time.time() + 10
        while self.runner.is_running and time.time() < deadline:
            time.sleep(0.05)
        self.assertFalse(self.runner.is_running)
        self.assertEqual(snap['batch']['rounds_completed'], 0)
        self.assertEqual(self.storage.list_records(batch_id=snap['batch']['id']), [])

    def test_error_status_only_after_all_workers_exit(self):
        release = threading.Event()
        engines = iter([FailingEngine(), FailingEngine()] + [BlockingEngine(release) for _ in range(40)])
        lock = threading.Lock()

        def factory(n, a):
            with lock:
                return next(engines)

        with mock.patch.object(model_registry, 'create_engine', side_effect=factory):
            self._start(rounds=8)
            deadline = time.time() + 10
            while self.runner.snapshot()['batch']['error'] is None and time.time() < deadline:
                time.sleep(0.05)
            snap = self.runner.snapshot()
            self.assertTrue(snap['is_running'])  # 其余 worker 仍在收尾
            self.assertEqual(snap['batch']['status'], 'running')
            release.set()
            snap = self._wait_done(timeout=30)
        self.assertEqual(snap['batch']['status'], 'error')

    def test_start_validation(self):
        for rounds in (0, 3, batch_mod.MAX_ROUNDS + 2):
            with self.assertRaises(batch_mod.BatchError):
                self._start(rounds=rounds)
        with self.assertRaises(batch_mod.BatchError):
            self._start(rounds=2, max_plies=batch_mod.MAX_PLIES_CAP + 1)

    def test_gpu_busy_refuses_start(self):
        with mock.patch.object(batch_mod, 'gpu_free_mib', return_value=100):
            with self.assertRaises(batch_mod.GpuBusyError):
                self._start(rounds=2, min_free_gpu_mib=4096)
        with mock.patch.object(batch_mod, 'gpu_free_mib', return_value=None), \
                mock.patch.object(model_registry, 'create_engine', side_effect=lambda n, a: FakeBatchEngine()):
            self._start(rounds=2, min_free_gpu_mib=4096)  # 查询不到显存时不阻塞

    def test_storage_migration_adds_batch_id(self):
        import sqlite3
        conn = sqlite3.connect(str(self.db))
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS t (id TEXT PRIMARY KEY)")
            conn.commit()
        finally:
            conn.close()
        storage2 = a_storage.ArenaStorage(self.db)
        rec = {"id": "mig-1", "created_at": "2026-01-01T00:00:00Z", "end_time": "2026-01-01T00:01:00Z",
               "white_model": "A", "white_arg": None, "black_model": "B", "black_arg": None,
               "moves": "e2e4", "ply_count": 1, "result": "1-0", "winner": "white",
               "termination_reason": "checkmate", "batch_id": "batch-1"}
        storage2.save_record(rec)
        row = storage2.get_record("mig-1")
        self.assertIsNotNone(row)
        self.assertEqual(row.get('batch_id'), 'batch-1')


class TestBatchAdminAuth(unittest.TestCase):
    """/api/arena/batch/start 的管理员鉴权。"""

    @staticmethod
    def _request(client='127.0.0.1', **headers):
        from starlette.requests import Request
        raw = [(k.replace('_', '-').lower().encode(), v.encode()) for k, v in headers.items()]
        return Request({"type": "http", "headers": raw, "client": (client, 12345)})

    def setUp(self):
        import app
        self.app = app
        patcher = mock.patch.dict('os.environ', {'UNICHESS_ADMIN_TOKEN': 's3cret'})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _denied(self, req):
        with self.assertRaises(self.app.HTTPException) as ctx:
            self.app.require_admin(req)
        self.assertEqual(ctx.exception.status_code, 403)

    def test_valid_token_allowed_from_anywhere(self):
        self.app.require_admin(self._request('203.0.113.9', x_admin_token='s3cret', cf_connecting_ip='203.0.113.9'))

    def test_wrong_or_missing_token_denied_remotely(self):
        self._denied(self._request('203.0.113.9'))
        self._denied(self._request('203.0.113.9', x_admin_token='nope'))

    def test_direct_localhost_allowed(self):
        self.app.require_admin(self._request('127.0.0.1'))

    def test_tunneled_public_request_denied(self):
        # 隧道把公网请求转成 127.0.0.1，但 Cloudflare/nginx 会带上转发头
        self._denied(self._request('127.0.0.1', x_forwarded_for='198.51.100.7'))
        self._denied(self._request('127.0.0.1', cf_connecting_ip='198.51.100.7'))

    def test_no_token_configured_rejects_token_guessing(self):
        with mock.patch.dict('os.environ', {'UNICHESS_ADMIN_TOKEN': ''}), \
                mock.patch.object(self.app, 'ADMIN_TOKEN_FILE', pathlib.Path('/nonexistent/admin_token')):
            self._denied(self._request('203.0.113.9', x_admin_token=''))
            self._denied(self._request('203.0.113.9', x_admin_token='anything'))


if __name__ == '__main__':
    unittest.main()
