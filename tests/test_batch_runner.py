"""批量对弈 BatchRunner 单元测试。"""
from __future__ import annotations

import pathlib
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
        with mock.patch.object(model_registry, 'create_engine', side_effect=lambda n, a: FakeBatchEngine()):
            self._start(rounds=2)
        with self.assertRaises(batch_mod.BatchAlreadyRunningError):
            self._start(rounds=2)

    def test_rounds_complete_and_tally(self):
        with mock.patch.object(model_registry, 'create_engine', side_effect=lambda n, a: FakeBatchEngine()):
            self._start(rounds=4)
            snap = self._wait_done(timeout=60)
        self.assertEqual(snap['batch']['status'], 'completed')
        self.assertEqual(snap['batch']['rounds_completed'], 4)
        total = (snap['batch']['tally'] or {}).get('white', 0) + (snap['batch']['tally'] or {}).get('black', 0) + (snap['batch']['tally'] or {}).get('draw', 0)
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
            self._start(rounds=2, max_plies=2)
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


if __name__ == '__main__':
    unittest.main()
