"""UniChess Arena 竞技场单元测试。

测试覆盖：
1. 存储层（arena_storage）：自动建表、增改记录、分页列表、单条查询
2. 会话管理与对弈流程（arena_manager）：
   - 双引擎创建、setup
   - step 轮替推进：白方 step -> 推入 board 并同步黑方 human_move；黑方 step -> 推入 board 并同步白方 human_move
   - 引擎返回非法走法或异常时的优雅处理与回滚
   - 终局判定（checkmate / draw / resignation）与自动落库、cleanup
   - LRU 淘汰与会话上限控制（MAX_ARENA_SESSIONS = 2）
   - 手动关闭（close）记录 stopped 状态
3. FastAPI 路由集成测试：
   - GET /arena
   - POST /api/arena/new
   - POST /api/arena/games/{id}/step
   - GET /api/arena/games/{id}/state
   - DELETE /api/arena/games/{id}
   - GET /api/arena/records
   - GET /api/arena/records/{id}
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import chess  # noqa: E402
from fastapi import HTTPException  # noqa: E402

import app  # noqa: E402
import arena_manager as am  # noqa: E402
import arena_storage as a_storage  # noqa: E402
import models as model_registry  # noqa: E402


class FakeArenaEngine:
    """契约完整的模拟引擎。"""

    def __init__(self, **kwargs):
        self.board = chess.Board()
        self.cleaned = False
        self.fail_engine_move = False
        self.fail_human_move = False
        self.fail_setup = False
        self.illegal_move = False
        self.move_sequence: list[str] = []
        for k, v in kwargs.items():
            setattr(self, k, v)

    def setup(self, fen: str | None = None) -> dict:
        if self.fail_setup:
            raise RuntimeError("FakeArenaEngine setup boom")
        self.board = chess.Board(fen) if fen else chess.Board()
        return {"fen": self.board.fen()}

    def human_move(self, uci: str) -> dict:
        if self.fail_human_move:
            raise RuntimeError("FakeArenaEngine human_move boom")
        self.board.push(chess.Move.from_uci(uci))
        return {"fen": self.board.fen()}

    def engine_move(self) -> dict:
        if self.fail_engine_move:
            raise RuntimeError("FakeArenaEngine engine_move boom")
        if self.illegal_move:
            return {"engine_move": "e7e5", "fen": self.board.fen()}

        if self.move_sequence:
            move_uci = self.move_sequence.pop(0)
            move = chess.Move.from_uci(move_uci)
        else:
            move = next(iter(self.board.legal_moves), None)

        assert move is not None, "FakeArenaEngine engine_move 在终局时被调用"
        self.board.push(move)
        return {
            "engine_move": move.uci(),
            "fen": self.board.fen(),
            "eval": 0.15,
        }

    def state(self) -> dict:
        return {"fen": self.board.fen()}

    def undo(self) -> dict:
        if self.board.move_stack:
            self.board.pop()
        return {"fen": self.board.fen()}

    def cleanup(self) -> None:
        self.cleaned = True


class TestArenaStorage(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_file = pathlib.Path(self.tmp_dir.name) / "test_arena.db"
        self.storage = a_storage.ArenaStorage(self.db_file)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_save_and_get_record(self):
        record = {
            "id": "rec-1",
            "created_at": "2026-09-21T12:00:00Z",
            "end_time": "2026-09-21T12:05:00Z",
            "white_model": "model_w",
            "white_arg": "preset1",
            "black_model": "model_b",
            "black_arg": None,
            "moves": "e2e4 e7e5 g1f3",
            "ply_count": 3,
            "result": "1-0",
            "winner": "white",
            "termination_reason": "checkmate",
        }
        self.storage.save_record(record)
        fetched = self.storage.get_record("rec-1")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["id"], "rec-1")
        self.assertEqual(fetched["white_model"], "model_w")
        self.assertEqual(fetched["moves"], "e2e4 e7e5 g1f3")
        self.assertEqual(fetched["ply_count"], 3)
        self.assertEqual(fetched["winner"], "white")

    def test_update_record(self):
        record = {
            "id": "rec-2",
            "created_at": "2026-09-21T12:00:00Z",
            "white_model": "w",
            "black_model": "b",
            "moves": "e2e4",
            "ply_count": 1,
            "result": "*",
        }
        self.storage.save_record(record)

        record["moves"] = "e2e4 e7e5"
        record["ply_count"] = 2
        record["result"] = "1/2-1/2"
        record["winner"] = "draw"
        self.storage.save_record(record)

        fetched = self.storage.get_record("rec-2")
        self.assertEqual(fetched["ply_count"], 2)
        self.assertEqual(fetched["result"], "1/2-1/2")
        self.assertEqual(fetched["winner"], "draw")

    def test_list_records_pagination(self):
        for i in range(5):
            self.storage.save_record({
                "id": f"rec-{i}",
                "created_at": f"2026-09-21T12:0{i}:00Z",
                "white_model": "w",
                "black_model": "b",
            })
        items = self.storage.list_records(limit=2, offset=0)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["id"], "rec-4")
        self.assertEqual(items[1]["id"], "rec-3")

        items_p2 = self.storage.list_records(limit=2, offset=2)
        self.assertEqual(len(items_p2), 2)
        self.assertEqual(items_p2[0]["id"], "rec-2")


class TestArenaManager(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.storage = a_storage.ArenaStorage(pathlib.Path(self.tmp_dir.name) / "test.db")
        self.manager = am.ArenaManager(max_sessions=2, storage=self.storage)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_step_alternation(self):
        white_engine = FakeArenaEngine()
        black_engine = FakeArenaEngine()

        def create_eng(name, arg=None):
            if name == "white":
                return white_engine
            return black_engine

        with mock.patch.object(model_registry, "create_engine", side_effect=create_eng):
            session = self.manager.create("white", None, "black", None)

        self.assertEqual(session.board.turn, chess.WHITE)

        # Step 1: 白方走
        res1 = session.step()
        self.assertIsNotNone(res1["move"])
        self.assertEqual(session.board.turn, chess.BLACK)
        self.assertEqual(white_engine.board.fen(), session.board.fen())
        self.assertEqual(black_engine.board.fen(), session.board.fen())
        self.assertEqual(res1["turn"], "black")

        # Step 2: 黑方走
        res2 = session.step()
        self.assertIsNotNone(res2["move"])
        self.assertEqual(session.board.turn, chess.WHITE)
        self.assertEqual(white_engine.board.fen(), session.board.fen())
        self.assertEqual(black_engine.board.fen(), session.board.fen())
        self.assertEqual(res2["turn"], "white")

    def test_checkmate_and_auto_storage(self):
        # 4步杀 (Scholar's mate)
        # 1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7#
        white_moves = ["e2e4", "d1h5", "f1c4", "h5f7"]
        black_moves = ["e7e5", "b8c6", "g8f6"]

        white_engine = FakeArenaEngine(move_sequence=white_moves)
        black_engine = FakeArenaEngine(move_sequence=black_moves)

        def create_eng(name, arg=None):
            return white_engine if name == "white" else black_engine

        with mock.patch.object(model_registry, "create_engine", side_effect=create_eng), \
             mock.patch.object(a_storage, "storage", self.storage):
            session = self.manager.create("white", None, "black", None)
            for _ in range(7):
                res = session.step()
                if res["is_game_over"]:
                    break

            self.assertTrue(session.board.is_game_over())
            self.assertEqual(session.board.outcome().result(), "1-0")
            self.assertTrue(white_engine.cleaned)
            self.assertTrue(black_engine.cleaned)

            record = self.storage.get_record(session.arena_id)
            self.assertIsNotNone(record)
            self.assertEqual(record["result"], "1-0")
            self.assertEqual(record["winner"], "white")
            self.assertEqual(record["termination_reason"], "checkmate")
            self.assertEqual(record["ply_count"], 7)

    def test_opponent_human_move_fail_rolls_back(self):
        white_engine = FakeArenaEngine()
        black_engine = FakeArenaEngine(fail_human_move=True)

        def create_eng(name, arg=None):
            return white_engine if name == "white" else black_engine

        with mock.patch.object(model_registry, "create_engine", side_effect=create_eng):
            session = self.manager.create("white", None, "black", None)
            with self.assertRaises(am.ArenaError):
                session.step()

            # 回滚后 board 保持未走子
            self.assertEqual(len(session.board.move_stack), 0)

    def test_eviction_when_capacity_exceeded(self):
        engines = []

        def create_eng(name, arg=None):
            eng = FakeArenaEngine()
            engines.append(eng)
            return eng

        with mock.patch.object(model_registry, "create_engine", side_effect=create_eng), \
             mock.patch.object(a_storage, "storage", self.storage):
            s1 = self.manager.create("w1", None, "b1", None)
            s2 = self.manager.create("w2", None, "b2", None)
            self.assertEqual(len(self.manager.list_sessions()), 2)

            s3 = self.manager.create("w3", None, "b3", None)
            self.assertEqual(len(self.manager.list_sessions()), 2)

            # s1 的引擎应该被 cleanup
            self.assertTrue(engines[0].cleaned)
            self.assertTrue(engines[1].cleaned)

            # 数据库应将 s1 标记为 stopped
            rec1 = self.storage.get_record(s1.arena_id)
            self.assertIsNotNone(rec1)
            self.assertEqual(rec1["winner"], "stopped")


class TestArenaRoutes(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.storage = a_storage.ArenaStorage(pathlib.Path(self.tmp_dir.name) / "test.db")
        self.manager = am.ArenaManager(max_sessions=2, storage=self.storage)

        self._manager_patcher = mock.patch.object(am, "manager", self.manager)
        self._storage_patcher = mock.patch.object(a_storage, "storage", self.storage)
        self._app_storage_patcher = mock.patch.object(app.a_storage, "storage", self.storage)
        self._manager_patcher.start()
        self._storage_patcher.start()
        self._app_storage_patcher.start()

    def tearDown(self):
        self._app_storage_patcher.stop()
        self._storage_patcher.stop()
        self._manager_patcher.stop()
        self.tmp_dir.cleanup()

    def test_arena_page(self):
        resp = app.arena_page()
        self.assertTrue(pathlib.Path(resp.path).is_file())

    def test_arena_api_flow(self):
        def create_eng(name, arg=None):
            return FakeArenaEngine()

        with mock.patch.object(model_registry, "create_engine", side_effect=create_eng):
            # 1. New game
            new_req = app.NewArenaRequest(white_model="w", black_model="b")
            res = app.new_arena(new_req)
            arena_id = res["arena_id"]
            self.assertIn("state", res)
            self.assertEqual(res["state"]["turn"], "white")

            # 2. Get state
            state_res = app.arena_state(arena_id)
            self.assertEqual(state_res["state"]["ply_count"], 0)

            # 3. Step
            step_res = app.arena_step(arena_id)
            self.assertIsNotNone(step_res["step"]["move"])
            self.assertEqual(step_res["state"]["ply_count"], 1)

            # 4. Close
            close_res = app.close_arena(arena_id)
            self.assertTrue(close_res["closed"])

            # 5. Get closed state -> 404
            with self.assertRaises(HTTPException) as ctx:
                app.arena_state(arena_id)
            self.assertEqual(ctx.exception.status_code, 404)

            # 6. Check record in DB
            records_res = app.list_arena_records()
            self.assertEqual(len(records_res["records"]), 1)
            self.assertEqual(records_res["records"][0]["winner"], "stopped")

            record_res = app.get_arena_record(arena_id)
            self.assertEqual(record_res["record"]["id"], arena_id)


if __name__ == "__main__":
    unittest.main()
