"""jobs.py（批量对弈 / 观战 → UniChessKit 后台 job）测试。

job 在独立进程里跑，进程内 mock 够不到，所以这里在临时目录搭假模型目录并把
models.MODELS_DIR 指过去：

- nat1 / nat2：声明 KIT_FACTORY，适配模块放在模型目录内（kit registry 要求模块位于 root 之下，
  两者顶层模块名不同，同一 job 进程加载 A、B 时不冲突），走 kit 原生 Player。
- wrap：只有六方法的 GameEngine，由 kit.serving 包装；预设 slow（每步 sleep）/ fail（走几步后抛异常）。

另含从旧 test_arena / test_batch_runner 迁来的存储层与管理员鉴权测试。

运行：python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import chess  # noqa: E402
from fastapi import HTTPException  # noqa: E402

import app  # noqa: E402
import arena_storage as a_storage  # noqa: E402
import jobs  # noqa: E402
import models as model_registry  # noqa: E402
import session_manager as sm  # noqa: E402

NATIVE_ENGINE = '''
class GameEngine:
    KIT_FACTORY = "{mod}:make_player_factory"
    def __init__(self, **kw): pass
    def setup(self, fen=None): return {{}}
    def human_move(self, uci): return {{}}
    def engine_move(self): raise NotImplementedError
    def state(self): return {{}}
    def undo(self): return {{}}
    def cleanup(self): pass
'''

NATIVE_ADAPTER = '''
from unichess_kit.testing.fakes import make_fake_player_factory

def make_player_factory(preset=None):
    return make_fake_player_factory(name="{name}", salt=preset or "", simulations=8)
'''

WRAPPED_ENGINE = '''
import time
import chess

class GameEngine:
    def __init__(self, delay=0.0, fail_after=None):
        self.delay, self.fail_after, self.n = delay, fail_after, 0
        self.board = chess.Board()
    def setup(self, fen=None):
        self.board = chess.Board(fen) if fen else chess.Board()
        return {"fen": self.board.fen()}
    def human_move(self, uci):
        self.board.push_uci(uci)
        return {"fen": self.board.fen()}
    def engine_move(self):
        self.n += 1
        if self.fail_after is not None and self.n > self.fail_after:
            raise RuntimeError("wrap 故意失败")
        time.sleep(self.delay)
        mv = sorted(self.board.legal_moves, key=lambda m: m.uci())[0]
        self.board.push(mv)
        return {"engine_move": mv.uci(), "eval": 0.25}
    def state(self): return {"fen": self.board.fen()}
    def undo(self):
        self.board.pop()
        return {"fen": self.board.fen()}
    def cleanup(self): pass
'''


def _has_nvidia_smi() -> bool:
    return shutil.which("nvidia-smi") is not None


class FakeModelsMixin:
    """临时 models 目录 + 临时 arena 库 + 临时 job / 租约目录。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(self._tmp.name)
        self.models_dir = root / "models"
        for name in ("nat1", "nat2"):
            d = self.models_dir / name
            d.mkdir(parents=True)
            mod = f"{name}_adapter"
            (d / "engine.py").write_text(NATIVE_ENGINE.format(mod=mod), encoding="utf-8")
            (d / f"{mod}.py").write_text(NATIVE_ADAPTER.format(name=name), encoding="utf-8")
            (d / "config.json").write_text(json.dumps({"p": {}}), encoding="utf-8")
        d = self.models_dir / "wrap"
        d.mkdir()
        (d / "engine.py").write_text(WRAPPED_ENGINE, encoding="utf-8")
        (d / "config.json").write_text(json.dumps({
            "fast": {}, "slow": {"delay": 1.5}, "fail": {"fail_after": 3}}), encoding="utf-8")

        patcher = mock.patch.object(model_registry, "MODELS_DIR", self.models_dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        model_registry._engine_class_cache.clear()
        model_registry._config_cache.clear()
        self.addCleanup(model_registry._engine_class_cache.clear)
        self.addCleanup(model_registry._config_cache.clear)

        self.jobs_dir = root / "jobs"
        self.lease_dir = str(root / "leases")
        self.storage = a_storage.ArenaStorage(root / "arena.db")
        self.env_extra = {"UNICHESS_GPU_LEASE_DIR": self.lease_dir}
        self.addCleanup(self._tmp.cleanup)

    def batch_service(self, **kw):
        kw.setdefault("gpu_mib", 0)
        return jobs.BatchService(self.storage, jobs_dir=self.jobs_dir, env_extra=self.env_extra,
                                 lease_dir=self.lease_dir, **kw)

    def arena_service(self, **kw):
        kw.setdefault("gpu_mib", 0)
        svc = jobs.ArenaService(self.storage, jobs_dir=self.jobs_dir, env_extra=self.env_extra,
                                lease_dir=self.lease_dir, **kw)
        self.addCleanup(svc.close_all)
        return svc

    @staticmethod
    def wait_batch(svc, timeout=120.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snap = svc.snapshot()
            if not snap["is_running"]:
                cur = svc._current
                if cur is not None:
                    cur["handle"].wait(30)      # Windows：进程退出前 job.log 仍被占用
                return snap
            time.sleep(0.3)
        raise AssertionError("批次超时未结束")


class TestResolveEngine(FakeModelsMixin, unittest.TestCase):
    def test_native_spec(self):
        ref = jobs.resolve_engine("nat1", "p")
        self.assertTrue(ref.native)
        self.assertEqual(ref.spec["factory"], "nat1_adapter:make_player_factory")
        self.assertEqual(pathlib.Path(ref.spec["root"]), (self.models_dir / "nat1").resolve())
        self.assertEqual(ref.spec["kwargs"], {"preset": "p"})
        self.assertEqual(ref.label, "nat1(p)")

    def test_wrapped_spec_carries_preset_kwargs(self):
        ref = jobs.resolve_engine("wrap", "slow")
        self.assertFalse(ref.native)
        self.assertEqual(ref.spec["factory"], jobs.WRAPPED_FACTORY)
        self.assertEqual(ref.spec["kwargs"]["engine_kwargs"], {"delay": 1.5})
        self.assertTrue(ref.spec["kwargs"]["engine_path"].endswith("engine.py"))

    def test_empty_arg_means_default(self):
        ref = jobs.resolve_engine("wrap", "")
        self.assertIsNone(ref.arg)
        self.assertEqual(ref.label, "wrap")

    def test_errors(self):
        with self.assertRaises(model_registry.ModelNotFoundError):
            jobs.resolve_engine("nope", None)
        with self.assertRaises(model_registry.ModelNotFoundError):
            jobs.resolve_engine("../etc", None)
        with self.assertRaises(model_registry.ArgPresetNotFoundError):
            jobs.resolve_engine("wrap", "nope")

    def test_eval_is_white_pov(self):
        self.assertEqual(jobs._eval_of({"info": {"q": 0.4}}, mover_white=True), 0.4)
        self.assertEqual(jobs._eval_of({"info": {"q": 0.4}}, mover_white=False), -0.4)
        self.assertEqual(jobs._eval_of({"info": {"eval": 0.25, "q": 0.9}}, False), 0.25)
        self.assertIsNone(jobs._eval_of(None, True))


class TestBatchService(FakeModelsMixin, unittest.TestCase):
    def test_native_batch_completes_with_per_model_tally(self):
        svc = self.batch_service()
        snap = svc.start("nat1", "p", "nat2", None, rounds=4, max_plies=30)
        self.assertEqual(snap["batch"]["white_model"], "nat1")
        final = self.wait_batch(svc)
        batch = final["batch"]
        self.assertEqual(batch["status"], "completed", batch)
        self.assertEqual(batch["rounds_completed"], 4)
        self.assertEqual(sum(batch["tally"].values()), 4)
        self.assertEqual(batch["summary"]["games"], 4)
        # 落库：4 局都带 batch_id，id 为 <batch>-g<n>，模型按执色互换
        stored = self.storage.get_batch(batch["id"])
        self.assertEqual(stored["status"], "completed")
        self.assertEqual(stored["tally"], batch["tally"])
        recs = self.storage.list_records(limit=50, batch_id=batch["id"])
        self.assertEqual(len(recs), 4)
        self.assertEqual({r["id"] for r in recs}, {f"{batch['id']}-g{i}" for i in range(4)})
        self.assertEqual({r["white_model"] for r in recs}, {"nat1", "nat2"})
        for r in recs:
            board = chess.Board()
            for uci in r["moves"].split():
                board.push_uci(uci)             # 着法（含开局）可从初始局面完整重放
            self.assertEqual(len(board.move_stack), len(r["moves"].split()))
        # 历史筛选：批次局 / 非批次局
        self.assertEqual(len(self.storage.list_records(batch_id=a_storage.BATCH_ANY)), 4)
        self.assertEqual(self.storage.list_records(batch_id=""), [])

    def test_wrapped_batch_and_double_start(self):
        svc = self.batch_service()
        svc.start("wrap", "slow", "wrap", "fast", rounds=2, max_plies=6)
        with self.assertRaises(jobs.BatchAlreadyRunningError):
            svc.start("wrap", None, "wrap", None, rounds=2)
        snap = svc.snapshot()
        self.assertTrue(snap["is_running"])
        time.sleep(3)
        snap = svc.snapshot()
        if snap["is_running"]:                  # 慢引擎：应能看到对局快照
            self.assertTrue(snap["workers"])
            w = snap["workers"][0]
            self.assertIn(w["white_model"], ("wrap",))
            chess.Board(w["fen"])
        final = self.wait_batch(svc)
        self.assertEqual(final["batch"]["status"], "completed", final["batch"])
        self.assertEqual(final["batch"]["rounds_completed"], 2)
        # 结束后可以再开一批
        svc.start("wrap", None, "wrap", None, rounds=2, max_plies=4)
        last = self.wait_batch(svc)["batch"]
        self.assertEqual(last["status"], "completed")
        # 运维从库里删掉已结束的批次后，快照回落到库里最新的批次，且不被回写
        conn = self.storage._get_connection()
        try:
            with conn:
                conn.execute("DELETE FROM arena_batches WHERE id = ?", (last["id"],))
        finally:
            conn.close()
        snap = svc.snapshot()
        self.assertEqual(snap["batch"]["id"], final["batch"]["id"])
        self.assertIsNone(self.storage.get_batch(last["id"]))

    def test_validation(self):
        svc = self.batch_service()
        for kw in ({"rounds": 3}, {"rounds": 0}, {"rounds": jobs.MAX_ROUNDS + 2},
                   {"rounds": 2, "max_plies": 0}, {"rounds": 2, "max_plies": jobs.MAX_PLIES_CAP + 1}):
            with self.assertRaises(jobs.InvalidBatchConfigError, msg=kw):
                svc.start("wrap", None, "wrap", None, **kw)
        with self.assertRaises(model_registry.ModelNotFoundError):
            svc.start("nope", None, "wrap", None, rounds=2)
        self.assertFalse(svc.is_running)
        self.assertEqual(self.storage.list_batches(), [])

    def test_engine_error_stops_batch(self):
        svc = self.batch_service()
        svc.start("wrap", "fail", "wrap", None, rounds=4, max_plies=40)
        final = self.wait_batch(svc)
        self.assertEqual(final["batch"]["status"], "error")
        self.assertIn("故意失败", final["batch"]["error"] or "")

    @unittest.skipUnless(_has_nvidia_smi(), "需要 nvidia-smi")
    def test_gpu_busy_refuses_start(self):
        svc = self.batch_service(gpu_mib=10_000_000)
        with self.assertRaises(jobs.GpuBusyError):
            svc.start("nat1", None, "nat2", None, rounds=2)
        self.assertFalse(svc.is_running)
        self.assertEqual(self.storage.list_batches()[0]["status"], "gpu_busy")

    @unittest.skipIf(os.name == "nt", "Windows 下 stop 只 terminate 主进程")
    def test_stop(self):
        svc = self.batch_service()
        svc.start("wrap", "slow", "wrap", "slow", rounds=8, max_plies=200)
        snap = svc.stop()
        self.assertFalse(snap["is_running"])
        self.assertEqual(snap["batch"]["status"], "stopped")

    def test_recover_after_restart(self):
        svc = self.batch_service()
        svc.start("nat1", None, "nat2", None, rounds=2, max_plies=20)
        batch_id = self.wait_batch(svc)["batch"]["id"]
        # 模拟服务在收尾前重启：库里仍是 running，但 job 目录已写完
        row = self.storage.get_batch(batch_id)
        row.update(status="running", rounds_completed=0, end_time=None)
        self.storage.save_batch(row)
        # 另一个 running 批次的 job 目录不存在 → interrupted
        ghost = dict(row, id="ghost", created_at="2000-01-01T00:00:00+00:00")
        self.storage.save_batch(ghost)
        svc2 = self.batch_service()
        self.assertEqual(self.storage.get_batch(batch_id)["status"], "completed")
        self.assertEqual(self.storage.get_batch(batch_id)["rounds_completed"], 2)
        self.assertEqual(self.storage.get_batch("ghost")["status"], "interrupted")
        self.assertFalse(svc2.is_running)

    def test_recover_dead_job_is_interrupted(self):
        svc = self.batch_service()
        # 手工造一个进程已死（没有锁持有者）、状态停在 running 的 job 目录
        job_dir = self.jobs_dir / "dead"
        job_dir.mkdir(parents=True)
        (job_dir / "job.json").write_text("{}", encoding="utf-8")
        (job_dir / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
        self.storage.save_batch({"id": "dead", "created_at": "2026-01-01T00:00:00+00:00",
                                 "white_model": "wrap", "black_model": "wrap", "rounds_planned": 2,
                                 "rounds_completed": 0, "status": "running",
                                 "tally": {"a_win": 0, "b_win": 0, "draw": 0}})
        svc2 = self.batch_service()
        self.assertEqual(self.storage.get_batch("dead")["status"], "interrupted")
        self.assertFalse(svc2.is_running)
        del svc


class TestArenaService(FakeModelsMixin, unittest.TestCase):
    def _play_out(self, watch, limit=200):
        steps = []
        for _ in range(limit):
            s = watch.step(wait_s=30)
            steps.append(s)
            if s["is_game_over"]:
                return steps
        self.fail("观战对局未结束")

    def test_game_plays_to_end_and_saves(self):
        svc = self.arena_service(max_plies=12)
        watch = svc.create("nat1", "p", "wrap", None)
        st = watch.state()
        self.assertEqual(st["white_model"], "nat1")
        self.assertEqual(st["ply_count"], 0)
        steps = self._play_out(watch)
        moved = [s for s in steps if s["move"]]
        self.assertEqual(moved[0]["turn"], "white")     # turn = 刚走棋的一方
        self.assertEqual(moved[1]["turn"], "black")
        self.assertEqual(moved[1]["eval"], 0.25)        # 六方法引擎自带 eval 透传
        last = steps[-1]
        self.assertTrue(last["is_game_over"])
        self.assertIsNotNone(last["termination_reason"])
        self.assertEqual(last["ply_count"], len(last["san_history"]))
        rec = self.storage.get_record(watch.arena_id)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["ply_count"], last["ply_count"])
        self.assertEqual(rec["termination_reason"], last["termination_reason"])
        # 终局后继续 step：幂等，不再出新着法
        again = watch.step(wait_s=0)
        self.assertTrue(again["is_game_over"])
        self.assertIsNone(again["move"])

    def test_start_from_fen(self):
        fen = "4k3/8/8/8/8/8/4P3/4K2R b K - 0 1"
        svc = self.arena_service(max_plies=6)
        watch = svc.create("wrap", None, "nat2", None, fen=fen)
        self.assertEqual(watch.state()["fen"], fen)
        s = watch.step(wait_s=30)
        self.assertEqual(s["turn"], "black")
        board = chess.Board(fen)
        board.push_uci(s["move"])
        self.assertEqual(s["fen"], board.fen())

    def test_pending_then_move(self):
        svc = self.arena_service(max_plies=4)
        watch = svc.create("wrap", "slow", "wrap", "slow")
        s = watch.step(wait_s=0.2)
        self.assertTrue(s["pending"])
        self.assertIsNone(s["move"])
        self.assertFalse(s["is_game_over"])
        s = watch.step(wait_s=30)
        self.assertFalse(s["pending"])
        self.assertIsNotNone(s["move"])

    def test_close_mid_game_records_stopped(self):
        svc = self.arena_service()
        watch = svc.create("wrap", "slow", "wrap", "slow")
        job_dir = watch.handle.dir
        svc.close(watch.arena_id)
        rec = self.storage.get_record(watch.arena_id)
        self.assertEqual(rec["winner"], "stopped")
        self.assertEqual(rec["termination_reason"], "stopped")
        self.assertFalse(job_dir.exists())
        with self.assertRaises(jobs.ArenaNotFoundError):
            svc.get(watch.arena_id)

    def test_engine_error_surfaces(self):
        svc = self.arena_service()
        with self.assertRaises(jobs.ArenaError):      # 失败得快时 create 就报，否则在 step 报
            watch = svc.create("wrap", "fail", "wrap", None)
            for _ in range(20):
                watch.step(wait_s=30)

    def test_eviction_and_errors(self):
        svc = self.arena_service(max_sessions=2)
        a = svc.create("wrap", "slow", "wrap", "slow")
        b = svc.create("wrap", "slow", "wrap", "slow")
        c = svc.create("wrap", "slow", "wrap", "slow")
        with self.assertRaises(jobs.ArenaNotFoundError):
            svc.get(a.arena_id)
        svc.get(b.arena_id)
        svc.get(c.arena_id)
        self.assertEqual(self.storage.get_record(a.arena_id)["winner"], "stopped")
        with self.assertRaises(sm.InvalidFenError):
            svc.create("wrap", None, "wrap", None, fen="not a fen")
        with self.assertRaises(jobs.ArenaNotFoundError):
            svc.close("nope")

    def test_finish_unwatched(self):
        svc = self.arena_service(max_plies=6)
        watch = svc.create("nat1", None, "nat2", None)
        watch.handle.wait(60)
        svc.refresh()
        rec = self.storage.get_record(watch.arena_id)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["ply_count"], 6)


class TestRoutes(FakeModelsMixin, unittest.TestCase):
    """路由层错误码与响应形状（服务单例换成临时目录版本）。"""

    def setUp(self):
        super().setUp()
        services = {"batch": self.batch_service(), "arena": self.arena_service(max_plies=4)}
        patcher = mock.patch.dict(jobs._services, services, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(a_storage, "storage", self.storage)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _status(self, fn, *args):
        with self.assertRaises(HTTPException) as ctx:
            fn(*args)
        return ctx.exception.status_code

    def test_arena_routes(self):
        req = app.NewArenaRequest(white_model="wrap", black_model="nat1")
        resp = app.new_arena(req)
        arena_id = resp["arena_id"]
        self.assertEqual(resp["state"]["ply_count"], 0)
        step = app.arena_step(arena_id)
        self.assertEqual(step["step"]["turn"], "white")
        self.assertEqual(step["state"]["ply_count"], 1)
        self.assertEqual(app.arena_state(arena_id)["state"]["ply_count"], 1)
        self.assertTrue(app.close_arena(arena_id)["closed"])
        self.assertEqual(app.get_arena_record(arena_id)["record"]["winner"], "stopped")
        self.assertEqual(self._status(app.arena_step, arena_id), 404)
        self.assertEqual(self._status(app.close_arena, arena_id), 404)
        bad = app.NewArenaRequest
        self.assertEqual(self._status(app.new_arena, bad(white_model="nope", black_model="wrap")), 404)
        self.assertEqual(self._status(app.new_arena, bad(white_model="wrap", white_arg="x",
                                                         black_model="wrap")), 400)
        self.assertEqual(self._status(app.new_arena, bad(white_model="wrap", black_model="wrap",
                                                         fen="bad fen")), 400)

    def test_batch_requires_admin(self):
        from starlette.requests import Request
        remote = Request({"type": "http", "headers": [], "client": ("203.0.113.9", 1)})
        req = app.BatchStartRequest(white_model="wrap", black_model="wrap", rounds=2)
        with mock.patch.dict("os.environ", {"UNICHESS_ADMIN_TOKEN": "s3cret"}):
            self.assertEqual(self._status(app.start_batch, req, remote), 403)
            self.assertEqual(self._status(app.stop_batch, remote), 403)
        self.assertEqual(self.storage.list_batches(), [])
        self.assertFalse(app.batch_state()["is_running"])

    def test_batch_route_error_codes(self):
        from starlette.requests import Request
        local = Request({"type": "http", "headers": [], "client": ("127.0.0.1", 1)})
        bad = app.BatchStartRequest(white_model="wrap", black_model="wrap", rounds=3)
        self.assertEqual(self._status(app.start_batch, bad, local), 400)
        missing = app.BatchStartRequest(white_model="nope", black_model="wrap", rounds=2)
        self.assertEqual(self._status(app.start_batch, missing, local), 404)


class TestSessionClaimDraw(unittest.TestCase):
    """session_manager 终局改用 kit classify：可申请和棋（三次重复）即终局，拒绝继续走子。"""

    class Engine:
        def __init__(self):
            self.board = chess.Board()

        def setup(self, fen=None):
            self.board = chess.Board(fen) if fen else chess.Board()
            return {}

        def human_move(self, uci):
            self.board.push_uci(uci)
            return {}

        def engine_move(self):
            # 骑士来回跳，制造三次重复
            for uci in ("g8f6", "f6g8", "b8c6", "c6b8"):
                mv = chess.Move.from_uci(uci)
                if mv in self.board.legal_moves:
                    self.board.push(mv)
                    return {"engine_move": uci}
            raise AssertionError("no move")

        def state(self):
            return {}

        def undo(self):
            return {}

        def cleanup(self):
            pass

    def test_threefold_ends_game(self):
        with mock.patch.object(model_registry, "create_engine", lambda *a: self.Engine()):
            mgr = sm.SessionManager(max_sessions=2)
            session = mgr.create("x")
        for uci in ("g1f3", "f3g1", "g1f3", "f3g1"):
            session.human_move(uci)
        self.assertTrue(session.state()["is_game_over"])
        self.assertFalse(session.board.is_game_over())   # 旧判定会继续
        with self.assertRaises(sm.IllegalMoveError):
            session.human_move("g1f3")


class TestArenaStorage(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_file = pathlib.Path(self.tmp_dir.name) / "test_arena.db"
        self.storage = a_storage.ArenaStorage(self.db_file)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_save_and_get_record(self):
        self.storage.save_record({
            "id": "rec-1", "created_at": "2026-09-21T12:00:00Z", "end_time": "2026-09-21T12:05:00Z",
            "white_model": "model_w", "white_arg": "preset1", "black_model": "model_b",
            "black_arg": None, "moves": "e2e4 e7e5 g1f3", "ply_count": 3, "result": "1-0",
            "winner": "white", "termination_reason": "checkmate",
        })
        fetched = self.storage.get_record("rec-1")
        self.assertEqual(fetched["white_model"], "model_w")
        self.assertEqual(fetched["moves"], "e2e4 e7e5 g1f3")
        self.assertEqual(fetched["ply_count"], 3)
        self.assertEqual(fetched["winner"], "white")

    def test_update_record(self):
        record = {"id": "rec-2", "created_at": "2026-09-21T12:00:00Z", "white_model": "w",
                  "black_model": "b", "moves": "e2e4", "ply_count": 1, "result": "*"}
        self.storage.save_record(record)
        record.update(moves="e2e4 e7e5", ply_count=2, result="1/2-1/2", winner="draw")
        self.storage.save_record(record)
        fetched = self.storage.get_record("rec-2")
        self.assertEqual(fetched["ply_count"], 2)
        self.assertEqual(fetched["result"], "1/2-1/2")
        self.assertEqual(fetched["winner"], "draw")

    def test_list_records_pagination_and_filter(self):
        for i in range(5):
            self.storage.save_record({"id": f"rec-{i}", "created_at": f"2026-09-21T12:0{i}:00Z",
                                      "white_model": "w", "black_model": "b",
                                      "batch_id": "B1" if i % 2 else None})
        items = self.storage.list_records(limit=2, offset=0)
        self.assertEqual([r["id"] for r in items], ["rec-4", "rec-3"])
        self.assertEqual(self.storage.list_records(limit=2, offset=2)[0]["id"], "rec-2")
        self.assertEqual({r["id"] for r in self.storage.list_records(batch_id="B1")}, {"rec-1", "rec-3"})
        self.assertEqual(len(self.storage.list_records(batch_id=a_storage.BATCH_ANY)), 2)
        self.assertEqual(len(self.storage.list_records(batch_id="")), 3)

    def test_storage_migration_adds_batch_id(self):
        import sqlite3
        conn = sqlite3.connect(str(self.db_file))
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS t (id TEXT PRIMARY KEY)")
            conn.commit()
        finally:
            conn.close()
        storage2 = a_storage.ArenaStorage(self.db_file)
        storage2.save_record({"id": "mig-1", "created_at": "2026-01-01T00:00:00Z",
                              "white_model": "A", "black_model": "B", "moves": "e2e4",
                              "ply_count": 1, "result": "1-0", "batch_id": "batch-1"})
        self.assertEqual(storage2.get_record("mig-1").get("batch_id"), "batch-1")


class TestBatchAdminAuth(unittest.TestCase):
    """/api/arena/batch/start、/stop 的管理员鉴权。"""

    @staticmethod
    def _request(client='127.0.0.1', **headers):
        from starlette.requests import Request
        raw = [(k.replace('_', '-').lower().encode(), v.encode()) for k, v in headers.items()]
        return Request({"type": "http", "headers": raw, "client": (client, 12345)})

    def setUp(self):
        patcher = mock.patch.dict('os.environ', {'UNICHESS_ADMIN_TOKEN': 's3cret'})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _denied(self, req):
        with self.assertRaises(HTTPException) as ctx:
            app.require_admin(req)
        self.assertEqual(ctx.exception.status_code, 403)

    def test_valid_token_allowed_from_anywhere(self):
        app.require_admin(self._request('203.0.113.9', x_admin_token='s3cret',
                                        cf_connecting_ip='203.0.113.9'))

    def test_wrong_or_missing_token_denied_remotely(self):
        self._denied(self._request('203.0.113.9'))
        self._denied(self._request('203.0.113.9', x_admin_token='nope'))

    def test_direct_localhost_allowed(self):
        app.require_admin(self._request('127.0.0.1'))

    def test_tunneled_public_request_denied(self):
        # 隧道把公网请求转成 127.0.0.1，但 Cloudflare/nginx 会带上转发头
        self._denied(self._request('127.0.0.1', x_forwarded_for='198.51.100.7'))
        self._denied(self._request('127.0.0.1', cf_connecting_ip='198.51.100.7'))

    def test_no_token_configured_rejects_token_guessing(self):
        with mock.patch.dict('os.environ', {'UNICHESS_ADMIN_TOKEN': ''}), \
                mock.patch.object(app, 'ADMIN_TOKEN_FILE', pathlib.Path('/nonexistent/admin_token')):
            self._denied(self._request('203.0.113.9', x_admin_token=''))
            self._denied(self._request('203.0.113.9', x_admin_token='anything'))


if __name__ == '__main__':
    unittest.main()
