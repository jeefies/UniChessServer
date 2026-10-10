"""Stockfish 远端分析服务（/sf/v1/*）单元与集成测试。

覆盖：
1. 局面历史重建：起始 FEN + 走法历史、合法性校验、三次重复局面识别；
2. 评价指标转换：统一行棋方视角、厘兵 cp / 将杀 mate、WDL 千分比总和；
3. 两阶段同深度对拍：实战着法在候选内 vs 不在候选内（searchmoves 补搜）；
4. 深度上限 128 与耗时预算控制；
5. LRU 计算缓存命中；
6. 鉴权门禁：Authorization: Bearer <token> 与 X-Access-Token 双兼容；
7. API 端点完整性（/sf/v1/* 与 /api/sf/v1/*）。
"""
from __future__ import annotations

import os
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import chess  # noqa: E402
from fastapi import HTTPException  # noqa: E402

try:
    from starlette.testclient import TestClient

    _HAS_TESTCLIENT = True
except Exception:
    _HAS_TESTCLIENT = False

import app  # noqa: E402
from models.SF.engine import default_binary  # noqa: E402
import sf_analyzer  # noqa: E402


class PositionReconstructionTestCase(unittest.TestCase):
    """局面重构与历史一致性测试。"""

    def test_reconstruct_from_startpos(self):
        board = sf_analyzer.reconstruct_board(
            initial_fen=None, moves=["e2e4", "e7e5", "g1f3", "b8c6"]
        )
        self.assertEqual(board.turn, chess.WHITE)
        self.assertEqual(len(board.move_stack), 4)
        self.assertEqual(board.piece_at(chess.C6).symbol(), "n")

    def test_reconstruct_with_custom_fen(self):
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        board = sf_analyzer.reconstruct_board(initial_fen=fen, moves=["f1b5", "a7a6"])
        self.assertEqual(board.turn, chess.WHITE)
        self.assertEqual(board.piece_at(chess.A6).symbol(), "p")

    def test_invalid_fen_raises_value_error(self):
        with self.assertRaises(ValueError):
            sf_analyzer.reconstruct_board(initial_fen="not-a-valid-fen", moves=[])

    def test_illegal_history_move_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            sf_analyzer.reconstruct_board(initial_fen=None, moves=["e2e4", "e2e5"])
        self.assertIn("Illegal move", str(ctx.exception))

    def test_invalid_uci_raises_value_error(self):
        with self.assertRaises(ValueError):
            sf_analyzer.reconstruct_board(initial_fen=None, moves=["bad_uci"])

    def test_threefold_repetition_reconstructed_properly(self):
        # 走棋往返构成三次重复
        moves = [
            "g1f3", "g8f6", "f3g1", "f6g8",
            "g1f3", "g8f6", "f3g1", "f6g8",
        ]
        board = sf_analyzer.reconstruct_board(initial_fen=None, moves=moves)
        self.assertTrue(board.can_claim_threefold_repetition())


class StockfishAnalysisEngineTestCase(unittest.TestCase):
    """Stockfish 分析引擎两阶段搜索与对拍测试（依赖本地/远端 SF 19 二进制）。"""

    @classmethod
    def setUpClass(cls):
        binary = default_binary()
        if not pathlib.Path(binary).is_file():
            raise unittest.SkipTest(f"Stockfish 二进制不存在: {binary}")
        cls.analyzer = sf_analyzer.StockfishAnalyzer(binary=str(binary), threads=1, hash_mb=32)

    @classmethod
    def tearDownClass(cls):
        cls.analyzer.close()

    def test_health_info(self):
        info = self.analyzer.get_health_info()
        self.assertEqual(info["status"], "ok")
        self.assertIn("Stockfish", info["engine"]["name"])
        self.assertEqual(info["defaults"]["standardDepth"], 128)
        self.assertIn("deep", info["profiles"])
        self.assertIn("lightning", info["profiles"])
        self.assertEqual(info["profiles"]["lightning"]["multiPv"], 1)
        self.assertIn("syzygy", info["engine"])
        self.assertTrue(info["engine"]["parallelTwoStage"])

    def setUp(self):
        self.analyzer.clear_cache()

    def test_analyze_move_candidate_hit(self):
        # 常见开局：1. e4 e5 2. Nf3 Nc6 3. Bc4 (f1c4 是主流候选之一)
        res = self.analyzer.analyze_move(
            initial_fen=None,
            moves=["e2e4", "e7e5", "g1f3", "b8c6"],
            played_move="f1c4",
            profile="fast",
            depth=128,
            max_time_ms=700,
            multi_pv=2,
            request_id="test-req-001",
        )
        self.assertEqual(res["requestId"], "test-req-001")
        self.assertTrue(res["comparison"]["canCompare"])
        common_depth = res["comparison"]["commonDepth"]
        self.assertGreaterEqual(common_depth, 5)

        # 验证 best 与 played
        self.assertIsNotNone(res["best"])
        self.assertIsNotNone(res["played"])
        self.assertEqual(res["played"]["move"], "f1c4")
        self.assertEqual(res["played"]["depth"], common_depth)
        self.assertEqual(res["best"]["depth"], common_depth)

        # 统一视角验证：WDL 总和 1000
        best_wdl = res["best"]["wdl"]
        if best_wdl:
            self.assertEqual(best_wdl["win"] + best_wdl["draw"] + best_wdl["loss"], 1000)

        # 验证 stats
        self.assertFalse(res["stats"]["cached"])
        self.assertGreater(res["stats"]["elapsedMs"], 0)

    def test_analyze_move_cache_hit(self):
        # 第一次请求：未缓存
        res1 = self.analyzer.analyze_move(
            initial_fen=None,
            moves=["e2e4", "e7e5", "g1f3", "b8c6"],
            played_move="f1c4",
            profile="fast",
            depth=128,
            max_time_ms=700,
            multi_pv=2,
            request_id="test-req-001",
        )
        self.assertFalse(res1["stats"]["cached"])

        # 相同参数再次请求应命中缓存
        res2 = self.analyzer.analyze_move(
            initial_fen=None,
            moves=["e2e4", "e7e5", "g1f3", "b8c6"],
            played_move="f1c4",
            profile="fast",
            depth=128,
            max_time_ms=700,
            multi_pv=2,
            request_id="test-req-002",
        )
        self.assertTrue(res2["stats"]["cached"])
        self.assertEqual(res2["requestId"], "test-req-002")

    def test_analyze_move_non_candidate_triggers_step2(self):
        # a2a3 不在开局 top 2 候选中，会触发阶段二 searchmoves 补搜
        res = self.analyzer.analyze_move(
            initial_fen=None,
            moves=["e2e4", "e7e5", "g1f3", "b8c6"],
            played_move="a2a3",
            profile="fast",
            depth=128,
            max_time_ms=900,
            multi_pv=2,
            request_id="test-req-003",
        )
        self.assertTrue(res["comparison"]["canCompare"])
        self.assertEqual(res["played"]["move"], "a2a3")
        common_depth = res["comparison"]["commonDepth"]
        self.assertGreaterEqual(common_depth, 5)
        self.assertEqual(res["best"]["depth"], common_depth)
        self.assertEqual(res["played"]["depth"], common_depth)

        # a2a3 分数应劣于最佳招法，diffCp 应当为负
        diff_cp = res["comparison"]["diffCp"]
        if diff_cp is not None:
            self.assertLessEqual(diff_cp, 0)

    def test_evaluate_position(self):
        res = self.analyzer.evaluate(
            initial_fen=None,
            moves=["e2e4", "e7e5"],
            profile="fast",
            depth=128,
            max_time_ms=600,
            multi_pv=2,
            request_id="eval-001",
        )
        self.assertEqual(res["requestId"], "eval-001")
        self.assertGreaterEqual(res["completedDepth"], 5)
        self.assertEqual(len(res["candidates"]), 2)
        self.assertEqual(res["best"]["move"], res["candidates"][0]["move"])

    def test_illegal_played_move_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            self.analyzer.analyze_move(
                initial_fen=None,
                moves=["e2e4", "e7e5"],
                played_move="e1e8",  # 王不可能直接到 e8
            )
        self.assertIn("Illegal playedMove", str(ctx.exception))

    def test_analyze_move_lightning_profile(self):
        # 极速档：目标深度 22，搜满 22 层即停，不限时间预算，双路并发
        res = self.analyzer.analyze_move(
            initial_fen=None,
            moves=["e2e4", "e7e5", "g1f3", "b8c6"],
            played_move="a2a3",
            profile="lightning",
            request_id="test-lightning-001",
        )
        self.assertTrue(res["comparison"]["canCompare"])
        self.assertEqual(res["played"]["move"], "a2a3")
        self.assertEqual(res["comparison"]["commonDepth"], 22)
        self.assertTrue(res["engine"]["parallelTwoStage"])

    def test_evaluate_lightning_profile(self):
        # 极速档：evaluate 同样在搜满 22 层后即刻退出
        res = self.analyzer.evaluate(
            initial_fen=None,
            moves=["e2e4", "e7e5"],
            profile="lightning",
            request_id="test-lightning-eval-001",
        )
        self.assertEqual(res["completedDepth"], 22)
        self.assertEqual(len(res["candidates"]), 1)

    def test_opening_classification(self):
        # 意大利定式走法：f1c4
        res1 = self.analyzer.analyze_move(
            initial_fen=None,
            moves=["e2e4", "e7e5", "g1f3", "b8c6"],
            played_move="f1c4",
            profile="fast",
            depth=6,
            max_time_ms=500,
        )
        self.assertIsNotNone(res1.get("opening"))
        self.assertTrue(res1["opening"]["theory"])
        self.assertTrue(any(k in res1["opening"]["name"] for k in ("意大利", "双马防御")))

        # 偏离定式走法：a2a3
        res2 = self.analyzer.analyze_move(
            initial_fen=None,
            moves=["e2e4", "e7e5", "g1f3", "b8c6"],
            played_move="a2a3",
            profile="fast",
            depth=6,
            max_time_ms=500,
        )
        self.assertIsNotNone(res2.get("opening"))
        self.assertFalse(res2["opening"]["theory"])

    def test_syzygy_endgame_solve(self):
        # 简单车残局 (KR vs K)，应迅速得出杀法或残局胜势
        fen = "8/8/8/8/8/5k2/8/R3K3 w - - 0 1"
        res = self.analyzer.analyze_move(
            initial_fen=fen,
            moves=[],
            played_move="a1a3",
            profile="lightning",
            depth=22,
            max_time_ms=600,
        )
        self.assertTrue(res["comparison"]["canCompare"])
        # 若配置了 Syzygy，tbhits 应存在且非负
        self.assertIn("tbhits", res["stats"])
        self.assertGreaterEqual(res["stats"]["tbhits"], 0)
        # 应找到正分/将杀 (白方车王胜单王)
        best_score = res["best"]["score"]
        if best_score.get("mate") is not None:
            self.assertGreater(best_score["mate"], 0)
        elif best_score.get("cp") is not None:
            self.assertGreater(best_score["cp"], 500)

    def test_blunder_cutoff(self):
        # 挂后白送：1. e4 e5 2. Qh5 Nf6 3. Qxf7+ Kxf7，后送掉
        res = self.analyzer.analyze_move(
            initial_fen=None,
            moves=["e2e4", "e7e5", "d1h5", "g8f6"],
            played_move="h5f7",
            profile="lightning",
            depth=16,
            max_time_ms=1000,
        )
        self.assertTrue(res["comparison"]["canCompare"])
        self.assertEqual(res["played"]["move"], "h5f7")
        # 挂后后分数暴跌，diffCp 为严重负分
        diff_cp = res["comparison"]["diffCp"]
        if diff_cp is not None:
            self.assertLessEqual(diff_cp, -500)


class SfApiRoutesIntegrationTestCase(unittest.TestCase):
    """FastAPI 路由与鉴权集成测试。"""

    def setUp(self):
        self.env_patcher = mock.patch.dict("os.environ", {"UNICHESS_ACCESS_TOKEN": "sf-token-123"})
        self.env_patcher.start()
        self.addCleanup(self.env_patcher.stop)

    def test_health_with_bearer_token(self):
        if not _HAS_TESTCLIENT:
            raise unittest.SkipTest("starlette TestClient 不可用")
        client = TestClient(app.app)
        resp = client.get(
            "/sf/v1/health",
            headers={"Authorization": "Bearer sf-token-123", "x-forwarded-for": "1.2.3.4"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["defaults"]["standardDepth"], 128)

    def test_health_with_x_access_token(self):
        if not _HAS_TESTCLIENT:
            raise unittest.SkipTest("starlette TestClient 不可用")
        client = TestClient(app.app)
        resp = client.get(
            "/api/sf/v1/health",
            headers={"X-Access-Token": "sf-token-123", "x-forwarded-for": "1.2.3.4"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    def test_unauthorized_request_rejected(self):
        if not _HAS_TESTCLIENT:
            raise unittest.SkipTest("starlette TestClient 不可用")
        client = TestClient(app.app)
        resp = client.get(
            "/sf/v1/health",
            headers={"x-forwarded-for": "1.2.3.4"},
        )
        self.assertEqual(resp.status_code, 401)

    def test_analyze_move_post(self):
        if not _HAS_TESTCLIENT:
            raise unittest.SkipTest("starlette TestClient 不可用")
        client = TestClient(app.app)
        payload = {
            "requestId": "api-test-01",
            "position": {
                "initialFen": None,
                "moves": ["e2e4", "e7e5", "g1f3", "b8c6"],
            },
            "playedMove": "f1c4",
            "profile": "fast",
            "limits": {"depth": 128, "maxTimeMs": 600},
            "multiPv": 2,
            "maxPvPlies": 10,
        }
        resp = client.post(
            "/sf/v1/analyze-move",
            json=payload,
            headers={"Authorization": "Bearer sf-token-123", "x-forwarded-for": "1.2.3.4"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["requestId"], "api-test-01")
        self.assertTrue(data["comparison"]["canCompare"])
        self.assertIn("best", data)
        self.assertIn("played", data)

    def test_illegal_move_returns_400(self):
        if not _HAS_TESTCLIENT:
            raise unittest.SkipTest("starlette TestClient 不可用")
        client = TestClient(app.app)
        payload = {
            "position": {"moves": ["e2e4"]},
            "playedMove": "e1e8",
        }
        resp = client.post(
            "/sf/v1/analyze-move",
            json=payload,
            headers={"Authorization": "Bearer sf-token-123", "x-forwarded-for": "1.2.3.4"},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Illegal playedMove", resp.json()["detail"])

    def test_openapi_json_endpoint_is_public(self):
        if not _HAS_TESTCLIENT:
            raise unittest.SkipTest("starlette TestClient 不可用")
        client = TestClient(app.app)
        # 无需任何 Token 请求头，模拟代理请求
        resp = client.get(
            "/sf/v1/openapi.json",
            headers={"x-forwarded-for": "1.2.3.4"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("openapi", data)
        self.assertIn("/sf/v1/analyze-move", data["paths"])
        self.assertIn("/sf/v1/evaluate", data["paths"])
        self.assertIn("/sf/v1/health", data["paths"])
        self.assertIn("/sf/v1/release", data["paths"])
        schemas = data["components"]["schemas"]
        self.assertIn("AnalyzeMoveRequest", schemas)
        self.assertIn("AnalyzeMoveResponse", schemas)
        self.assertIn("ComparisonResult", schemas)
        self.assertIn("ScoreDetail", schemas)
        self.assertIn("OpeningInfo", schemas)

    def test_disk_cache_roundtrip(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tf:
            tmp_db = tf.name
        try:
            cache = sf_analyzer.DiskCache(tmp_db)
            key = ("test_action", "startpos", ("e2e4",), "e7e5", 22)
            self.assertIsNone(cache.get(key))
            sample_val = {"best": {"move": "e7e5", "score": {"value": 0}}, "stats": {"nodes": 100}}
            cache.put(key, sample_val, depth=22)
            self.assertEqual(cache.count(), 1)
            retrieved = cache.get(key)
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved["best"]["move"], "e7e5")
            self.assertEqual(retrieved["stats"]["nodes"], 100)
        finally:
            if os.path.isfile(tmp_db):
                os.remove(tmp_db)

    def test_release_engines_endpoint(self):
        if not _HAS_TESTCLIENT:
            raise unittest.SkipTest("starlette TestClient 不可用")
        client = TestClient(app.app)
        resp = client.post(
            "/sf/v1/release",
            headers={"Authorization": "Bearer sf-token-123", "x-forwarded-for": "1.2.3.4"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("released", data)

    def test_health_shows_memory_idle_status(self):
        if not _HAS_TESTCLIENT:
            raise unittest.SkipTest("starlette TestClient 不可用")
        client = TestClient(app.app)
        resp = client.get(
            "/sf/v1/health",
            headers={"Authorization": "Bearer sf-token-123", "x-forwarded-for": "1.2.3.4"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("memory", data)
        self.assertIn("resident", data["memory"])
        self.assertIn("idleTimeoutSeconds", data["memory"])
        self.assertIn("diskCacheRecords", data["memory"])


    def test_classify_move_judgment_and_accuracy(self):
        # 0 cp 或同一招法 -> best, 100%
        self.assertEqual(sf_analyzer.classify_move_judgment(0, "e2e4", "e2e4"), "best")
        self.assertEqual(sf_analyzer.calculate_move_accuracy(0), 100.0)
        self.assertEqual(sf_analyzer.calculate_move_accuracy(None), 100.0)

        # -15 cp -> excellent
        self.assertEqual(sf_analyzer.classify_move_judgment(-15, "g1f3", "e2e4"), "excellent")
        self.assertTrue(sf_analyzer.calculate_move_accuracy(-15) > 90.0)

        # -35 cp -> good
        self.assertEqual(sf_analyzer.classify_move_judgment(-35, "b1c3", "e2e4"), "good")

        # -80 cp -> inaccuracy
        self.assertEqual(sf_analyzer.classify_move_judgment(-80, "d2d3", "e2e4"), "inaccuracy")

        # -180 cp -> mistake
        self.assertEqual(sf_analyzer.classify_move_judgment(-180, "h2h4", "e2e4"), "mistake")

        # -350 cp -> blunder
        self.assertEqual(sf_analyzer.classify_move_judgment(-350, "f2f3", "e2e4"), "blunder")
        self.assertTrue(sf_analyzer.calculate_move_accuracy(-350) < 30.0)

    def test_review_game_endpoint(self):
        if not _HAS_TESTCLIENT:
            raise unittest.SkipTest("starlette TestClient 不可用")
        client = TestClient(app.app)
        payload = {
            "moves": ["e2e4", "e7e5", "g1f3", "b8c6"],
            "profile": "lightning",
            "concurrency": 2,
        }
        resp = client.post(
            "/sf/v1/review",
            json=payload,
            headers={"Authorization": "Bearer sf-token-123", "x-forwarded-for": "1.2.3.4"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["totalPlies"], 4)
        self.assertEqual(data["analyzedPlies"], 4)
        self.assertIn("summary", data)
        self.assertIn("whiteAccuracy", data["summary"])
        self.assertIn("blackAccuracy", data["summary"])
        self.assertIn("whiteJudgments", data["summary"])
        self.assertEqual(len(data["moves"]), 4)
        for m in data["moves"]:
            self.assertIn("judgment", m)
            self.assertIn("accuracy", m)
            self.assertIn("san", m)

    def test_review_game_with_pgn(self):
        if not _HAS_TESTCLIENT:
            raise unittest.SkipTest("starlette TestClient 不可用")
        client = TestClient(app.app)
        pgn_text = "1. e4 c5 2. Nf3 d6 3. d4 cxd4 *"
        payload = {
            "pgn": pgn_text,
            "profile": "lightning",
        }
        resp = client.post(
            "/sf/v1/analyze-game",
            json=payload,
            headers={"Authorization": "Bearer sf-token-123", "x-forwarded-for": "1.2.3.4"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["totalPlies"], 6)
        self.assertEqual(data["analyzedPlies"], 6)
        self.assertEqual(data["moves"][0]["san"], "e4")
        self.assertEqual(data["moves"][1]["san"], "c5")

    def test_openapi_contains_review_endpoints(self):
        if not _HAS_TESTCLIENT:
            raise unittest.SkipTest("starlette TestClient 不可用")
        client = TestClient(app.app)
        resp = client.get(
            "/sf/v1/openapi.json",
            headers={"x-forwarded-for": "1.2.3.4"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("/sf/v1/review", data["paths"])
        self.assertIn("/sf/v1/analyze-game", data["paths"])
        schemas = data["components"]["schemas"]
        self.assertIn("ReviewGameRequest", schemas)
        self.assertIn("ReviewGameResponse", schemas)
        self.assertIn("GameReviewSummary", schemas)
        self.assertIn("GameMoveItem", schemas)


if __name__ == "__main__":
    unittest.main()
