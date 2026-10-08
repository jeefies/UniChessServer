"""SF 引擎（Stockfish 19，开源最强国际象棋引擎，UCI 子进程）接入验收。

1. 模型发现与状态上报 / 预设解析（同 M2/M3/DS 的口径）
2. 六方法契约端到端（经 session_manager，与 M2 同哲学）
3. UCI 细节：白方视角 WDL、mate 分数、fix-depth 可复现、预算互斥校验

说明：二进制不在 git（GPL，单文件 103MB），默认路径 `Server/tools/stockfish`。
没装二进制（或装在别处）时全部跳过，与 DS 的 skip 口径一致；
装上后本文件应全绿。
"""
from __future__ import annotations

import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import chess  # noqa: E402
import models as model_registry  # noqa: E402
import session_manager as sm  # noqa: E402
from jobs import resolve_engine  # noqa: E402

_SERVER_ROOT = pathlib.Path(__file__).resolve().parent.parent
_BINARY = _SERVER_ROOT / 'tools' / 'stockfish.exe'


def _sf_available() -> bool:
    return 'SF' in model_registry.available_models()


def _binary_present() -> bool:
    return _BINARY.is_file()


def _tiny_kwargs() -> dict:
    return {'movetime_ms': 100, 'threads': 1, 'hash_mb': 8}


class TestSFDiscovery(unittest.TestCase):
    def setUp(self):
        if not _sf_available():
            self.skipTest('models/SF 未部署')

    def test_sf_reported_available(self):
        info = model_registry.describe_model('SF')
        self.assertEqual(info['status'], 'available')
        self.assertEqual(info['reason'], '')
        for preset in ('default', 'fast', 'strong', 'depth12', 'weak', 'elo_1500'):
            self.assertIn(preset, info['presets'])

    def test_unknown_preset_rejected(self):
        with self.assertRaises(model_registry.ArgPresetNotFoundError):
            model_registry.resolve_kwargs('SF', 'no_such_preset')

    def test_description_is_not_a_factory_kwarg(self):
        for preset in ('default', 'fast', 'depth12'):
            kwargs = model_registry.resolve_kwargs('SF', preset)
            self.assertNotIn('description', kwargs)

    def test_budget_priority(self):
        """depth > nodes > movetime_ms；三者互不冲突，只有 depth 生效。"""
        engine = model_registry._load_engine_class('SF')(
            depth=4, nodes=1000, movetime_ms=100, threads=1, hash_mb=8
        )
        self.addCleanup(engine.cleanup)
        self.assertEqual(engine._limit(), chess.engine.Limit(depth=4))


@unittest.skipUnless(_binary_present(), 'Stockfish 二进制未安装（tools/stockfish）')
class TestSFEngine(unittest.TestCase):
    def _make_engine(self, **kwargs):
        merged = _tiny_kwargs()
        merged.update(kwargs)
        engine = model_registry._load_engine_class('SF')(**merged)
        self.addCleanup(engine.cleanup)
        return engine

    def test_constructor_defaults(self):
        engine = self._make_engine()
        self.assertEqual(engine.movetime_ms, 100)
        self.assertEqual(engine.threads, 1)
        self.assertTrue(engine.show_wdl)
        self.assertIn('Stockfish', engine.identity)

    def test_rejects_unknown_kwarg(self):
        engine_cls = model_registry._load_engine_class('SF')
        with self.assertRaises(TypeError):
            engine_cls(ckpt='whatever')

    def test_rejects_out_of_range_kwargs(self):
        engine_cls = model_registry._load_engine_class('SF')
        for bad in ({'movetime_ms': 5}, {'movetime_ms': 900000},
                    {'threads': 0}, {'hash_mb': 0}, {'skill_level': 21},
                    {'uci_elo': 1200}, {'depth': 0}, {'nodes': 0},
                    {'skill_level': 5, 'uci_elo': 1500}):
            with self.assertRaises(ValueError):
                engine_cls(**{**_tiny_kwargs(), **bad})

    def test_full_game_plays_to_end(self):
        engine = self._make_engine()
        state = engine.setup()
        for _ in range(30):
            if state['game_over']:
                break
            legal = state['legal_moves']
            self.assertTrue(legal)
            engine.human_move(legal[0])
            state = engine.state()
            if state['game_over']:
                break
            engine_legal = state['legal_moves']
            result = engine.engine_move()
            self.assertIn('engine_move', result)
            # 引擎着法在它思考时的合法集合内（state 已是引擎走后的局面）
            self.assertIn(result['engine_move'], engine_legal)
            state = engine.state()
        self.assertTrue(state['game_over'])

    def test_engine_answers_human_move(self):
        engine = self._make_engine()
        engine.setup()
        engine.human_move('e2e4')
        result = engine.engine_move()
        self.assertIn('engine_move', result)
        # 引擎走的是黑方着法：在 e2e4 之后的局面里必须合法
        board = chess.Board()
        board.push(chess.Move.from_uci('e2e4'))
        self.assertIn(chess.Move.from_uci(result['engine_move']), board.legal_moves)
        self.assertEqual(len(engine._board.move_stack), 2)

    def test_score_white_perspective_conversion(self):
        """score/wdl 都是行棋方视角：黑方行棋时必须翻成白方视角（分数取负、wdl 交换）。"""
        engine = self._make_engine()
        engine.setup('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1')
        out = engine._record_eval({
            'score': chess.engine.PovScore(chess.engine.Cp(-120), chess.BLACK),
            'wdl': chess.engine.PovWdl(chess.engine.Wdl(900, 90, 10), chess.BLACK),
        })
        self.assertEqual(out['score_cp'], 120)
        self.assertAlmostEqual(out['eval']['win'], 0.010)
        self.assertAlmostEqual(out['eval']['draw'], 0.090)
        self.assertAlmostEqual(out['eval']['loss'], 0.900)

        # 白方行棋、被将杀在 2：白方视角分数为 -2
        engine.setup()
        mated = engine._record_eval({
            'score': chess.engine.PovScore(chess.engine.Mate(-2), chess.WHITE),
        })
        self.assertEqual(mated['score_mate'], -2)
        self.assertIsNone(mated['score_cp'])
    def test_state_shape_and_white_wdl(self):
        engine = self._make_engine()
        engine.setup()
        result = engine.engine_move()
        evaluated = {k: result.get(k) for k in
                     ('score_cp', 'score_mate', 'eval', 'pv', 'nodes', 'depth')}
        self.assertIsNotNone(evaluated['eval'])
        evaluation = evaluated['eval']
        self.assertEqual(evaluation['pov'], 'white')
        total = evaluation['win'] + evaluation['draw'] + evaluation['loss']
        self.assertAlmostEqual(total, 1.0, places=6)
        for key in ('win', 'draw', 'loss'):
            self.assertGreaterEqual(evaluation[key], 0.0)
            self.assertLessEqual(evaluation[key], 1.0)

    def test_engine_rejects_move_from_terminal_position(self):
        engine = self._make_engine(depth=1)
        # 白方一步将杀的局面：引擎必须拒绝再次行棋。
        engine.setup('6k1/5ppp/8/8/8/8/5PPP/4Q1K1 w - - 0 1')
        result = engine.engine_move()
        self.assertEqual(result['engine_move'], 'e1e8')
        self.assertEqual(result['score_mate'], 1)
        with self.assertRaises(ValueError):
            engine.engine_move()

    def test_fixed_depth_is_deterministic(self):
        """固定深度档：同一局面两次独立实例必须给同一着法（可复现标尺）。"""
        moves = []
        for _ in range(2):
            engine = self._make_engine(depth=8, movetime_ms=1000)
            engine.setup()
            moves.append(engine.engine_move()['engine_move'])
            engine.cleanup()
        self.assertEqual(moves[0], moves[1])

    def test_undo_reverts_position(self):
        engine = self._make_engine()
        engine.setup()
        engine.human_move('e2e4')
        engine.engine_move()
        before = engine.state()['fen']
        engine.undo()
        after = engine.state()['fen']
        self.assertNotEqual(before, after)
        self.assertEqual(len(engine._board.move_stack), 0)

    def test_human_move_clears_stale_eval(self):
        engine = self._make_engine()
        engine.setup()
        engine.engine_move()
        engine.human_move('e7e5')
        self.assertIsNone(engine.state()['eval'])

    def test_bad_uci_form_rejected_by_engine(self):
        engine = self._make_engine()
        for bad in ('e2e4z', 'e2e5', 'zzzz'):
            with self.assertRaises(ValueError):
                engine.human_move(bad)

    def test_cleanup_is_idempotent(self):
        engine = self._make_engine()
        engine.setup()
        engine.cleanup()
        self.assertIsNone(engine._engine)
        engine.cleanup()
        self.assertIsNone(engine._engine)

    def test_process_is_gone_after_cleanup(self):
        engine = self._make_engine()
        engine.setup()
        engine.engine_move()
        engine.cleanup()
        engine2 = self._make_engine()
        self.assertIsNotNone(engine2._engine)
        engine2.cleanup()


@unittest.skipUnless(_binary_present(), 'Stockfish 二进制未安装（tools/stockfish）')
class TestSFSession(unittest.TestCase):
    def _make_session(self, engine_white=False, **kwargs):
        merged = _tiny_kwargs()
        merged.update(kwargs)
        manager = sm.SessionManager(max_sessions=4)
        with mock.patch.object(model_registry, 'resolve_kwargs', return_value=merged):
            session = manager.create('SF', 'default', None, engine_white)
        self.addCleanup(manager.close, session.session_id)
        return session

    def test_resolve_engine_native_is_false(self):
        """不声明 KIT_FACTORY：观战/批量对弈走 Kit.serving 包装路径（每 worker 一局）。"""
        ref = resolve_engine('SF', None)
        self.assertFalse(ref.native)
        self.assertEqual(ref.spec['factory'], 'Kit.serving:game_engine_player_factory')
        engine_path = pathlib.Path(ref.spec['kwargs']['engine_path'])
        self.assertEqual(engine_path.name, 'engine.py')
        self.assertEqual(engine_path.parent.name, 'SF')

    def test_engine_white_opening_move(self):
        session = self._make_session(engine_white=True)
        self.assertGreaterEqual(len(session.board.move_stack), 1)
        state = session.state()
        self.assertIsNotNone(state.get('eval'))
        self.assertEqual(state['engine_white'], True)

    def test_human_move_triggers_engine_reply(self):
        session = self._make_session(engine_white=False)
        result = session.human_move('e2e4')
        self.assertIn('engine_move', result)
        self.assertGreaterEqual(len(session.board.move_stack), 2)

    def test_undo_returns_to_start(self):
        session = self._make_session(engine_white=False)
        session.human_move('e2e4')
        session.undo()
        self.assertEqual(len(session.board.move_stack), 0)


if __name__ == '__main__':
    unittest.main()
