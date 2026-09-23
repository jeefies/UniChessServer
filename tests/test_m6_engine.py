"""M6 预览引擎接入验收测试。

覆盖：
1. 模型发现与状态上报（models/M6 符号链接可用，预设可解析）
2. 默认配置即 M6 代码默认配置中最强一档（64 模拟 / 批量 8 / 树复用）
3. 六方法契约端到端（真实权重 + 极小模拟预算）
4. 共享模型缓存：同 (ckpt, device) 只加载一次，会话间共享权重但搜索树独立
5. 服务层集成：非法走法回滚、悔棋双方同步、引擎执黑等待人类

说明：引擎仅支持 GPU 推理（无 CPU 配置），CUDA 不可用时相关用例 skip。
契约用例用 `simulations=2` 的极小预算，单步约 0.1s，整套几秒内跑完。
"""
from __future__ import annotations

import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import models as model_registry  # noqa: E402
import session_manager as sm  # noqa: E402

M6_CKPT = 'models/m6-preview-inference.pt'


def _m6_available() -> bool:
    return 'M6' in model_registry.available_models()


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _tiny_kwargs() -> dict:
    """契约用例用的极小预算 kwargs（经 mock 注入 resolve_kwargs）。"""
    return {
        'ckpt': M6_CKPT,
        'device': 'cuda',
        'simulations': 2,
        'eval_batch_size': 1,
    }


class TestM6Discovery(unittest.TestCase):
    """模型发现层：不加载权重，只验证上报与预设解析。"""

    def setUp(self):
        if not _m6_available():
            self.skipTest('models/M6 符号链接未配置')

    def test_m6_reported_available(self):
        info = model_registry.describe_model('M6')
        self.assertEqual(info['status'], 'available')
        self.assertIn('default', info['presets'])
        self.assertIn('preview', info['presets'])

    def test_default_preset_is_strongest_code_default(self):
        """默认预设必须是 M6 代码默认配置中最强的一档。"""
        kwargs = model_registry.resolve_kwargs('M6', 'default')
        # simulations=64：chess_ai/search/mcts.py 的 MCTSConfig 默认值，
        # 也是 M6 代码里最大的默认模拟预算（play_preview 的 16 是 CPU 演示弱化档）
        self.assertEqual(kwargs['simulations'], 64)
        # 批量评估与树复用取 play_preview.py 的生产默认
        self.assertEqual(kwargs['eval_batch_size'], 8)
        self.assertTrue(kwargs['reuse_tree'])
        self.assertEqual(kwargs['c_puct'], 1.5)

    def test_preview_preset_matches_shipped_defaults(self):
        kwargs = model_registry.resolve_kwargs('M6', 'preview')
        self.assertEqual(kwargs['simulations'], 16)
        self.assertEqual(kwargs['eval_batch_size'], 8)
        self.assertTrue(kwargs['reuse_tree'])

    def test_unknown_preset_rejected(self):
        with self.assertRaises(model_registry.ArgPresetNotFoundError):
            model_registry.resolve_kwargs('M6', 'no_such_preset')

    @unittest.skipUnless(_cuda_available(), 'CUDA 不可用')
    def test_constructor_defaults_are_strongest_code_default(self):
        """不传 arg_name 时，构造参数本身就是最强默认档。"""
        with mock.patch.object(
            model_registry, 'resolve_kwargs', return_value={}
        ):
            engine = model_registry.create_engine('M6', None)
        try:
            self.assertEqual(engine.simulations, 64)
            self.assertEqual(engine.eval_batch_size, 8)
            self.assertTrue(engine.reuse_tree)
            self.assertEqual(engine.c_puct, 1.5)
            self.assertEqual(engine.device, 'cuda')
        finally:
            engine.cleanup()


@unittest.skipUnless(
    _m6_available() and _cuda_available(),
    'models/M6 符号链接未配置或 CUDA 不可用',
)
class TestM6Contract(unittest.TestCase):
    """六方法契约 + 服务层集成：真实权重 + 极小预算。"""

    def _make_session(self, engine_white=False, fen=None):
        """经 mock 注入极小预算，走真实的 create/setup 链路。"""
        manager = sm.SessionManager(max_sessions=4)
        patcher = mock.patch.object(
            model_registry, 'resolve_kwargs', return_value=_tiny_kwargs()
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        session = manager.create('M6', 'default', fen, engine_white)
        self.addCleanup(manager.close, session.session_id)
        return session

    def test_full_round_trip(self):
        """引擎执白：开局自动走子 → 人类走法同步 → 引擎应答 → 状态自洽。"""
        session = self._make_session(engine_white=True)
        # setup 时引擎执白，服务层已自动触发开局第一步
        self.assertEqual(len(session.board.move_stack), 1)
        state = session.state()
        self.assertEqual(len(state['san_history']), 1)
        self.assertIsNotNone(state.get('last_move'))
        self.assertIsNotNone(state.get('engine_ms'))
        self.assertIsInstance(state.get('nodes'), int)
        # eval：白方视角 WDL，概率和约为 1
        evaluation = state.get('eval')
        self.assertIsNotNone(evaluation)
        self.assertEqual(evaluation['pov'], 'white')
        total = evaluation['win'] + evaluation['draw'] + evaluation['loss']
        self.assertAlmostEqual(total, 1.0, places=4)

        # 人类走一步，引擎必须应答一步
        opening = session.board.move_stack[-1].uci()
        human_uci = 'e7e5' if opening == 'e2e4' else 'e7e6'
        result = session.human_move(human_uci)
        self.assertIn('engine_move', result)
        self.assertEqual(len(session.board.move_stack), 3)
        # human_move 返回的是引擎应答的 engine_move 载荷；着法记录看 state()
        state = session.state()
        self.assertEqual(len(state['san_history']), 3)
        # 服务层权威 Board 与引擎 Board 必须一致
        self.assertEqual(session.board.fen(), session.engine.board.fen())

    def test_illegal_human_move_rejected_and_rolled_back(self):
        session = self._make_session(engine_white=True)
        fen_before = session.board.fen()
        with self.assertRaises(sm.IllegalMoveError):
            session.human_move('e2e4')  # 引擎执白已占 e4，不合法
        self.assertEqual(session.board.fen(), fen_before)
        self.assertEqual(session.engine.board.fen(), fen_before)

    def test_engine_black_waits_for_human(self):
        session = self._make_session(engine_white=False)
        self.assertEqual(len(session.board.move_stack), 0)
        result = session.human_move('d2d4')
        self.assertIn('engine_move', result)
        self.assertEqual(len(session.board.move_stack), 2)
        self.assertEqual(session.board.fen(), session.engine.board.fen())

    def test_undo_rolls_back_both_sides(self):
        session = self._make_session(engine_white=True)
        session.human_move('e7e5')
        self.assertEqual(len(session.board.move_stack), 3)
        session.undo()
        # 退掉引擎应答与人类走子，剩引擎开局第一步，且轮到人类
        self.assertEqual(len(session.board.move_stack), 1)
        self.assertEqual(len(session.engine.board.move_stack), 1)
        self.assertEqual(session.board.fen(), session.engine.board.fen())

    def test_shared_model_and_isolated_search(self):
        """同参数多个会话共享一份权重，但各自持有独立搜索树。"""
        s1 = self._make_session(engine_white=False)
        s2 = self._make_session(engine_white=False)
        self.assertIs(s1.engine.model, s2.engine.model)
        self.assertIsNot(s1.engine.mcts, s2.engine.mcts)

    def test_cleanup_releases_search_but_keeps_shared_model(self):
        session = self._make_session(engine_white=False)
        model = session.engine.model
        session.engine.cleanup()
        # cleanup 只释放会话级搜索树，共享权重必须保留给其它会话
        self.assertIsNone(session.engine.mcts)
        self.assertIs(session.engine.model, model)
        session.engine.cleanup()  # 幂等，第二次不得抛异常


if __name__ == '__main__':
    unittest.main()
