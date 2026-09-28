"""DS 引擎接入验收测试（Server 侧）。

覆盖：
1. 模型发现与状态上报（DS 部署后可用；Windows 无符号链接时 skip）
2. 预设解析（default / fast）
3. 六方法契约端到端（经 session_manager；fake _chat_impl 避免联网）
4. resolve_engine('DS', None) 断言 native=False（观战/批量走 Kit.serving 包装路径）

说明：DS 未部署（Windows 下 `models/DS` 是 git 符号链接的文本检出，不可发现）即 skip。
"""
from __future__ import annotations

import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import models as model_registry  # noqa: E402
import session_manager as sm  # noqa: E402
import chess  # noqa: E402


def _ds_available() -> bool:
    return 'DS' in model_registry.available_models()


def _tiny_ds_kwargs() -> dict:
    """契约用例用的极小预算 kwargs（经 mock 注入 resolve_kwargs）。"""
    return {
        'timeout_s': 30,
        'max_attempts': 2,
        'temperature': 0.0,
        'max_tokens': 16,
        'history_plies': 0,
        'thinking': False,
    }


class TestDSDiscovery(unittest.TestCase):
    """模型发现层：不加载权重，只验证上报与预设解析。"""

    def setUp(self):
        if not _ds_available():
            self.skipTest('models/DS 未部署（Windows 无符号链接）')

    def test_ds_reported_available(self):
        info = model_registry.describe_model('DS')
        self.assertEqual(info['status'], 'available')
        for preset in ('default', 'fast'):
            self.assertIn(preset, info['presets'])

    def test_default_preset(self):
        kwargs = model_registry.resolve_kwargs('DS', 'default')
        self.assertEqual(kwargs['timeout_s'], 30)
        self.assertEqual(kwargs['max_attempts'], 2)
        self.assertTrue(kwargs['thinking'])

    def test_fast_preset(self):
        kwargs = model_registry.resolve_kwargs('DS', 'fast')
        self.assertEqual(kwargs['timeout_s'], 10)
        self.assertFalse(kwargs['thinking'])

    def test_unknown_preset_rejected(self):
        with self.assertRaises(model_registry.ArgPresetNotFoundError):
            model_registry.resolve_kwargs('DS', 'no_such_preset')

    def test_description_is_not_a_factory_kwarg(self):
        for preset in ('default', 'fast'):
            kwargs = model_registry.resolve_kwargs('DS', preset)
            self.assertNotIn('description', kwargs)


@unittest.skipUnless(_ds_available(), 'models/DS 未部署')
class TestDSContract(unittest.TestCase):
    """六方法契约 + 服务层集成。"""

    def _make_session(self, engine_white=False, fen=None):
        manager = sm.SessionManager(max_sessions=4)
        patcher = mock.patch.object(
            model_registry, 'resolve_kwargs', return_value=_tiny_ds_kwargs()
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        session = manager.create('DS', 'default', fen, engine_white)
        self.addCleanup(manager.close, session.session_id)
        return session

    def test_constructor_defaults_via_resolve_kwargs(self):
        with mock.patch.object(model_registry, 'resolve_kwargs', return_value={}):
            engine = model_registry.create_engine('DS', None)
        try:
            self.assertEqual(engine.timeout_s, 30.0)
            self.assertEqual(engine.max_attempts, 2)
            self.assertEqual(engine.temperature, 0.2)
            self.assertTrue(engine.thinking)
        finally:
            engine.cleanup()

    def test_rejects_unknown_kwarg(self):
        engine_cls = model_registry._load_engine_class('DS')
        with self.assertRaises(TypeError):
            engine_cls(ckpt='whatever')

    def test_full_round_trip_engine_white(self):
        session = self._make_session(engine_white=True)
        self.assertEqual(len(session.board.move_stack), 1)
        state = session.state()
        self.assertEqual(len(state['san_history']), 1)
        human_uci = 'e7e5' if session.board.move_stack[-1].uci() == 'e2e4' else 'e7e6'
        fake = lambda *a, **k: {'content': human_uci, 'reasoning_content': ''}
        session.engine._chat_impl = staticmethod(fake)
        result = session.human_move(human_uci)
        self.assertIn('engine_move', result)
        self.assertEqual(len(session.board.move_stack), 3)

    def test_engine_black_waits_for_human(self):
        session = self._make_session(engine_white=False)
        self.assertEqual(len(session.board.move_stack), 0)
        fake = lambda *a, **k: {'content': 'e7e5', 'reasoning_content': ''}
        session.engine._chat_impl = staticmethod(fake)
        result = session.human_move('e2e4')
        self.assertIn('engine_move', result)
        self.assertEqual(len(session.board.move_stack), 2)

    def test_illegal_human_move_rejected_and_rolled_back(self):
        session = self._make_session(engine_white=False)
        before = session.board.fen()
        with self.assertRaises(sm.IllegalMoveError):
            session.human_move('e2e5')
        self.assertEqual(session.board.fen(), before)

    def test_bad_uci_form_rejected_by_engine(self):
        engine = model_registry._load_engine_class('DS')(**_tiny_ds_kwargs())
        self.addCleanup(engine.cleanup)
        for bad in ('e2e4z', 'e2e5', 'zzzz'):
            with self.assertRaises(ValueError):
                engine.human_move(bad)

    def test_undo_rolls_back_both_sides(self):
        session = self._make_session(engine_white=True)
        fake = lambda *a, **k: {'content': 'e7e5', 'reasoning_content': ''}
        session.engine._chat_impl = staticmethod(fake)
        session.human_move('e7e5')
        self.assertEqual(len(session.board.move_stack), 3)
        session.undo()
        self.assertEqual(len(session.board.move_stack), 1)

    def test_engine_rejects_move_from_terminal_position(self):
        engine = model_registry._load_engine_class('DS')(**_tiny_ds_kwargs())
        self.addCleanup(engine.cleanup)
        engine.setup('6k1/5ppp/8/8/8/8/5PPP/4Q1K1 w - - 0 1')
        while not engine.state()['game_over']:
            legal = engine.state()['legal_moves']
            if not legal:
                break
            engine.human_move(legal[0])
        with self.assertRaises(ValueError):
            engine.engine_move()

    def test_state_shape_no_fake_wdl(self):
        session = self._make_session(engine_white=False)
        fake = lambda *a, **k: {'content': 'e7e5', 'reasoning_content': 't' * 7}
        session.engine._chat_impl = staticmethod(fake)
        session.human_move('e2e4')
        result = session.human_move('e7e5')
        state = session.state()
        for key in ('fen', 'legal_moves', 'history', 'san_history',
                    'last_move', 'last_move_san', 'last_move_actor',
                    'engine_ms', 'in_check', 'game_over', 'eval', 'llm'):
            self.assertIn(key, state)
        self.assertIsNone(state['eval'])
        self.assertIsNotNone(state['llm'])
        self.assertEqual(state['llm']['engine_move'], 'e2e4')
        self.assertEqual(result['engine_move'], 'e2e4')

    def test_resolve_engine_native_is_false(self):
        ref = model_registry.resolve_engine('DS', None)
        self.assertFalse(ref.native)


if __name__ == '__main__':
    unittest.main()
