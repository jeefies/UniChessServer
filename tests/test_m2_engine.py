"""M2 引擎接入验收测试。

覆盖：
1. 模型发现与状态上报（models/M2 适配层 + 包在 import 根下）
2. 包完整性：两个权重的 SHA-256 与上游 README 声明一致（部署后自检）
3. 构造参数校验与默认档
4. 六方法契约端到端（CPU 即可，极小时间预算）
5. 共享模型 / 每会话独立评估器与搜索树（FastEvaluator 复用输入缓冲区，
   并发共用会互相踩，这是上游文档明说的约束）
6. 与 M3 同进程共存：M3 用 `src/chess_ai` 命名空间包，M2 是一堆裸顶层模块，
   两者都不要被对方顶替；且 M2 压 torch 线程数不能把 M3 弄坏
7. state() 的形态契约：eval 恒为 None（M2 只有 tanh 标量，不伪造 WDL），
   原生分在 value_tanh

说明：M2 是 CPU 引擎，没有 CUDA 也能跑；但**必须有 torch**和部署好的包
（import 根下 `M2/`），否则相关用例 skip。
"""
from __future__ import annotations

import hashlib
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import models as model_registry  # noqa: E402
import session_manager as sm  # noqa: E402

_SERVER_ROOT = pathlib.Path(__file__).resolve().parent.parent
M2_PACKAGE = _SERVER_ROOT.parent / 'M2'
POLICY_WEIGHTS = M2_PACKAGE / 'chess_model_balanced.pt'
POLICY_SHA256 = '60ab3e89ebf853380f67777825645c12c3716221f58f3b311a713ffa6d60ce00'
VALUE_SHA256 = '4e44421b422c8467a32c06c85a31a5591b8397ae24c2f54fdd39c778fe7dd4a0'


def _m2_available() -> bool:
    return 'M2' in model_registry.available_models()


def _package_ready() -> bool:
    if not POLICY_WEIGHTS.is_file():
        return False
    runs = sorted(M2_PACKAGE.glob('value_full_runs/*/value_full_epoch2.pt'))
    return bool(runs)


def _m3_available() -> bool:
    return 'M3' in model_registry.available_models()


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _value_weights() -> pathlib.Path:
    return sorted(M2_PACKAGE.glob('value_full_runs/*/value_full_epoch2.pt'))[-1]


def _tiny_kwargs() -> dict:
    """契约用例用的极小预算 kwargs（经 mock 注入 resolve_kwargs）。"""
    return {'seconds': 0.2, 'depth': 2, 'qdepth': 4, 'claim_draw': True}


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class TestM2Discovery(unittest.TestCase):
    """模型发现层：不加载权重，只验证上报与预设解析。"""

    def setUp(self):
        if not _m2_available():
            self.skipTest('models/M2 适配层未部署')

    def test_m2_reported_available(self):
        info = model_registry.describe_model('M2')
        self.assertEqual(info['status'], 'available')
        for preset in ('default', 'fast', 'deep'):
            self.assertIn(preset, info['presets'])

    def test_default_preset(self):
        kwargs = model_registry.resolve_kwargs('M2', 'default')
        self.assertEqual(kwargs['seconds'], 2.0)
        self.assertEqual(kwargs['depth'], 5)
        self.assertEqual(kwargs['qdepth'], 6)
        self.assertTrue(kwargs['claim_draw'])

    def test_fast_preset(self):
        kwargs = model_registry.resolve_kwargs('M2', 'fast')
        self.assertEqual(kwargs['seconds'], 0.5)
        self.assertEqual(kwargs['depth'], 4)

    def test_deep_preset(self):
        kwargs = model_registry.resolve_kwargs('M2', 'deep')
        self.assertEqual(kwargs['seconds'], 5.0)
        self.assertEqual(kwargs['depth'], 6)

    def test_unknown_preset_rejected(self):
        with self.assertRaises(model_registry.ArgPresetNotFoundError):
            model_registry.resolve_kwargs('M2', 'no_such_preset')

    def test_description_is_not_a_factory_kwarg(self):
        """config.json 里的 description 是给 UI 看的，不能被当引擎参数。"""
        for preset in ('default', 'fast', 'deep'):
            kwargs = model_registry.resolve_kwargs('M2', preset)
            self.assertNotIn('description', kwargs)

    def test_package_weights_match_upstream_readme(self):
        """两个权重必须与上游 README 声明的 SHA-256 一致。"""
        if not _package_ready():
            self.skipTest('M2 包未部署（import 根下没有 M2/）')
        self.assertEqual(
            _sha256(POLICY_WEIGHTS), POLICY_SHA256,
            '策略网 chess_model_balanced.pt 与上游声明不符'
        )
        self.assertEqual(
            _sha256(_value_weights()), VALUE_SHA256,
            '价值网 value_full_epoch2.pt 与上游 declare 不符'
        )


@unittest.skipUnless(
    _m2_available() and _torch_available() and _package_ready(),
    'models/M2 未部署、无 torch 或包缺失',
)
class TestM2Contract(unittest.TestCase):
    """六方法契约 + 服务层集成。"""

    def _make_session(self, engine_white=False, fen=None):
        """经 mock 注入极小预算，走真实的 create/setup 链路。"""
        manager = sm.SessionManager(max_sessions=4)
        patcher = mock.patch.object(
            model_registry, 'resolve_kwargs', return_value=_tiny_kwargs()
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        session = manager.create('M2', 'default', fen, engine_white)
        self.addCleanup(manager.close, session.session_id)
        return session

    @staticmethod
    def _histories(session):
        return (
            [m.uci() for m in session.board.move_stack],
            session.engine.state()['history'],
        )

    def test_constructor_defaults(self):
        with mock.patch.object(model_registry, 'resolve_kwargs', return_value={}):
            engine = model_registry.create_engine('M2', None)
        try:
            self.assertEqual(engine.seconds, 2.0)
            self.assertEqual(engine.depth, 5)
            self.assertEqual(engine.qdepth, 6)
            self.assertTrue(engine.claim_draw)
            self.assertEqual(engine.threads, 4)
        finally:
            engine.cleanup()

    def test_rejects_unknown_kwarg(self):
        engine_cls = model_registry._load_engine_class('M2')
        with self.assertRaises(TypeError):
            engine_cls(ckpt='whatever')

    def test_rejects_out_of_range_budget(self):
        engine_cls = model_registry._load_engine_class('M2')
        for bad in ({'seconds': 0}, {'seconds': 400}, {'depth': 0}, {'depth': 11}):
            with self.assertRaises(ValueError):
                engine_cls(**bad)

    def test_full_round_trip_engine_white(self):
        """引擎执白：开局自动走子 -> 人类同步 -> 引擎应答 -> 两边自洽。"""
        session = self._make_session(engine_white=True)
        self.assertEqual(len(session.board.move_stack), 1)
        state = session.state()
        self.assertEqual(len(state['san_history']), 1)
        self.assertIsNotNone(state.get('nodes'))
        self.assertIsInstance(state.get('nodes'), int)

        opening = session.board.move_stack[-1].uci()
        human_uci = 'e7e5' if opening == 'e2e4' else 'e7e6'
        result = session.human_move(human_uci)
        self.assertIn('engine_move', result)
        self.assertEqual(len(session.board.move_stack), 3)
        server, engine = self._histories(session)
        self.assertEqual(server, engine)

    def test_engine_black_waits_for_human(self):
        session = self._make_session(engine_white=False)
        self.assertEqual(len(session.board.move_stack), 0)
        result = session.human_move('d2d4')
        self.assertIn('engine_move', result)
        self.assertEqual(len(session.board.move_stack), 2)
        server, engine = self._histories(session)
        self.assertEqual(server, engine)

    def test_illegal_human_move_rejected_and_rolled_back(self):
        session = self._make_session(engine_white=False)
        before = session.board.fen()
        with self.assertRaises(sm.IllegalMoveError):
            session.human_move('e2e5')
        self.assertEqual(session.board.fen(), before)
        server, engine = self._histories(session)
        self.assertEqual(server, engine)

    def test_bad_uci_form_rejected_by_engine(self):
        engine = model_registry._load_engine_class('M2')(**_tiny_kwargs())
        self.addCleanup(engine.cleanup)
        for bad in ('e2e4z', 'e2e5', 'zzzz'):
            with self.assertRaises(ValueError):
                engine.human_move(bad)

    def test_undo_rolls_back_both_sides(self):
        session = self._make_session(engine_white=True)
        session.human_move('e7e5')
        self.assertEqual(len(session.board.move_stack), 3)
        session.undo()
        self.assertEqual(len(session.board.move_stack), 1)
        self.assertEqual(len(session.engine.state()['history']), 1)
        server, engine = self._histories(session)
        self.assertEqual(server, engine)

    def test_engine_rejects_move_from_terminal_position(self):
        engine = model_registry._load_engine_class('M2')(**_tiny_kwargs())
        self.addCleanup(engine.cleanup)
        # 一步将杀的残局：白后 g7#
        engine.setup('6k1/5ppp/8/8/8/8/5PPP/4Q1K1 w - - 0 1')
        while not engine.state()['game_over']:
            engine.human_move(engine.state()['legal_moves'][0])
        with self.assertRaises(ValueError):
            engine.engine_move()

    def test_shared_model_and_isolated_search(self):
        """权重进程级共享；评估器与搜索树每会话独立（缓冲区别共用）。"""
        e1 = model_registry.create_engine('M2', 'fast')
        e2 = model_registry.create_engine('M2', 'fast')
        try:
            self.assertIs(e1._policy, e2._policy)
            self.assertIs(e1._value, e2._value)
            self.assertIsNot(e1._evaluator, e2._evaluator)
            self.assertIsNot(e1._search, e2._search)
            # 两个会话走不同开局，局面必须互不污染
            e1.human_move('d2d4')
            e2.human_move('e2e4')
            m1 = e1.engine_move()['engine_move']
            m2 = e2.engine_move()['engine_move']
            self.assertNotEqual(e1.state()['fen'], e2.state()['fen'])
            self.assertNotEqual(e1.state()['history'], e2.state()['history'])
            self.assertTrue(m1 and m2)
        finally:
            e1.cleanup()
            e2.cleanup()

    def test_state_shape_no_fake_wdl(self):
        """M2 只有 tanh 标量：eval 恒 None，原生分放 value_tanh。"""
        session = self._make_session(engine_white=False)
        session.human_move('e2e4')
        state = session.state()
        for key in ('fen', 'legal_moves', 'history', 'san_history',
                    'last_move', 'in_check', 'game_over', 'nodes', 'depth',
                    'value_tanh', 'eval'):
            self.assertIn(key, state)
        self.assertIsNone(state['eval'])
        value = state['value_tanh']
        self.assertIsInstance(value, float)
        self.assertGreaterEqual(value, -1.0)
        self.assertLessEqual(value, 1.0)
        # 引擎侧 state 也必须是同一口径
        self.assertIsNone(session.engine.state()['eval'])

    def test_threads_compression_is_bounded(self):
        """M2 会把 torch 线程压到 4，但不能超过机器核数、也不能为非正数。"""
        engine = model_registry._load_engine_class('M2')(**_tiny_kwargs())
        self.addCleanup(engine.cleanup)
        import torch
        self.assertEqual(torch.get_num_threads(), 4)


@unittest.skipUnless(
    _m2_available() and _m3_available() and _torch_available() and _package_ready(),
    '需要 M2、M3 与 torch 同时可用',
)
class TestM2M3Coexistence(unittest.TestCase):
    """M2（裸顶层模块）与 M3（src/chess_ai 命名空间包）同进程共存。"""

    def _make(self, name, engine_white=False):
        manager = sm.SessionManager(max_sessions=4)
        kwargs = _tiny_kwargs() if name == 'M2' else {
            'device': 'cuda', 'simulations': 2, 'eval_batch_size': 1,
        }
        patcher = mock.patch.object(
            model_registry, 'resolve_kwargs', return_value=kwargs
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        session = manager.create(name, 'default', None, engine_white)
        self.addCleanup(manager.close, session.session_id)
        return session

    def test_both_orders_coexist(self):
        """先 M3 后 M2、先 M2 后 M3，两条顺序都要能各自走子。"""
        s3 = self._make('M3')
        s2 = self._make('M2')
        r3 = s3.human_move('e2e4')
        r2 = s2.human_move('d2d4')
        self.assertIn('engine_move', r2)
        self.assertEqual(len(s3.board.move_stack), 2)
        self.assertEqual(len(s2.board.move_stack), 2)
        self.assertEqual(
            [m.uci() for m in s3.board.move_stack], s3.engine.state()['history']
        )
        self.assertEqual(
            [m.uci() for m in s2.board.move_stack], s2.engine.state()['history']
        )
        self.assertNotEqual(s2.board.fen(), s3.board.fen())

    def test_chess_ai_namespace_belongs_to_m3(self):
        """M2 不得用裸模块顶替 M3 的 chess_ai 命名空间包。"""
        s3 = self._make('M3')
        s2 = self._make('M2')
        s2.human_move('d2d4')          # 触发 M2 的裸模块加载
        import chess_ai
        paths = list(getattr(chess_ai, '__path__', []))
        self.assertTrue(paths, 'chess_ai 未加载')
        self.assertIn('M3', paths[0],
                      f'chess_ai 被别的包顶替了: {paths[0]}')


if __name__ == '__main__':
    unittest.main()
