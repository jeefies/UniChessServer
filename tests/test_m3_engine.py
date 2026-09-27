"""M3 引擎接入验收测试。

覆盖：
1. 模型发现与状态上报（models/M3 符号链接可用，预设可解析）
2. 默认配置即上游 M6 模型代码默认配置中最强一档（64 模拟 / 批量 8 / 树复用）
3. 六方法契约端到端（真实权重 + 极小模拟预算）
4. 共享模型缓存：同 (artifact, device) 只加载一次，会话间共享权重但搜索树独立
5. 包完整性：SHA256SUMS 与实际文件一致（部署后自检）
6. 服务层集成：非法走法回滚、悔棋双方同步、引擎执黑等待人类、
   state() 的 fen 以服务层权威 Board 为准

2026-09-27 起对应上游冻结包 `M6-30M-UniChessServer-playable.zip`（30M finalist，
模型在 Server 注册名为 `M3`，包内文件名与 SHA 仍是上游原样，勿改）：
`src/chess_ai/` 布局 + `weights/m6-3p6m-30m-inference.pt`）。该包不可改动，
因此本文件的断言只能依据它的**公开契约**：六方法、config.json 预设、
`state()` 字段与 `engine.py` 模块级缓存钩子。

说明：引擎仅支持 GPU 推理（无 CPU 配置），CUDA 不可用时相关用例 skip。
契约用例用 `simulations=2` 的极小预算，单步约 0.1s，整套几秒内跑完。
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

M3_DIR = pathlib.Path(__file__).resolve().parent.parent / 'models' / 'M3'
M3_ARTIFACT = M3_DIR / 'weights' / 'm6-3p6m-30m-inference.pt'


def _m3_available() -> bool:
    return 'M3' in model_registry.available_models()


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _tiny_kwargs() -> dict:
    """契约用例用的极小预算 kwargs（经 mock 注入 resolve_kwargs）。

    M3 的 engine.py 只接受 simulations / eval_batch_size / reuse_tree / c_puct /
    device 五个参数，权重路径由包内硬编码（它是冻结产物），不再有 ckpt 参数。
    """
    return {
        'device': 'cuda',
        'simulations': 2,
        'eval_batch_size': 1,
    }


def _engine_globals(engine):
    """取 engine.py 的模块全局命名空间。

    冻结包的权重缓存钩子 `_load_shared_model` 是模块级函数、不在类上暴露，
    共享缓存的断言只能从这里拿（包不可改，加公开入口不属于我们能做的）。
    """
    return type(engine).__init__.__globals__


class TestM3Discovery(unittest.TestCase):
    """模型发现层：不加载权重，只验证上报与预设解析。"""

    def setUp(self):
        if not _m3_available():
            self.skipTest('models/M3 符号链接未配置')

    def test_m6_reported_available(self):
        info = model_registry.describe_model('M3')
        self.assertEqual(info['status'], 'available')
        self.assertIn('default', info['presets'])
        self.assertIn('preview', info['presets'])

    def test_default_preset_is_strongest_code_default(self):
        """默认预设必须是上游 M6 模型代码默认配置中最强的一档。"""
        kwargs = model_registry.resolve_kwargs('M3', 'default')
        self.assertEqual(kwargs['simulations'], 64)
        self.assertEqual(kwargs['eval_batch_size'], 8)
        self.assertTrue(kwargs['reuse_tree'])
        self.assertEqual(kwargs['c_puct'], 1.5)

    def test_preview_preset_matches_shipped_defaults(self):
        kwargs = model_registry.resolve_kwargs('M3', 'preview')
        self.assertEqual(kwargs['simulations'], 16)
        self.assertEqual(kwargs['eval_batch_size'], 8)
        self.assertTrue(kwargs['reuse_tree'])

    def test_unknown_preset_rejected(self):
        with self.assertRaises(model_registry.ArgPresetNotFoundError):
            model_registry.resolve_kwargs('M3', 'no_such_preset')

    def test_bundle_artifact_present(self):
        """包内必须带推理权重，且 SHA256SUMS 的条目数与实际文件对得上。"""
        self.assertTrue(M3_ARTIFACT.is_file(),
                        f'缺少包内推理权重: {M3_ARTIFACT}')
        lines = (M3_DIR / 'SHA256SUMS').read_text(encoding='utf-8').splitlines()
        entries = [l for l in lines if l.strip()]
        self.assertTrue(entries, 'SHA256SUMS 为空')
        listed = {l.split(None, 1)[1].strip() for l in entries}
        actual = {
            str(p.relative_to(M3_DIR)).replace('\\', '/')
            for p in M3_DIR.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts
            and p.name != 'SHA256SUMS'  # 清单不列自己
        }
        self.assertSetEqual(listed, actual,
                            'SHA256SUMS 与实际文件不一致（多出或缺失）')

    @unittest.skipUnless(_cuda_available(), 'CUDA 不可用')
    def test_bundle_integrity(self):
        """逐文件核验 SHA256SUMS：部署后自检，权重被换/截断立即暴露。"""
        bad = []
        for line in (M3_DIR / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            digest, name = line.split(None, 1)
            path = M3_DIR / name.strip()
            if not path.is_file():
                bad.append(f'{name}: 缺失')
                continue
            h = hashlib.sha256(path.read_bytes()).hexdigest()
            if h != digest.strip():
                bad.append(f'{name}: 校验和不符')
        self.assertEqual(bad, [], '包完整性校验失败')

    @unittest.skipUnless(_cuda_available(), 'CUDA 不可用')
    def test_constructor_defaults_are_strongest_code_default(self):
        """不传 arg_name 时，构造参数本身就是最强默认档。"""
        with mock.patch.object(
            model_registry, 'resolve_kwargs', return_value={}
        ):
            engine = model_registry.create_engine('M3', None)
        try:
            self.assertEqual(engine.simulations, 64)
            self.assertEqual(engine.eval_batch_size, 8)
            self.assertTrue(engine.reuse_tree)
            self.assertEqual(engine.c_puct, 1.5)
            # device 是 torch.device 而非字符串（新版包改过类型）
            self.assertEqual(engine.device.type, 'cuda')
        finally:
            engine.cleanup()

    @unittest.skipUnless(_cuda_available(), 'CUDA 不可用')
    def test_cpu_device_rejected(self):
        """包明确禁用 CPU 回退，device='cpu' 必须显式报错而不是悄悄退化。"""
        engine_cls = model_registry._load_engine_class('M3')
        with self.assertRaises(ValueError):
            engine_cls(device='cpu')


@unittest.skipUnless(
    _m3_available() and _cuda_available(),
    'models/M3 符号链接未配置或 CUDA 不可用',
)
class TestM3Contract(unittest.TestCase):
    """六方法契约 + 服务层集成：真实权重 + 极小预算。"""

    def _make_session(self, engine_white=False, fen=None):
        """经 mock 注入极小预算，走真实的 create/setup 链路。"""
        manager = sm.SessionManager(max_sessions=4)
        patcher = mock.patch.object(
            model_registry, 'resolve_kwargs', return_value=_tiny_kwargs()
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        session = manager.create('M3', 'default', fen, engine_white)
        self.addCleanup(manager.close, session.session_id)
        return session

    @staticmethod
    def _histories(session):
        """(服务层, 引擎) 双方的着法序列，用于比对两边是否同步。

        不能直接比 fen：新版包的 state() 用 `en_passant="fen"`（X-FEN 口径，
        双步进兵后总写给区格），服务层用 python-chess 默认口径（仅在有合法
        吃过路兵时才写给区格）。两者在"能否吃过路兵"上完全等价，差别只在
        字符串，服务层 state() 又会用自己的 fen 覆盖，故比着法序列。
        """
        server = [m.uci() for m in session.board.move_stack]
        return server, session.engine.state()['history']

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
        # 服务层权威 Board 与引擎 Board 必须同步
        server, engine = self._histories(session)
        self.assertEqual(server, engine)

    def test_illegal_human_move_rejected_and_rolled_back(self):
        session = self._make_session(engine_white=True)
        fen_before = session.board.fen()
        with self.assertRaises(sm.IllegalMoveError):
            session.human_move('e2e4')  # 引擎执白已占 e4，不合法
        self.assertEqual(session.board.fen(), fen_before)
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

    def test_undo_rolls_back_both_sides(self):
        session = self._make_session(engine_white=True)
        session.human_move('e7e5')
        self.assertEqual(len(session.board.move_stack), 3)
        session.undo()
        # 退掉引擎应答与人类走子，剩引擎开局第一步，且轮到人类
        self.assertEqual(len(session.board.move_stack), 1)
        self.assertEqual(len(session.engine.state()['history']), 1)
        server, engine = self._histories(session)
        self.assertEqual(server, engine)

    def test_undo_from_start_is_bounded(self):
        """开局第一步之前就悔棋：不能倒退到负数步，两边仍须同步。"""
        session = self._make_session(engine_white=False)
        session.undo()
        self.assertEqual(len(session.board.move_stack), 0)
        self.assertEqual(session.engine.state()['history'], [])
        # 悔棋后轮到引擎（引擎执白场景由 create 触发），这里引擎执黑，应仍等人类
        result = session.human_move('d2d4')
        self.assertIn('engine_move', result)
        self.assertEqual(len(session.board.move_stack), 2)

    def test_shared_model_and_isolated_search(self):
        """同参数多个会话共享一份权重，但各自持有独立搜索树。"""
        s1 = self._make_session(engine_white=False)
        s2 = self._make_session(engine_white=False)

        globals_ = _engine_globals(s1.engine)
        load_shared = globals_['_load_shared_model']
        device = s1.engine.device
        model_a, _ = load_shared(device)
        model_b, _ = load_shared(device)
        self.assertIs(model_a, model_b, '同 (artifact, device) 必须只加载一次权重')

        # 搜索树会话级隔离：两边走不同开局，局面互不污染
        s1.human_move('e2e4')
        s2.human_move('d2d4')
        self.assertNotEqual(self._histories(s1)[1], self._histories(s2)[1])
        self.assertIsNot(s1.engine._mcts, s2.engine._mcts)

    def test_cleanup_releases_search_but_keeps_shared_model(self):
        """cleanup 只释放会话级搜索树，共享权重必须保留给其它会话。"""
        s1 = self._make_session(engine_white=False)
        s2 = self._make_session(engine_white=False)
        s1.human_move('e2e4')
        s1.engine.cleanup()
        self.assertIsNone(s1.engine._mcts)
        s1.engine.cleanup()  # 幂等，第二次不得抛异常
        # 另一个会话不受影响，仍能正常应答
        result = s2.human_move('d2d4')
        self.assertIn('engine_move', result)

    def test_state_fen_is_authoritative_server_board(self):
        """state() 的 fen/legal_moves 以服务层 Board 为准，不被引擎口径覆盖。"""
        session = self._make_session(engine_white=False)
        session.human_move('e2e4')   # 紧随其后引擎 e7e5：过路格写法分歧的场景
        state = session.state()
        self.assertEqual(state['fen'], session.board.fen())
        self.assertEqual(
            state['legal_moves'],
            [m.uci() for m in session.board.legal_moves],
        )
        self.assertEqual(state['is_game_over'], False)


if __name__ == '__main__':
    unittest.main()
