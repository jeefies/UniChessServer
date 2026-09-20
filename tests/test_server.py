"""UniChessServer 验收测试套子（标准库 unittest，远端 conda 环境运行）。

覆盖本轮代码审查的全部修复点：
1. models/__init__.py 路径穿越防护（CRITICAL）
2. /api/models 如实上报 not_implemented（不再谎报 available）
3. 每会话锁：同一 session 的并发请求串行（双 Tab/重复提交不再写坏引擎）
4. undo 后轮到引擎时重新触发 engine_move（不再永久卡死）
5. human_move 引擎失败时回滚服务层 Board（不再永久错位）
6. create() 中 setup 失败时释放已建引擎（不再泄漏）
7. LRU 淘汰在锁外 cleanup，且淘汰数受 MAX_SESSIONS 约束
8. engine_move() 缺少 "engine_move" 键时硬报错（不再静默错位）
9. state() 的 fen 以服务层权威 Board 为准（引擎返回值不覆盖）
10. 路由层错误码：非法 FEN 400、未接入 501、会话不存在 404、非法走法 400

运行（远端）：
    cd /home/jeefy/UniChess/Server && \
    /home/jeefy/miniconda3/envs/unichess/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import pathlib
import sys
import threading
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import chess  # noqa: E402

import app  # noqa: E402
import models as model_registry  # noqa: E402
import session_manager as sm  # noqa: E402


class FakeEngine:
    """契约完整的假引擎：确定性走"当前局面第一个合法着"，可注入故障。

    instances 记录所有构造出来的实例，用于断言 cleanup/泄漏。
    """

    instances: list['FakeEngine'] = []

    def __init__(self, **kwargs):
        self.board = chess.Board()
        self.cleaned = False
        self.fail_setup = False
        self.fail_human_move = False
        self.omit_engine_move_key = False
        self.bogus_state_fen = False
        # 应用注入的故障开关（fail_setup / bogus_state_fen / ...）
        for k, v in kwargs.items():
            setattr(self, k, v)
        FakeEngine.instances.append(self)

    # -- 契约六方法 -------------------------------------------------------
    def setup(self, fen: str | None = None) -> dict:
        if self.fail_setup:
            raise RuntimeError('FakeEngine setup boom')
        self.board = chess.Board(fen) if fen else chess.Board()
        return {'fen': self.board.fen()}

    def human_move(self, uci: str) -> dict:
        if self.fail_human_move:
            raise RuntimeError('FakeEngine human_move boom')
        self.board.push(chess.Move.from_uci(uci))
        return {'fen': self.board.fen()}

    def engine_move(self) -> dict:
        move = next(iter(self.board.legal_moves), None)
        if self.omit_engine_move_key:
            return {'fen': self.board.fen()}
        assert move is not None, 'FakeEngine.engine_move 在终局被调用'
        self.board.push(move)
        return {'engine_move': move.uci(), 'fen': self.board.fen()}

    def state(self) -> dict:
        fen = '8/8/8/8/8/8/8/8 w - - 0 1' if self.bogus_state_fen else self.board.fen()
        return {'fen': fen, 'san_history': ['<b>x</b>'], 'source': 'fake'}

    def undo(self) -> dict:
        for _ in range(2):
            if self.board.move_stack:
                self.board.pop()
        return {'fen': self.board.fen()}

    def cleanup(self) -> None:
        self.cleaned = True

    # -- 测试辅助 ---------------------------------------------------------
    @property
    def last_engine(self):
        return self


def make_engine(**flags):
    """经 mock 让 model_registry.create_engine 返回注入了故障的 FakeEngine。"""
    eng = FakeEngine(**flags)
    return eng, mock.patch.object(model_registry, 'create_engine', return_value=eng)


class ServerTestCase(unittest.TestCase):
    def setUp(self):
        FakeEngine.instances.clear()
        # 每个用例独立的管理器，避免全局单例串味
        self.manager = sm.SessionManager(max_sessions=4)
        self._manager_patcher = mock.patch.object(sm, 'manager', self.manager)
        self._manager_patcher.start()
        self.addCleanup(self._manager_patcher.stop)

    def tearDown(self):
        for s in list(self.manager._sessions.values()):
            try:
                s.cleanup()
            except Exception:
                pass


class TestModelRegistrySecurity(ServerTestCase):
    """模型名安全与状态上报。"""

    def test_traversal_names_rejected(self):
        for bad in ['../../tmp/evil', '..', '.', '.hidden', 'a/b', 'resnet/../resnet',
                    '', 'resnet/', '/etc', '..%2f']:
            with self.subTest(bad=bad), self.assertRaises(model_registry.ModelNotFoundError):
                model_registry.create_engine(bad)

    def test_valid_name_passes_validation(self):
        # resnet 是真实存在的占位模型：校验通过、进入 IMPLEMENTED 检查
        with self.assertRaises(model_registry.ModelNotImplementedError):
            model_registry.create_engine('resnet')

    def test_symlinked_model_dir_is_allowed(self):
        """运营者创建的符号链接模型目录应可正常加载（集成外部项目的既定方式）。"""
        import tempfile
        link_name = 'test_symlink_model'
        link_path = model_registry.MODELS_DIR / link_name
        with tempfile.TemporaryDirectory() as td:
            target = pathlib.Path(td)
            (target / 'engine.py').write_text(
                'class GameEngine:\n'
                '    IMPLEMENTED = True\n'
                '    def __init__(self, **kw): pass\n'
                '    def setup(self, fen=None): return {}\n'
                '    def human_move(self, uci): return {}\n'
                '    def engine_move(self): return {"engine_move": "e2e4"}\n'
                '    def state(self): return {}\n'
                '    def undo(self): return {}\n'
                '    def cleanup(self): pass\n',
                encoding='utf-8',
            )
            link_path.symlink_to(target, target_is_directory=True)
            try:
                self.assertIn(link_name, model_registry.available_models())
                info = model_registry.describe_model(link_name)
                self.assertEqual(info['status'], 'available')
                eng = model_registry.create_engine(link_name)
                self.assertTrue(hasattr(eng, 'engine_move'))
            finally:
                link_path.unlink()
        # 清理后不再列出
        self.assertNotIn(link_name, model_registry.available_models())

    def test_stub_model_status_not_implemented(self):
        """resnet 仍是占位引擎：状态必须如实报 not_implemented。"""
        info = model_registry.describe_model('resnet')
        self.assertEqual(info['status'], 'not_implemented')
        self.assertTrue(info['reason'])

    def test_symlinked_real_engine_is_available(self):
        """models/T 符号链接（Transformer 项目真实引擎）应可用。

        T 只在配了符号链接的主机上存在；没有则跳过（其它主机只有 resnet）。
        """
        if 'T' not in model_registry.available_models():
            self.skipTest('models/T 符号链接未配置')
        info = model_registry.describe_model('T')
        self.assertEqual(info['status'], 'available')
        self.assertIsInstance(info['presets'], list)


class TestSessionLifecycle(ServerTestCase):
    """修复 3/4/5/6/8/9 + 既有契约。"""

    def _new(self, engine_white=False, fen=None, **flags):
        eng, patcher = make_engine(**flags)
        with patcher:
            session = self.manager.create('fake', None, fen, engine_white)
        return session, eng

    def test_engine_black_setup_no_engine_move(self):
        session, eng = self._new(engine_white=False)
        self.assertEqual(len(session.board.move_stack), 0)
        self.assertEqual(session.state()['legal_moves'],
                         [m.uci() for m in chess.Board().legal_moves])

    def test_engine_white_setup_triggers_opening(self):
        session, eng = self._new(engine_white=True)
        self.assertEqual(len(session.board.move_stack), 1)
        self.assertEqual(session.board.turn, chess.BLACK)
        # 服务层 Board 与引擎内部局面一致
        self.assertEqual(eng.board.fen(), session.board.fen())

    def test_human_move_then_engine_reply(self):
        session, eng = self._new()
        result = session.human_move('e2e4')
        self.assertIn('engine_move', result)
        self.assertEqual(len(session.board.move_stack), 2)
        self.assertEqual(session.board.turn, chess.WHITE)
        self.assertEqual(eng.board.fen(), session.board.fen())

    def test_illegal_move_rejected_and_board_untouched(self):
        session, eng = self._new()
        with self.assertRaises(sm.IllegalMoveError):
            session.human_move('e2e5')
        self.assertEqual(len(session.board.move_stack), 0)

    def test_human_move_when_engine_turn_rejected(self):
        """引擎轮到的位置提交人类走法必须被拒。

        正常对局里服务层总会推进到人类轮，唯一能构造出"引擎轮到"的
        合法途径是终局 FEN（setup 不会触发引擎走子）。
        这里用一步杀 FEN（白被将死、白行棋）且引擎执白。
        """
        mate_fen = 'rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3'
        session, eng = self._new(engine_white=True, fen=mate_fen)
        self.assertTrue(session.board.is_game_over())
        with self.assertRaises(sm.IllegalMoveError):
            session.human_move('e2e4')

    def test_engine_failure_rolls_back_board(self):
        """修复 5：引擎抛异常后服务层 Board 必须回滚，双方停同一局面。"""
        session, eng = self._new(fail_human_move=True)
        with self.assertRaises(RuntimeError):
            session.human_move('e2e4')
        self.assertEqual(len(session.board.move_stack), 0)
        self.assertEqual(eng.board.fen(), session.board.fen())

    def test_undo_engine_white_retriggers_engine(self):
        """修复 3：引擎执白开局后人类未走就悔棋，必须重新触发引擎走子。

        这是唯一会卡死的场景：pop 掉唯一一步后回到初始局面、轮到引擎，
        而 undo 路径若不重跑调度，就再没有请求会调用 engine_move()，
        人类提交又会被 IllegalMoveError 拒绝——对局永久停在"引擎思考中"。
        （走完一整轮后再悔棋回到的是人类轮，无需重触发，由另一个用例覆盖。）
        """
        session, eng = self._new(engine_white=True)
        self.assertEqual(len(session.board.move_stack), 1)
        result = session.undo()
        self.assertIn('engine_move', result)
        self.assertEqual(len(session.board.move_stack), 1)
        self.assertEqual(session.board.turn, chess.BLACK)  # 引擎重新走完，又轮到人类
        self.assertEqual(eng.board.fen(), session.board.fen())

    def test_undo_engine_black_back_to_human(self):
        session, eng = self._new(engine_white=False)
        session.human_move('e2e4')
        self.assertEqual(len(session.board.move_stack), 2)
        session.undo()
        self.assertEqual(len(session.board.move_stack), 0)
        self.assertEqual(session.board.turn, chess.WHITE)

    def test_undo_after_full_round_stays_human_turn(self):
        """对照：引擎执白走完整一轮后悔棋，回到人类轮，引擎不应多走。"""
        session, eng = self._new(engine_white=True)
        session.human_move('d7d5')
        self.assertEqual(len(session.board.move_stack), 3)
        result = session.undo()
        self.assertNotIn('engine_move', result)
        self.assertEqual(len(session.board.move_stack), 1)
        self.assertEqual(session.board.turn, chess.BLACK)
        self.assertEqual(eng.board.fen(), session.board.fen())

    def test_missing_engine_move_key_is_hard_error(self):
        """修复 8：engine_move() 不返回 "engine_move" 键必须报错。"""
        session, eng = self._new(omit_engine_move_key=True)
        with self.assertRaises(sm.SessionError):
            session.human_move('e2e4')

    def test_state_fen_uses_service_board(self):
        """修复 9：引擎返回的 fen 不得覆盖服务层权威 Board。"""
        session, eng = self._new(bogus_state_fen=True)
        state = session.state()
        self.assertEqual(state['fen'], chess.Board().fen())
        self.assertNotEqual(state['fen'], '8/8/8/8/8/8/8/8 w - - 0 1')

    def test_close_calls_cleanup_once(self):
        session, eng = self._new()
        self.manager.close(session.session_id)
        self.assertTrue(eng.cleaned)
        self.assertEqual(self.manager.list_sessions(), [])
        # 关闭不存在的会话不报错（DELETE 幂等）
        self.manager.close('nonexistent')


class TestSessionManagerCapacity(ServerTestCase):
    """修复 6/7：create 泄漏、淘汰 cleanup。"""

    def test_setup_failure_cleans_up_engine(self):
        """修复 6：setup 失败时已建引擎必须被 cleanup，不能泄漏。"""
        eng, patcher = make_engine(fail_setup=True)
        with patcher:
            with self.assertRaises(RuntimeError):
                self.manager.create('fake')
        self.assertTrue(eng.cleaned)

    def test_eviction_cleans_up_oldest(self):
        manager = sm.SessionManager(max_sessions=2)
        with mock.patch.object(sm, 'manager', manager):
            engines = []
            for _ in range(3):
                eng, patcher = make_engine()
                with patcher:
                    s = manager.create('fake')
                engines.append((s, eng))
            self.assertEqual(len(manager.list_sessions()), 2)
            self.assertTrue(engines[0][1].cleaned)   # 最旧被淘汰并 cleanup
            self.assertFalse(engines[1][1].cleaned)
            self.assertFalse(engines[2][1].cleaned)
            for s, _ in engines[1:]:
                manager.close(s.session_id)

    def test_get_refreshes_lru_order(self):
        manager = sm.SessionManager(max_sessions=2)
        with mock.patch.object(sm, 'manager', manager):
            made = []
            for _ in range(2):
                eng, patcher = make_engine()
                with patcher:
                    made.append((manager.create('fake'), eng))
            manager.get(made[0][0].session_id)  # 刷新最旧 → made[1] 变最旧
            eng, patcher = make_engine()
            with patcher:
                manager.create('fake')
            self.assertFalse(made[0][1].cleaned)
            self.assertTrue(made[1][1].cleaned)
            for s, _ in made:
                try:
                    manager.close(s.session_id)
                except Exception:
                    pass


class TestConcurrency(ServerTestCase):
    """修复 2：同一会话的并发请求必须串行。"""

    def test_concurrent_same_move_serialized(self):
        eng, patcher = make_engine()
        with patcher:
            session = self.manager.create('fake')
        results = {'ok': 0, 'err': 0}
        lock = threading.Lock()

        def worker():
            try:
                session.human_move('e2e4')
                with lock:
                    results['ok'] += 1
            except sm.IllegalMoveError:
                with lock:
                    results['err'] += 1

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 只有一个请求能成功推 e2e4（之后该着法不再合法），且 Board/引擎不错位
        self.assertEqual(results['ok'], 1)
        self.assertEqual(results['err'], 7)
        self.assertEqual(len(session.board.move_stack), 2)
        self.assertEqual(eng.board.fen(), session.board.fen())

    def test_concurrent_engine_white_moves_serialized(self):
        """引擎执白时同着法并发：只有第一个能推入，Board/引擎始终一致。"""
        eng, patcher = make_engine()
        with patcher:
            session = self.manager.create('fake', None, None, True)
        ok = []

        def worker(uci):
            try:
                session.human_move(uci)
                ok.append(uci)
            except sm.IllegalMoveError:
                pass

        threads = [threading.Thread(target=worker, args=('d7d5',)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # d7d5 只有第一个请求能成功（之后该着法不再合法）；
        # 走完一整轮后又轮到人类，但同样的着法已非法，不会二次推入。
        self.assertEqual(len(ok), 1)
        self.assertEqual(len(session.board.move_stack), 3)
        self.assertEqual(eng.board.fen(), session.board.fen())


class TestRoutes(ServerTestCase):
    """路由层错误码与 /api/models 状态。"""

    def test_models_lists_real_status(self):
        body = app.list_models()
        self.assertEqual(body['models']['resnet']['status'], 'not_implemented')
        # T（Transformer 项目符号链接引擎）在本机已配置时必须报 available
        if 'T' in model_registry.available_models():
            self.assertEqual(body['models']['T']['status'], 'available')
        # 已删除的冗长占位名不应再出现
        self.assertNotIn('transformer', body['models'])

    def test_new_game_stub_returns_501(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            app.new_game(app.NewGameRequest(model_name='resnet'))
        self.assertEqual(ctx.exception.status_code, 501)
        # 501 文案面向用户，不得泄露内部迁移路线（旧文案含"尚未迁移"）
        self.assertNotIn('尚未迁移', ctx.exception.detail)
        self.assertIn('暂不可对局', ctx.exception.detail)

    def test_new_game_traversal_returns_404(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            app.new_game(app.NewGameRequest(model_name='../../tmp/evil'))
        self.assertEqual(ctx.exception.status_code, 404)

    def test_new_game_bad_fen_returns_400(self):
        from fastapi import HTTPException
        eng, patcher = make_engine()
        with patcher:
            with self.assertRaises(HTTPException) as ctx:
                app.new_game(app.NewGameRequest(model_name='fake', fen='not-a-fen'))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_full_move_flow_through_routes(self):
        eng, patcher = make_engine()
        with patcher:
            resp = app.new_game(app.NewGameRequest(model_name='fake'))
        sid = resp['session_id']
        self.assertEqual(resp['state']['is_game_over'], False)

        state = app.game_state(sid)
        self.assertEqual(state['state']['fen'], chess.Board().fen())

        moved = app.make_move(sid, app.MoveRequest(uci='e2e4'))
        self.assertEqual(len(moved['state']['legal_moves']) > 0, True)

        undone = app.undo_move(sid)
        self.assertEqual(undone['state']['fen'], chess.Board().fen())

        closed = app.close_game(sid)
        self.assertTrue(closed['closed'])

        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            app.game_state(sid)
        self.assertEqual(ctx.exception.status_code, 404)

    def test_illegal_move_route_returns_400(self):
        from fastapi import HTTPException
        eng, patcher = make_engine()
        with patcher:
            resp = app.new_game(app.NewGameRequest(model_name='fake'))
        try:
            with self.assertRaises(HTTPException) as ctx:
                app.make_move(resp['session_id'], app.MoveRequest(uci='e2e5'))
            self.assertEqual(ctx.exception.status_code, 400)
        finally:
            app.close_game(resp['session_id'])

    def test_move_unknown_session_returns_404(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            app.make_move('nope', app.MoveRequest(uci='e2e4'))
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == '__main__':
    unittest.main()
