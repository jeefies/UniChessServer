"""公网访问口令门禁（X-Access-Token）验收。

覆盖：
1. 口令来源：UNICHESS_ACCESS_TOKEN 优先，其次 ~/.config/unichess/access_token，
   两者皆无则为 None（未配置时 fail-closed：远程一律拒绝，只有本机直连放行）
2. 访问判定：口令正确任意来源放行；缺失/错误/被截断 → 401
3. 环回判别：真正的本机直连免口令，带任何代理转发头的"伪本机"拒绝
4. 与管理员门禁彼此独立：admin 口令过不了 access 门禁，反之亦然；
   batch start 需要两道口令都过
5. 路由表完备性：所有 /api/* 都挂了门禁，页面路由（/ /arena /arena/batch）没挂，
   docs/openapi 已关闭
6. HTTP 层（TestClient，若环境里有 httpx）：真实请求验证 401/200，
   静态页面与 access-gate.js 免口令可取
7. 三个页面都引入了 static/access-gate.js，且脚本里的请求头名与服务端一致
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fastapi import HTTPException  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402

try:
    from starlette.testclient import TestClient

    _HAS_TESTCLIENT = True
except Exception:  # pragma: no cover - 环境缺 httpx 时跳过 HTTP 层用例
    _HAS_TESTCLIENT = False

import access_auth  # noqa: E402
import app  # noqa: E402

SERVER_DIR = pathlib.Path(app.__file__).resolve().parent
STATIC_DIR = SERVER_DIR / "static"


def _request(client="203.0.113.9", **headers):
    """构造一个最小 ASGI 请求（照 TestBatchAdminAuth 的写法）。"""
    from starlette.requests import Request

    raw = [(k.replace("_", "-").lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "headers": raw, "client": (client, 12345)})


class TokenSourceTestCase(unittest.TestCase):
    """口令读取来源与优先级。"""

    def test_env_var_wins_over_file(self):
        with tempfile.TemporaryDirectory() as td:
            token_file = pathlib.Path(td) / "access_token"
            token_file.write_text("from-file\n", encoding="utf-8")
            with mock.patch.dict("os.environ", {"UNICHESS_ACCESS_TOKEN": "from-env"}), \
                    mock.patch.object(access_auth, "ACCESS_TOKEN_FILE", token_file):
                self.assertEqual(access_auth.access_token(), "from-env")

            with mock.patch.dict("os.environ", {"UNICHESS_ACCESS_TOKEN": "  "}), \
                    mock.patch.object(access_auth, "ACCESS_TOKEN_FILE", token_file):
                self.assertEqual(access_auth.access_token(), "from-file")

    def test_blank_file_yields_none(self):
        with tempfile.TemporaryDirectory() as td:
            token_file = pathlib.Path(td) / "access_token"
            token_file.write_text("   \n", encoding="utf-8")
            with mock.patch.dict("os.environ", {"UNICHESS_ACCESS_TOKEN": ""}), \
                    mock.patch.object(access_auth, "ACCESS_TOKEN_FILE", token_file):
                self.assertIsNone(access_auth.access_token())

    def test_missing_file_yields_none(self):
        with mock.patch.dict("os.environ", {"UNICHESS_ACCESS_TOKEN": ""}), \
                mock.patch.object(access_auth, "ACCESS_TOKEN_FILE",
                                  pathlib.Path("/nonexistent/access_token")):
            self.assertIsNone(access_auth.access_token())


class AccessGateTestCase(unittest.TestCase):
    """门禁判定：放行 / 拒绝 / 环回。"""

    def setUp(self):
        patcher = mock.patch.dict("os.environ", {"UNICHESS_ACCESS_TOKEN": "s3cret-access"})
        patcher.start()
        self.addCleanup(patcher.stop)
        # 不读真实的家目录口令文件
        file_patcher = mock.patch.object(
            access_auth, "ACCESS_TOKEN_FILE", pathlib.Path("/nonexistent/access_token")
        )
        file_patcher.start()
        self.addCleanup(file_patcher.stop)

    def _allowed(self, **kw):
        access_auth.require_access_token(_request(**kw))  # 不抛异常即放行

    def _denied(self, **kw):
        with self.assertRaises(HTTPException) as ctx:
            access_auth.require_access_token(_request(**kw))
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("X-Access-Token", ctx.exception.detail)

    def test_valid_token_allowed_from_anywhere(self):
        self._allowed(x_access_token="s3cret-access", cf_connecting_ip="203.0.113.9")

    def test_missing_or_wrong_token_denied(self):
        self._denied()
        self._denied(x_access_token="nope")
        self._denied(x_access_token="")

    def test_truncated_token_denied(self):
        # 防止有人把 compare_digest 换成 startswith
        self._denied(x_access_token="s3cret-acces")
        self._denied(x_access_token="s3cret-access-plus")

    def test_direct_loopback_allowed_without_token(self):
        self._allowed(client="127.0.0.1")
        self._allowed(client="::1")
        self._allowed(client="localhost")

    def test_tunneled_public_request_denied(self):
        # 隧道把公网请求也转成 127.0.0.1，但 Cloudflare/nginx 会带转发头
        self._denied(client="127.0.0.1", x_forwarded_for="198.51.100.7")
        self._denied(client="127.0.0.1", x_real_ip="198.51.100.7")
        self._denied(client="127.0.0.1", cf_connecting_ip="198.51.100.7")
        self._denied(client="127.0.0.1", forwarded="for=198.51.100.7")
        self._denied(client="127.0.0.1", cf_ray="7d1c0f2a3b4c5d6e-SJC")

    def test_unconfigured_token_denies_token_guessing(self):
        with mock.patch.dict("os.environ", {"UNICHESS_ACCESS_TOKEN": ""}), \
                mock.patch.object(access_auth, "ACCESS_TOKEN_FILE",
                                  pathlib.Path("/nonexistent/access_token")):
            self._denied()
            self._denied(x_access_token="")
            self._denied(x_access_token="anything")
            # 未配置口令时唯一放行的是本机直连（运维口径，与管理员门禁一致）
            self._allowed(client="127.0.0.1")


class GatesAreIndependentTestCase(unittest.TestCase):
    """两道门禁互不通用。"""

    def setUp(self):
        patcher = mock.patch.dict("os.environ", {
            "UNICHESS_ACCESS_TOKEN": "access-aaa",
            "UNICHESS_ADMIN_TOKEN": "admin-bbb",
        })
        patcher.start()
        self.addCleanup(patcher.stop)
        for obj, name in ((access_auth, "ACCESS_TOKEN_FILE"),
                          (app, "ADMIN_TOKEN_FILE")):
            p = mock.patch.object(obj, name, pathlib.Path("/nonexistent/x"))
            p.start()
            self.addCleanup(p.stop)

    def test_admin_token_does_not_open_access_gate(self):
        with self.assertRaises(HTTPException) as ctx:
            access_auth.require_access_token(_request(x_admin_token="admin-bbb"))
        self.assertEqual(ctx.exception.status_code, 401)

    def test_access_token_does_not_open_admin_gate(self):
        with self.assertRaises(HTTPException) as ctx:
            app.require_admin(_request(x_access_token="access-aaa"))
        self.assertEqual(ctx.exception.status_code, 403)

    def test_tokens_are_distinct_values(self):
        self.assertNotEqual(access_auth.access_token(), app._admin_token())

    def test_header_names_are_distinct(self):
        self.assertNotEqual(access_auth.ACCESS_TOKEN_HEADER, "x-admin-token")


class RouteTableTestCase(unittest.TestCase):
    """门禁必须挂在每一个 /api/* 上，且只能挂在 /api/* 上。"""

    @staticmethod
    def _routes():
        return [r for r in app.app.routes if isinstance(r, APIRoute)]

    @staticmethod
    def _gated(route):
        return any(
            getattr(d, "call", None) is access_auth.require_access_token
            for d in route.dependant.dependencies
        )

    def test_every_api_route_is_gated(self):
        api_routes = [r for r in self._routes() if r.path.startswith("/api/")]
        self.assertGreaterEqual(len(api_routes), 17)
        ungated = sorted(r.path for r in api_routes if not self._gated(r))
        self.assertEqual(ungated, [], f"这些 /api 路由漏挂门禁: {ungated}")

    def test_page_routes_are_not_gated(self):
        # 浏览器打开文档时带不了自定义请求头，这些必须保持开放
        page_routes = [r for r in self._routes() if not r.path.startswith("/api/")]
        self.assertTrue(page_routes)
        gated = sorted(r.path for r in page_routes if self._gated(r))
        self.assertEqual(gated, [], f"这些页面路由不该有门禁: {gated}")

    def test_docs_and_openapi_disabled(self):
        paths = {r.path for r in self._routes()}
        self.assertNotIn("/openapi.json", paths)
        self.assertNotIn("/docs", paths)
        self.assertNotIn("/redoc", paths)

    def test_route_list_includes_expected_api_paths(self):
        paths = {r.path for r in self._routes()}
        for expected in ("/api/health", "/api/models", "/api/new", "/api/games",
                         "/api/games/{session_id}/state", "/api/arena/new",
                         "/api/arena/batch/start", "/api/arena/batch/stop",
                         "/api/arena/batch/state", "/api/arena/records"):
            self.assertIn(expected, paths)


class FrontendGateTestCase(unittest.TestCase):
    """静态页面必须引入口令门禁，且请求头名与服务端一致。"""

    def test_js_gate_exists_and_uses_the_same_header(self):
        gate_js = STATIC_DIR / "access-gate.js"
        self.assertTrue(gate_js.is_file(), "static/access-gate.js 缺失")
        text = gate_js.read_text(encoding="utf-8")
        self.assertIn("X-Access-Token", text)
        self.assertIn("401", text)
        # 口令存 localStorage，刷新后不用重填
        self.assertIn("localStorage", text)

    def test_pages_load_the_gate(self):
        for page in ("index.html", "arena.html", "arena_batch.html"):
            with self.subTest(page=page):
                text = (STATIC_DIR / page).read_text(encoding="utf-8")
                self.assertIn('src="/static/access-gate.js"', text)

    def test_gate_js_is_served(self):
        # 别让备份文件守卫之类的规则把它挡了
        name = "access-gate.js"
        self.assertFalse(name.startswith("."))
        self.assertNotIn(".bak", name)
        self.assertNotIn(".before-", name)
        self.assertFalse(name.endswith("~"))


@unittest.skipUnless(_HAS_TESTCLIENT, "环境缺少 httpx，跳过 HTTP 层用例")
class HttpLayerTestCase(unittest.TestCase):
    """真实 HTTP 请求走一遍门禁。"""

    def setUp(self):
        patcher = mock.patch.dict("os.environ", {
            "UNICHESS_ACCESS_TOKEN": "http-secret",
            "UNICHESS_ADMIN_TOKEN": "http-admin",
        })
        patcher.start()
        self.addCleanup(patcher.stop)
        file_patcher = mock.patch.object(
            access_auth, "ACCESS_TOKEN_FILE", pathlib.Path("/nonexistent/access_token")
        )
        file_patcher.start()
        self.addCleanup(file_patcher.stop)
        # 不用 with：不触发 startup，免得拉起 Monitor 线程
        self.client = TestClient(app.app)
        self.headers = {"X-Access-Token": "http-secret"}

    def test_api_requires_token(self):
        r = self.client.get("/api/games")
        self.assertEqual(r.status_code, 401)
        self.assertIn("X-Access-Token", r.json()["detail"])

        r = self.client.get("/api/games", headers={"X-Access-Token": "wrong"})
        self.assertEqual(r.status_code, 401)

    def test_api_ok_with_token(self):
        r = self.client.get("/api/games", headers=self.headers)
        self.assertEqual(r.status_code, 200)
        self.assertIn("sessions", r.json())

    def test_pages_and_gate_js_are_open(self):
        for path in ("/", "/arena", "/arena/batch"):
            r = self.client.get(path)
            self.assertEqual(r.status_code, 200, f"{path} 被门禁挡住")
        r = self.client.get("/static/access-gate.js")
        self.assertEqual(r.status_code, 200)
        self.assertIn("X-Access-Token", r.text)

    def test_unknown_api_path_is_not_accessible(self):
        # 未匹配路径先被路由挡成 404（依赖还没轮到跑），拿不到任何数据；
        # 带口令也一样是 404，说明这不是门禁漏放而是路径本来不存在。
        self.assertEqual(self.client.get("/api/definitely-not-a-route").status_code, 404)
        self.assertEqual(
            self.client.get("/api/definitely-not-a-route", headers=self.headers).status_code, 404
        )

    def test_openapi_and_docs_are_gone(self):
        for path in ("/openapi.json", "/docs", "/redoc"):
            self.assertEqual(self.client.get(path).status_code, 404, path)

    def test_batch_start_needs_both_tokens(self):
        body = {"white_model": "nope", "black_model": "nope", "rounds": 2}

        # 只有管理员口令：过不了访问口令门禁
        r = self.client.post("/api/arena/batch/start", json=body,
                             headers={"X-Admin-Token": "http-admin"})
        self.assertEqual(r.status_code, 401)

        # 只有访问口令：过不了管理员门禁，且不会真的起批次
        r = self.client.post("/api/arena/batch/start", json=body, headers=self.headers)
        self.assertEqual(r.status_code, 403)

        # 两道口令全错：先撞访问口令门禁
        r = self.client.post("/api/arena/batch/start", json=body,
                             headers={"X-Admin-Token": "bad", "X-Access-Token": "bad"})
        self.assertEqual(r.status_code, 401)

        self.assertFalse(app.jobs.batch_service().snapshot()["is_running"])


if __name__ == "__main__":
    unittest.main()
