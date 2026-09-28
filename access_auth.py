"""公网访问口令门禁：所有 /api/* 一律要求请求头 X-Access-Token。

与 `app.py` 里的管理员门禁（`X-Admin-Token`，只保护批量对弈 start/stop）
**彼此独立**，刻意不复用对方的任何函数与常量：

- 口令来源不同：access 走 `UNICHESS_ACCESS_TOKEN` / `~/.config/unichess/access_token`，
  admin 走 `UNICHESS_ADMIN_TOKEN` / `~/.config/unichess/admin_token`；
- 覆盖面不同：access 覆盖**全部** `/api/*`，admin 只在批量对弈 start/stop 之上再叠一层；
- 代码不同文件：复制一份环回/转发头判别逻辑，是为了日后加转发头时不会
  "改了 admin 忘了 access"（`tests/test_access_auth.py` 有对拍，两者漂移会红）。

静态页面（`/`、`/static/*`）**不设**门禁：浏览器打开文档时无法带自定义请求头，
口令只能由页面 JS 带上（见 `static/access-gate.js`：注入请求头 + 401 时弹遮罩索要）。
"""
from __future__ import annotations

import hmac
import os
from pathlib import Path

from fastapi import HTTPException, Request

ACCESS_TOKEN_FILE = Path.home() / ".config" / "unichess" / "access_token"
ACCESS_TOKEN_ENV = "UNICHESS_ACCESS_TOKEN"
ACCESS_TOKEN_HEADER = "x-access-token"

_DENIED_DETAIL = "本服务的接口需要访问口令：请求头 X-Access-Token 与主机配置的口令一致"

# 下面两组常量与 app.py 的管理员门禁故意重复定义（见模块 docstring 第三条）。
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_PROXY_HEADERS = ("x-forwarded-for", "x-real-ip", "cf-connecting-ip", "forwarded", "cf-ray")


def access_token() -> str | None:
    """当前生效的访问口令：环境变量优先，否则读文件（600 权限，不入库）。"""
    token = os.environ.get(ACCESS_TOKEN_ENV, "").strip()
    if token:
        return token
    try:
        token = ACCESS_TOKEN_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return token or None


def is_direct_loopback(request: Request) -> bool:
    """是不是"真正的本机直连"（本机运维脚本免口令，公网一律要口令）。

    隧道把公网流量也送到 127.0.0.1，所以只要带任何代理转发头就不算本机。
    这个判别与管理员门禁共用同一套口径（有一套对拍测试兜着）。
    """
    client = request.client.host if request.client else ""
    if client not in _LOOPBACK_HOSTS:
        return False
    return not any(h in request.headers for h in _PROXY_HEADERS)


def require_access_token(request: Request) -> None:
    """FastAPI 依赖：挂到每个 /api/* 路由上，不通过就抛 401。"""
    expected = access_token()
    supplied = request.headers.get(ACCESS_TOKEN_HEADER, "")
    if expected and supplied and hmac.compare_digest(supplied.encode(), expected.encode()):
        return
    if is_direct_loopback(request):
        return
    raise HTTPException(status_code=401, detail=_DENIED_DETAIL)
