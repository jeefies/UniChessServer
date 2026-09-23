"""UniChessServer：统一对局接口的 FastAPI 服务。

设计目标：任意棋类模型只需在 models/{model_name}/engine.py 里暴露一个
GameEngine 类（见 models/__init__.py 顶部注释的契约），即可通过本服务的
/api/* 路由被前端调用，不需要关心 HTTP、会话生命周期、并发、其它模型的
实现细节。

关键设计（对应"同一局需要复用同一棵搜索树"的澄清）：
- 每局对应一个长期存活的 GameEngine 实例，由 session_manager 管理，
  同一局内所有请求都路由到同一个实例，搜索树/cache 可以正确复用推进。
- 进程内最多同时持有 4 个对局实例（滑动窗口 LRU），超过则强制淘汰最旧
  的一个并调用其 cleanup() 释放资源。
- 竞技场（Arena）管理双引擎自动对战，最多同时持有 2 个对局实例。
- 走法合法性判定的权威在服务层（python-chess），不信任模型引擎自行判断。
- /api/models 如实上报每个引擎的状态（available / not_implemented），
  不把必然返回 501 的占位引擎广告成可用。
- /static 只提供服务页面文件：.bak / .before-* / 点文件 / ~ 结尾的备份
  一律 404，避免旧版前端源码随部署目录外泄。

本文件只负责路由和调度，不包含任何模型专属逻辑（编码/解码/网络定义/
搜索算法等一律留在各自的 models/{model_name}/ 目录里）。

启动：
    python app.py --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import argparse
import hmac
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import arena_manager as am
import arena_storage as a_storage
import batch_runner as batch_mod
import models as model_registry
import session_manager as sm

app = FastAPI(title="UniChessServer", description="统一棋类对局接口")

STATIC_DIR = Path(__file__).resolve().parent / "static"


class NewGameRequest(BaseModel):
    model_name: str
    arg_name: str | None = None
    fen: str | None = None
    engine_white: bool = False


class MoveRequest(BaseModel):
    uci: str


class NewArenaRequest(BaseModel):
    white_model: str
    white_arg: str | None = None
    black_model: str
    black_arg: str | None = None
    fen: str | None = None


class BatchStartRequest(BaseModel):
    white_model: str
    white_arg: str | None = None
    black_model: str
    black_arg: str | None = None
    rounds: int = 8
    max_plies: int = 400


# --- 管理员鉴权（批量对弈等耗 GPU 的操作） ---
#
# 口令来源：环境变量 UNICHESS_ADMIN_TOKEN，否则读 ~/.config/unichess/admin_token（不入库）。
# 请求头 X-Admin-Token 与口令一致即放行；否则仅放行"真正的本机直连"。
# 公网流量经隧道到达时对端地址也是 127.0.0.1，因此带任何代理转发头
# （Cloudflare / nginx 会追加）的请求一律不算本机。
ADMIN_TOKEN_FILE = Path.home() / ".config" / "unichess" / "admin_token"
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_PROXY_HEADERS = ("x-forwarded-for", "x-real-ip", "cf-connecting-ip", "forwarded", "cf-ray")


def _admin_token() -> str | None:
    token = os.environ.get("UNICHESS_ADMIN_TOKEN", "").strip()
    if token:
        return token
    try:
        token = ADMIN_TOKEN_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return token or None


def require_admin(request: Request) -> None:
    supplied = request.headers.get("x-admin-token", "")
    expected = _admin_token()
    if supplied and expected and hmac.compare_digest(supplied.encode(), expected.encode()):
        return
    client = request.client.host if request.client else ""
    proxied = any(h in request.headers for h in _PROXY_HEADERS)
    if client in _LOOPBACK_HOSTS and not proxied:
        return
    raise HTTPException(status_code=403, detail="批量对弈仅限管理员启动（需要有效的管理员口令）")


def _session_error_to_http(e: Exception) -> HTTPException:
    if isinstance(e, (sm.SessionNotFoundError, am.ArenaNotFoundError)):
        return HTTPException(status_code=404, detail=str(e))
    if isinstance(e, (sm.InvalidFenError, sm.IllegalMoveError)):
        return HTTPException(status_code=400, detail=str(e))
    return HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.get("/api/models")
def list_models():
    """列出当前已发现模型及真实状态（available / not_implemented / error）。"""
    result = {}
    for name in model_registry.available_models():
        result[name] = model_registry.describe_model(name)
    return {"models": result}


@app.post("/api/new")
def new_game(req: NewGameRequest):
    try:
        session = sm.manager.create(req.model_name, req.arg_name, req.fen, req.engine_white)
    except model_registry.ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (model_registry.ModelNotImplementedError, NotImplementedError) as e:
        raise HTTPException(status_code=501, detail=str(e) or f"{req.model_name} 引擎尚未接入，暂不可对局。")
    except model_registry.ArgPresetNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except sm.SessionError as e:
        # 含 InvalidFenError（400）/ IllegalMoveError（400）等会话层错误
        raise _session_error_to_http(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    return {"session_id": session.session_id, "state": session.state()}


@app.post("/api/games/{session_id}/move")
def make_move(session_id: str, req: MoveRequest):
    try:
        session = sm.manager.get(session_id)
        result = session.human_move(req.uci)
    except sm.SessionError as e:
        raise _session_error_to_http(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    return {"session_id": session_id, "result": result, "state": session.state()}


@app.get("/api/games/{session_id}/state")
def game_state(session_id: str):
    try:
        session = sm.manager.get(session_id)
        state = session.state()
    except sm.SessionError as e:
        raise _session_error_to_http(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return {"session_id": session_id, "state": state}


@app.post("/api/games/{session_id}/undo")
def undo_move(session_id: str):
    try:
        session = sm.manager.get(session_id)
        result = session.undo()
    except sm.SessionError as e:
        raise _session_error_to_http(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    return {"session_id": session_id, "result": result, "state": session.state()}


@app.delete("/api/games/{session_id}")
def close_game(session_id: str):
    try:
        sm.manager.close(session_id)
    except sm.SessionError as e:
        raise _session_error_to_http(e)
    except Exception as e:
        # engine.cleanup() 抛错时也必须返回 JSON detail：
        # 纯文本 500 会让前端 api() 解析不到 detail，只显示通用错误。
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return {"session_id": session_id, "closed": True}


@app.get("/api/games")
def list_games():
    return {"sessions": sm.manager.list_sessions()}


# --- Arena 竞技场路由 ---


@app.get("/arena")
def arena_page():
    arena_html = STATIC_DIR / "arena.html"
    if not arena_html.is_file():
        raise HTTPException(status_code=404, detail="Arena page not found")
    return FileResponse(str(arena_html))


@app.post("/api/arena/new")
def new_arena(req: NewArenaRequest):
    try:
        session = am.manager.create(
            white_model=req.white_model,
            white_arg=req.white_arg,
            black_model=req.black_model,
            black_arg=req.black_arg,
            fen=req.fen,
        )
    except model_registry.ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (model_registry.ModelNotImplementedError, NotImplementedError) as e:
        raise HTTPException(status_code=501, detail=str(e) or "引擎尚未接入，暂不可对局。")
    except model_registry.ArgPresetNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (sm.SessionError, am.ArenaError) as e:
        raise _session_error_to_http(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    return {"arena_id": session.arena_id, "state": session.state()}


@app.post("/api/arena/games/{arena_id}/step")
def arena_step(arena_id: str):
    try:
        session = am.manager.get(arena_id)
        step_result = session.step()
    except (sm.SessionError, am.ArenaError) as e:
        raise _session_error_to_http(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    return {"arena_id": arena_id, "step": step_result, "state": session.state()}


@app.get("/api/arena/games/{arena_id}/state")
def arena_state(arena_id: str):
    try:
        session = am.manager.get(arena_id)
        state = session.state()
    except (sm.SessionError, am.ArenaError) as e:
        raise _session_error_to_http(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    return {"arena_id": arena_id, "state": state}


@app.delete("/api/arena/games/{arena_id}")
def close_arena(arena_id: str):
    try:
        am.manager.close(arena_id)
    except (sm.SessionError, am.ArenaError) as e:
        raise _session_error_to_http(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    return {"arena_id": arena_id, "closed": True}


@app.get("/api/arena/records")
def list_arena_records(limit: int = 50, offset: int = 0, batch_id: str | None = None):
    try:
        records = a_storage.storage.list_records(limit=limit, offset=offset, batch_id=batch_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return {"records": records, "limit": limit, "offset": offset, "batch_id": batch_id}


@app.get("/api/arena/records/{record_id}")
def get_arena_record(record_id: str):
    try:
        record = a_storage.storage.get_record(record_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    if record is None:
        raise HTTPException(status_code=404, detail=f'对弈记录 "{record_id}" 未找到')
    return {"record": record}


# --- 批量对弈路由 ---

@app.get("/arena/batch")
def batch_page():
    batch_html = STATIC_DIR / "arena_batch.html"
    if not batch_html.is_file():
        raise HTTPException(status_code=404, detail="Batch page not found")
    return FileResponse(str(batch_html))


@app.post("/api/arena/batch/start")
def start_batch(req: BatchStartRequest, request: Request):
    require_admin(request)
    try:
        runner = batch_mod.BatchRunner.get()
        config = batch_mod.BatchConfig(
            white_model=req.white_model,
            white_arg=req.white_arg,
            black_model=req.black_model,
            black_arg=req.black_arg,
            rounds=req.rounds,
            max_plies=req.max_plies,
        )
        snapshot = runner.start(config)
    except batch_mod.BatchAlreadyRunningError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except batch_mod.GpuBusyError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except batch_mod.BatchError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except model_registry.ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (model_registry.ModelNotImplementedError, NotImplementedError) as e:
        raise HTTPException(status_code=501, detail=str(e) or "引擎尚未接入，暂不可对局。")
    except model_registry.ArgPresetNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return snapshot


@app.get("/api/arena/batch/state")
def batch_state():
    try:
        runner = batch_mod.BatchRunner.get()
        return runner.snapshot()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.get("/api/health")
def health():
    return {"status": "ok", "models": model_registry.available_models()}


# 静态前端（对局页），迁移自旧 unichess-server。
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.middleware("http")
    async def _static_backup_guard(request: Request, call_next):
        """拦截 /static 下的备份/点文件：.bak、.before-*、.*、*~。"""
        path = request.url.path
        if path.startswith("/static/"):
            name = path[len("/static/"):].rsplit("/", 1)[-1]
            if (
                not name
                or name.startswith(".")
                or ".bak" in name
                or ".before-" in name
                or name.endswith("~")
            ):
                return JSONResponse(status_code=404, content={"detail": "Not Found"})
        return await call_next(request)

    @app.get("/")
    def index():
        return FileResponse(str(STATIC_DIR / "index.html"))


def main():
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
