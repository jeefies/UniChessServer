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
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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


def _session_error_to_http(e: Exception) -> HTTPException:
    if isinstance(e, sm.SessionNotFoundError):
        return HTTPException(status_code=404, detail=str(e))
    if isinstance(e, sm.InvalidFenError):
        return HTTPException(status_code=400, detail=str(e))
    if isinstance(e, sm.IllegalMoveError):
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


@app.get("/api/health")
def health():
    return {"status": "ok", "models": model_registry.available_models()}


# 静态前端（对局页），迁移自旧 unichess-server。
# 注意：旧的 /board 与 /status 训练看板页面依赖的 /api/board、/api/status*
# 端点在本服务中不存在，页面已归档（不再服务），路由一并移除——
# 继续服务一个数据端点全部 404 的死页面，只会得到"刷新失败"的空白看板。
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.middleware("http")
    async def _static_backup_guard(request: Request, call_next):
        """拦截 /static 下的备份/点文件：.bak、.before-*、.*、*~。

        StaticFiles 会服务目录内一切文件，历史备份（如
        board.html.bak-rebuild-20260917）等于公开旧版前端源码。
        """
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
