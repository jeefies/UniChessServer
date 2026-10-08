"""Stockfish 远端分析服务 API 路由（/sf/v1/*）。

实现需求文档 Server/wants.md 的接口定义：
- GET /sf/v1/health       查询就绪状态、引擎版本、支持的参数和上限
- POST /sf/v1/evaluate     返回局面评价及候选走法
- POST /sf/v1/analyze-move 核心单步深度分析与同深度比对
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

import access_auth
from sf_analyzer import (
    DEFAULT_MAX_PV_PLIES,
    DEFAULT_MAX_TIME_MS,
    DEFAULT_MULTI_PV,
    STANDARD_DEPTH,
    get_analyzer,
)

router = APIRouter(
    prefix="/sf/v1",
    tags=["Stockfish Analysis"],
    dependencies=[Depends(access_auth.require_access_token)],
)

# 兼容路由（同时支持挂在 /api/sf/v1 下）
api_router = APIRouter(
    prefix="/api/sf/v1",
    tags=["Stockfish Analysis"],
    dependencies=[Depends(access_auth.require_access_token)],
)


class PositionInput(BaseModel):
    initialFen: str | None = Field(
        None, description="起始 FEN，缺省为国际象棋标准起始局面。"
    )
    moves: list[str] = Field(
        default_factory=list,
        description="从 initialFen 到当前分析局面之前的全部 UCI 历史走法列表。",
    )


class MoveAnalysisLimits(BaseModel):
    depth: int | None = Field(
        STANDARD_DEPTH, description="搜索深度上限，标准深度固定为 128。"
    )
    maxTimeMs: int | None = Field(
        DEFAULT_MAX_TIME_MS, description="总时间预算（毫秒），覆盖全部搜索步骤。"
    )


class AnalyzeMoveRequest(BaseModel):
    requestId: str | None = Field(None, description="请求唯一标识，便于跟踪或取消。")
    position: PositionInput = Field(..., description="待分析着法落子前的局面历史。")
    playedMove: str = Field(..., description="实战走出的单步 UCI 着法（如 f1c4）。")
    profile: str | None = Field("deep", description="预设档位（fast / standard / deep / ultra）。")
    limits: MoveAnalysisLimits | None = Field(
        None, description="搜索预算限制；若省略则继承 profile 或标准深度 128。"
    )
    multiPv: int | None = Field(
        DEFAULT_MULTI_PV, ge=1, le=10, description="首阶段多候选线数（默认 2）。"
    )
    maxPvPlies: int | None = Field(
        DEFAULT_MAX_PV_PLIES, ge=1, le=64, description="返回 PV 的最大步数截断（默认 12）。"
    )


class EvaluateRequest(BaseModel):
    requestId: str | None = Field(None, description="请求唯一标识。")
    position: PositionInput = Field(..., description="待评估局面。")
    profile: str | None = Field("standard", description="预设档位。")
    limits: MoveAnalysisLimits | None = Field(None, description="搜索限制。")
    multiPv: int | None = Field(DEFAULT_MULTI_PV, ge=1, le=10, description="候选线数。")
    maxPvPlies: int | None = Field(DEFAULT_MAX_PV_PLIES, ge=1, le=64, description="PV 步数。")


@router.get("/health")
@api_router.get("/health")
def health() -> dict[str, Any]:
    """查询分析引擎就绪状态、配置参数与上限。"""
    analyzer = get_analyzer()
    return analyzer.get_health_info()


@router.post("/analyze-move")
@api_router.post("/analyze-move")
async def analyze_move(req: AnalyzeMoveRequest, request: Request) -> dict[str, Any]:
    """一次完成最佳走法、第二候选与实战走法的同深度对拍。"""
    analyzer = get_analyzer()

    depth = req.limits.depth if req.limits else None
    max_time_ms = req.limits.maxTimeMs if req.limits else None

    cancel_event = threading.Event()

    loop = asyncio.get_running_loop()

    def run_worker():
        return analyzer.analyze_move(
            initial_fen=req.position.initialFen,
            moves=req.position.moves,
            played_move=req.playedMove,
            profile=req.profile,
            depth=depth,
            max_time_ms=max_time_ms,
            multi_pv=req.multiPv,
            max_pv_plies=req.maxPvPlies,
            request_id=req.requestId,
            cancel_event=cancel_event,
        )

    task = loop.run_in_executor(None, run_worker)

    try:
        # 支持监听连接断开或取消
        while not task.done():
            if await request.is_disconnected():
                cancel_event.set()
                analyzer.cancel(req.requestId)
                break
            await asyncio.sleep(0.05)

        return await task
    except asyncio.CancelledError:
        cancel_event.set()
        analyzer.cancel(req.requestId)
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@router.post("/evaluate")
@api_router.post("/evaluate")
async def evaluate(req: EvaluateRequest, request: Request) -> dict[str, Any]:
    """返回任意局面的综合评价与候选走法列表。"""
    analyzer = get_analyzer()

    depth = req.limits.depth if req.limits else None
    max_time_ms = req.limits.maxTimeMs if req.limits else None

    cancel_event = threading.Event()
    loop = asyncio.get_running_loop()

    def run_worker():
        return analyzer.evaluate(
            initial_fen=req.position.initialFen,
            moves=req.position.moves,
            profile=req.profile,
            depth=depth,
            max_time_ms=max_time_ms,
            multi_pv=req.multiPv,
            max_pv_plies=req.maxPvPlies,
            request_id=req.requestId,
            cancel_event=cancel_event,
        )

    task = loop.run_in_executor(None, run_worker)

    try:
        while not task.done():
            if await request.is_disconnected():
                cancel_event.set()
                analyzer.cancel(req.requestId)
                break
            await asyncio.sleep(0.05)

        return await task
    except asyncio.CancelledError:
        cancel_event.set()
        analyzer.cancel(req.requestId)
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
