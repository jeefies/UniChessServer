"""Stockfish 远端分析服务 API 路由（/sf/v1/*）。

实现需求文档 Server/wants.md 的接口定义：
- GET /sf/v1/health       查询就绪状态、引擎版本、支持的参数和上限
- POST /sf/v1/evaluate     返回局面评价及候选走法
- POST /sf/v1/analyze-move 核心单步深度分析与同深度比对
- GET /sf/v1/openapi.json  专供 AI / Agent 使用的 OpenAPI 3.1 规格描述
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

OPENAPI_DESCRIPTION = """## Stockfish 国际象棋无状态深度分析服务（API 指南与约束）

面向移动端与 AI Agent 的高性能国际象棋分析服务。所有接口均为纯无状态设计，服务端不存储对局记录或用户数据。

### 核心物理量与度量口径（AI 调用必须遵循）：
1. **统一行棋方视角（Root Perspective）**：
   - 所有的分数值（score）与胜平负概率（wdl）均恒从**当前请求根局面的走子方（Side to Move）**视角给出：
   - 正值（+）表示当前走子方优势，负值（-）表示劣势。
2. **单位度量**：
   - `cp`：厘兵（centipawns），100 cp = 1 个兵。
   - `mate`：将杀步数，正数为我方将杀对手，负数为我方被对手将杀。
   - `wdl`：胜平负千分比，win + draw + loss = 1000。
3. **局面重构约束（Threefold Repetition）**：
   - 请求必须携带 `initialFen` 加上完整的历史 UCI 走法列表 `moves`，服务端从初始局面推演，以准确识别**三次重复局面**和五十步和棋规则。
4. **同深度对拍（Fair Comparison）**：
   - `/sf/v1/analyze-move` 会双路并行对根局面搜索最佳候选与实战走法 `playedMove`；
   - 最终比对保证在两者的**共同完成深度（commonDepth）**下对齐，绝不跨深度比较。
   - `diffCp = played.score.value - best.score.value`：通常为 <= 0，表示走这一步相较最佳着法的厘兵损失。

### 分析档位（Profiles）：
- **`lightning`（极速档，推荐 AI 对话场景使用）**：纯基于目标深度 22 层驱动，**严格搜满 22 层即停，不设时间预算限制**（仅保留 60s 异常安全兜底），单候选 + 实战走法双路并行推演。
- **`fast`（快档）**：纯基于目标深度 22 层驱动，严格搜满 22 层即停，不限时间预算，双候选 MultiPV=2。
- **`standard`（标准档）**：时间预算 4.0 秒，深度上限 128，双候选。
- **`deep`（深度档）**：纯基于时间预算驱动（4.0 秒），不限制层数（深度上限 128），进行充分深入推演（通常可达 24~30+ 层）。
- **`ultra`（超深档）**：纯基于时间预算驱动（10.0 秒），不限制层数（深度上限 128），用于关键着法极限推演。

### 开局定式与残局库扩展能力：
1. **开局定式识别（`opening`）**：
   - 服务端内置完整 ECO 开局前缀树，自动返回当前局面所处的定式名称（如 `name: "西班牙开局 柏林防御"`）、定式标记（`theory: true/false`）及定式匹配半步序。
2. **Syzygy 3-4-5 残局库（Tablebase）**：
   - 服务端常驻挂载 Syzygy 3-4-5 残局表，3-5 子局面可瞬间命中查表，深度直接穿透至 72~128 层，并在 `stats.tbhits` 返回残局库探测命中次数。

### 辅助棋力评语分类标准（供 AI 解说棋步）：
- `diffCp == 0` 或 `played == best`：🌟 最佳着法 (Best Move)
- `-15 <= diffCp < 0`：✅ 优秀着法 (Excellent)
- `-40 <= diffCp < -15`：👍 良好着法 (Good)
- `-100 <= diffCp < -40`：⚠️ 轻微缓着 (Inaccuracy)
- `-250 <= diffCp < -100`：❓ 疑问着 / 错误 (Mistake)
- `diffCp < -250`：❌ 严重大漏 (Blunder)
"""

# 受门禁保护的计算接口
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

# 公开给 AI / Agent / GPT Actions 读取 Schema 的公开端点（无需口令）
public_sf_router = APIRouter(
    prefix="/sf/v1",
    tags=["Stockfish Spec"],
)

public_api_router = APIRouter(
    prefix="/api/sf/v1",
    tags=["Stockfish Spec"],
)


# --- 请求模型 ---

class PositionInput(BaseModel):
    initialFen: str | None = Field(
        None, description="起始 FEN，缺省为国际象棋标准起始局面。"
    )
    moves: list[str] = Field(
        default_factory=list,
        description="从 initialFen 到当前分析局面之前的全部 UCI 历史走法列表（如 ['e2e4', 'e7e5']）。",
    )


class MoveAnalysisLimits(BaseModel):
    depth: int | None = Field(
        None, description="搜索深度上限；若省略则继承 profile（如 lightning 默认为 22，其余默认为 128）。"
    )
    maxTimeMs: int | None = Field(
        None, description="总时间预算（毫秒）；若省略则继承 profile（如 deep 默认为 4000，lightning 不限时间预算）。"
    )


class AnalyzeMoveRequest(BaseModel):
    requestId: str | None = Field(None, description="请求唯一标识，便于跟踪或取消。")
    position: PositionInput = Field(..., description="待分析着法落子前的局面历史。")
    playedMove: str = Field(..., description="实战走出的单步 UCI 着法（如 f1c4）。")
    profile: str | None = Field("deep", description="预设档位（lightning / fast / standard / deep / ultra）。")
    limits: MoveAnalysisLimits | None = Field(
        None, description="搜索预算限制；若省略则继承 profile 或标准深度 128。"
    )
    multiPv: int | None = Field(
        None, ge=1, le=10, description="首阶段多候选线数（若省略则继承 profile：lightning 默认为 1，其余默认为 2）。"
    )
    maxPvPlies: int | None = Field(
        DEFAULT_MAX_PV_PLIES, ge=1, le=64, description="返回 PV 的最大步数截断（默认 12）。"
    )


class EvaluateRequest(BaseModel):
    requestId: str | None = Field(None, description="请求唯一标识。")
    position: PositionInput = Field(..., description="待评估局面。")
    profile: str | None = Field("standard", description="预设档位（lightning / fast / standard / deep / ultra）。")
    limits: MoveAnalysisLimits | None = Field(None, description="搜索限制。")
    multiPv: int | None = Field(None, ge=1, le=10, description="候选线数（若省略则继承 profile）。")
    maxPvPlies: int | None = Field(DEFAULT_MAX_PV_PLIES, ge=1, le=64, description="PV 步数。")


# --- 响应模型 ---

class ScoreDetail(BaseModel):
    type: str = Field(..., description="评分类型：'cp'（厘兵 centipawns）或 'mate'（将杀步数）")
    value: int = Field(..., description="评分数值。统一为请求根局面当前行棋方视角：正值有利，负值不利。")


class WdlDetail(BaseModel):
    win: int = Field(..., description="根局面当前行棋方获胜概率千分比")
    draw: int = Field(..., description="和棋概率千分比")
    loss: int = Field(..., description="根局面当前行棋方失败概率千分比")


class EvalItem(BaseModel):
    move: str | None = Field(None, description="UCI 格式着法，如 f1c4")
    depth: int = Field(..., description="实际完成的计算深度")
    seldepth: int | None = Field(None, description="最大选择性搜索深度")
    score: ScoreDetail | None = Field(None, description="行棋方视角的局面评分")
    wdl: WdlDetail | None = Field(None, description="胜平负千分比（总和 1000）")
    pv: list[str] = Field(default_factory=list, description="后续主变例走法列表（UCI 格式）")


class ComparisonResult(BaseModel):
    canCompare: bool = Field(..., description="是否能在同深度下进行有效对比")
    commonDepth: int = Field(..., description="共同完成深度（同深度比对基准）")
    diffCp: int | None = Field(
        None,
        description="厘兵亏损值：played.score - best.score。负值表示相较最佳着法的损失（厘兵，100 cp = 1 兵）。",
    )
    diffWdlLoss: int | None = Field(
        None, description="失败率增加值：played.loss - best.loss（千分比）"
    )


class OpeningInfo(BaseModel):
    name: str = Field(..., description="开局名称或体系")
    theory: bool = Field(..., description="是否属于理论定式着法")
    ply: int = Field(..., description="理论定式匹配步数")


class EngineStats(BaseModel):
    nodes: int | None = Field(None, description="搜索总节点数")
    nps: int | None = Field(None, description="每秒搜索节点数")
    hashfull: int | None = Field(None, description="置换表千分比占用")
    tbhits: int | None = Field(None, description="Syzygy 残局库命中次数")
    elapsedMs: int | None = Field(None, description="总耗时（毫秒）")
    cached: bool = Field(False, description="是否命中服务端 LRU 计算缓存")


class AnalyzeMoveResponse(BaseModel):
    requestId: str | None = Field(None, description="请求标识")
    best: EvalItem = Field(..., description="引擎分析得出的最佳走法")
    played: EvalItem | None = Field(None, description="实战走出的走法评估")
    second: EvalItem | None = Field(None, description="第二候选走法，用于评估最佳走法是否明显突出")
    previousBest: EvalItem | None = Field(
        None, description="上一完成深度（depth-1）的最佳走法，用于评估评分稳定性"
    )
    comparison: ComparisonResult = Field(..., description="同深度对比结果")
    opening: OpeningInfo | None = Field(None, description="开局理论名称与体系识别信息")
    engine: dict[str, Any] = Field(..., description="引擎版本与运行配置")
    stats: EngineStats = Field(..., description="耗时与搜索统计")


class EvaluateResponse(BaseModel):
    requestId: str | None = Field(None, description="请求标识")
    completedDepth: int = Field(..., description="实际完成深度")
    best: EvalItem | None = Field(None, description="最佳走法")
    candidates: list[EvalItem] = Field(default_factory=list, description="各候选走法评分排序列表")
    opening: OpeningInfo | None = Field(None, description="开局理论名称与体系识别信息")
    engine: dict[str, Any] = Field(..., description="引擎版本与运行配置")
    stats: EngineStats = Field(..., description="耗时与搜索统计")


class ReviewGameRequest(BaseModel):
    requestId: str | None = Field(None, description="请求唯一标识，便于跟踪或取消。")
    initialFen: str | None = Field(None, description="起始 FEN，缺省为国际象棋标准起始局面。")
    moves: list[str] = Field(
        default_factory=list,
        description="从起始局面开始的全部 UCI 历史走法序列（如 ['e2e4', 'e7e5', ...]）。",
    )
    pgn: str | None = Field(
        None, description="可选 PGN 文本格式棋谱。若传入且 moves 为空，则自动解析其中的 moves 与 FEN。"
    )
    profile: str | None = Field(
        "lightning", description="预设档位（默认 lightning：目标深度 22，极速并行复盘）。"
    )
    concurrency: int | None = Field(
        None, ge=1, le=8, description="并发分析工作引擎进程数（若省略则自适应，Core Ultra 20核默认为4）。"
    )


class GameMoveItem(BaseModel):
    ply: int = Field(..., description="半步序号（1-indexed）")
    move: str = Field(..., description="实战 UCI 走法，如 e2e4")
    san: str | None = Field(None, description="标准代数记谱法（SAN），如 e4, Nf3")
    turn: str = Field(..., description="行棋方 ('white' 或 'black')")
    bestMove: str | None = Field(None, description="引擎计算的最佳着法")
    diffCp: int | None = Field(
        None, description="厘兵亏损值：0为最佳，负数表示相较最佳着法的损失（厘兵，100 cp = 1 兵）"
    )
    diffWdlLoss: int | None = Field(None, description="失败率增加值千分比")
    judgment: str = Field(
        ..., description="棋步评语：best(🌟最佳), excellent(👍优秀), good(🆗良好), inaccuracy(⚠️疑问手), mistake(❓失着), blunder(❌败着)"
    )
    accuracy: float = Field(..., description="单步准确率百分比（0.0 ~ 100.0）")
    depth: int = Field(..., description="实际共同完成深度")
    score: ScoreDetail | None = Field(None, description="行棋方视角的局面评分")
    wdl: WdlDetail | None = Field(None, description="胜平负千分比概率")
    opening: OpeningInfo | None = Field(None, description="开局定式识别")
    cached: bool = Field(False, description="是否命中持久化磁盘缓存")
    elapsedMs: int = Field(..., description="单步评估计算耗时（毫秒）")


class GameReviewSummary(BaseModel):
    whiteAccuracy: float = Field(..., description="白方全局综合准确率（0.0 ~ 100.0）")
    blackAccuracy: float = Field(..., description="黑方全局综合准确率（0.0 ~ 100.0）")
    whiteAcpl: float = Field(..., description="白方平均厘兵损失 (Average Centipawn Loss)")
    blackAcpl: float = Field(..., description="黑方平均厘兵损失 (Average Centipawn Loss)")
    whiteJudgments: dict[str, int] = Field(default_factory=dict, description="白方各类着法统计计数")
    blackJudgments: dict[str, int] = Field(default_factory=dict, description="黑方各类着法统计计数")


class ReviewGameResponse(BaseModel):
    requestId: str | None = Field(None, description="请求标识")
    totalPlies: int = Field(..., description="对局总半步数")
    analyzedPlies: int = Field(..., description="实际分析半步数")
    cacheHits: int = Field(..., description="命中缓存的步数")
    elapsedMs: int = Field(..., description="全盘并发复盘总耗时（毫秒）")
    effectivePliesPerSecond: float = Field(..., description="整盘实际复盘吞吐量（步/秒）")
    summary: GameReviewSummary = Field(..., description="全盘统计摘要（准确率、ACPL、招法统计）")
    moves: list[GameMoveItem] = Field(..., description="逐步分析评测详情列表")


# --- 端点定义 ---

@public_sf_router.get("/openapi.json", include_in_schema=False)
@public_api_router.get("/openapi.json", include_in_schema=False)
def get_sf_openapi() -> dict[str, Any]:
    """生成并返回专供 AI / Agent 使用的 Stockfish 分析 API OpenAPI 3.1 描述。"""
    from fastapi.openapi.utils import get_openapi

    schema = get_openapi(
        title="Stockfish Chess Analysis Service API",
        version="1.0.0",
        description=OPENAPI_DESCRIPTION,
        routes=router.routes,
    )
    schema["servers"] = [
        {"url": "https://chess.jeefy.top", "description": "UniChess 线上生产服务器"},
        {"url": "http://127.0.0.1:8000", "description": "本地开发 / 直连服务"},
    ]
    schema.setdefault("components", {})["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "description": "通过 Authorization: Bearer <token> 鉴权",
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-Access-Token",
            "description": "通过 X-Access-Token: <token> 鉴权",
        },
    }
    schema["security"] = [{"BearerAuth": []}, {"ApiKeyAuth": []}]
    return schema


@router.get("/health")
@api_router.get("/health")
def health() -> dict[str, Any]:
    """查询分析引擎就绪状态、配置参数与上限。"""
    analyzer = get_analyzer()
    return analyzer.get_health_info()


@router.post("/release")
@api_router.post("/release")
def release_engines() -> dict[str, Any]:
    """主动释放 Stockfish 引擎常驻内存（关闭子进程并归还约 1.5GB 内存）。"""
    analyzer = get_analyzer()
    freed = analyzer.release_idle_engines(force=True)
    return {
        "status": "ok",
        "released": freed,
        "message": "Stockfish 引擎与内存已成功释放" if freed else "当前无运行中的引擎需释放",
    }


@router.post("/analyze-move", response_model=AnalyzeMoveResponse)
@api_router.post("/analyze-move", response_model=AnalyzeMoveResponse)
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


@router.post("/evaluate", response_model=EvaluateResponse)
@api_router.post("/evaluate", response_model=EvaluateResponse)
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


@router.post("/review", response_model=ReviewGameResponse)
@router.post("/analyze-game", response_model=ReviewGameResponse)
@api_router.post("/review", response_model=ReviewGameResponse)
@api_router.post("/analyze-game", response_model=ReviewGameResponse)
async def review_game(req: ReviewGameRequest, request: Request) -> dict[str, Any]:
    """对整局人类对局进行全盘高并发多核并行复盘，返回每步同深度对比、评语与全局统计。"""
    analyzer = get_analyzer()

    moves = list(req.moves)
    initial_fen = req.initialFen
    if req.pgn and not moves:
        import io
        import chess.pgn
        try:
            game = chess.pgn.read_game(io.StringIO(req.pgn))
            if game is None:
                raise ValueError("无法解析提供的 PGN 棋谱文本")
            moves = [m.uci() for m in game.mainline_moves()]
            initial_fen = game.headers.get("FEN") or initial_fen
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"PGN 解析失败: {exc}") from exc

    if not moves:
        raise HTTPException(status_code=400, detail="必须提供至少包含一步走法的 moves 列表或有效的 pgn 文本")

    cancel_event = threading.Event()
    loop = asyncio.get_running_loop()

    def run_worker():
        return analyzer.review_game(
            initial_fen=initial_fen,
            moves=moves,
            profile=req.profile,
            concurrency=req.concurrency,
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
