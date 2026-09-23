"""UniChessServer 模型注册与加载。

每个模型在 models/{model_name}/ 下提供：

    engine.py    模块级暴露 class GameEngine（见下方契约）
    config.json  可选。格式 {"<arg_name>": {**kwargs, ...}}，
                 供 <model_name>/<arg_name> 这种预设选择方式查表。

本模块只负责“发现模型 + 加载 engine.py 模块 + 按 arg_name 查 kwargs”，
不负责会话生命周期管理（那部分在 session_manager.py）。

GameEngine 契约（每个 engine.py 必须实现，缺失方法视为未实现该模型）：

    class GameEngine:
        # 可选类属性：占位/未接入引擎置 False，并给出 NOT_IMPLEMENTED_REASON。
        # /api/models 据此如实上报状态，而不是把必然 501 的引擎广告成可用。
        IMPLEMENTED: bool = True
        NOT_IMPLEMENTED_REASON: str = ""
        # 可选：UniChessKit 原生 PlayerFactory（"包.模块:函数"，模块须在模型目录内，
        # 以 preset=<arg_name> 调用）。声明后批量对弈 / 观战走 kit 原生路径（跨局攒批）；
        # 不声明则由 kit 把本类的六方法包装成 Player（jobs.py）。
        KIT_FACTORY: str = "unichess_r.kit_adapter:make_player_factory"

        def __init__(self, **kwargs): ...
        def setup(self, fen: str | None = None) -> dict: ...
        def human_move(self, uci: str) -> dict: ...
        def engine_move(self) -> dict: ...
        def state(self) -> dict: ...
        def undo(self) -> dict: ...
        def cleanup(self) -> None: ...

engine_move() 的返回值中 "engine_move"（UCI 字符串）是**强制**字段，不再是建议：
服务层依据它推进权威 Board（见 session_manager._push_engine_result），缺失即报错，
绝不允许静默错位。

human_move(uci) 与 engine_move() 是两个对称的原子操作，取代早期设计里
耦合了"人走一步+引擎答一步"的单一 move(uci) 方法：

    - human_move(uci): 只应用服务层已校验合法的人类走法，不触发引擎思考。
    - engine_move(): 不接受走法参数，引擎按当前局面自行决定并走一步，
      返回值建议至少包含 {"engine_move": "e2e4", ...}。

这样"引擎执白时新局需要先走一步"之类的调度时序完全由服务层
（session_manager.py）决定，GameEngine 不需要知道自己执哪方、
是否自由模式等对弈规则层面的概念，只管"应用一步"和"想一步"。

cleanup() 是强制要求实现的方法：会话滑动窗口淘汰最旧实例时会调用它来
释放资源（GPU 显存、搜索树/cache 等）。未实现 cleanup 的模型视为不合格，
加载时会报错，而不是静默忽略。
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import threading
from pathlib import Path
from typing import Any

MODELS_DIR = Path(__file__).resolve().parent

_REQUIRED_METHODS = ('setup', 'human_move', 'engine_move', 'state', 'undo', 'cleanup')

# model_name 只允许安全的目录名字符：杜绝 "../" 之类的路径穿越
# （_load_engine_class 会 exec_module 任意 engine.py，穿越即可执行任意代码）。
_MODEL_NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')

_engine_class_cache: dict[str, type] = {}
_config_cache: dict[str, dict[str, dict[str, Any]]] = {}
_cache_lock = threading.Lock()


class ModelNotFoundError(Exception):
    pass


class ModelNotImplementedError(Exception):
    """models/{name}/engine.py 存在，但未按契约实现 GameEngine。"""


class ArgPresetNotFoundError(Exception):
    """models/{name}/config.json 中不存在指定的 arg_name。"""


def available_models() -> list[str]:
    """列出 models/ 下所有含 engine.py 的模型目录名。

    models/<name> 允许是符号链接：这是运营者信任配置，用于按原样集成
    UniChess/ResNet、UniChess/Transformer 等外部 git 项目（单一事实源，
    避免拷贝双份代码后漂移）。创建符号链接需要主机文件系统写权限，
    远程请求无法做到，因此不构成攻击面（与"直接往 models/ 放真实目录"
    信任级相同）。
    """
    return sorted(
        p.name for p in MODELS_DIR.iterdir()
        if p.is_dir() and not p.name.startswith('_') and (p / 'engine.py').is_file()
    )


def _validate_model_name(model_name: str) -> None:
    """校验 model_name，拒绝路径穿越（CRITICAL 修复的防护点）。

    威胁模型：model_name 来自未认证的公网请求，_load_engine_class 会
    exec_module 找到的 engine.py，因此必须保证它只能指向 models/<name>。
    控制点是**名字白名单**：拒绝 "/"、".."、点开头等一切分隔/逃逸字符。

    models/<name> 本身允许是符号链接（集成外部项目的既定方式，见
    available_models 注释）：链接由运营者创建，需要文件系统写权限，
    远程请求无法制造或修改，与"直接放真实目录"信任级相同。
    """
    if not isinstance(model_name, str) or not _MODEL_NAME_RE.match(model_name):
        raise ModelNotFoundError(
            f'非法的模型名 "{model_name}"：仅允许字母/数字/./_/-，且不能以 . 开头。'
            f'可用模型: {available_models()}'
        )
    # 双保险：正则已不可能产出 ".." 路径分量，这里防未来放宽正则时回归
    if '..' in pathlib.PurePosixPath(model_name).parts:
        raise ModelNotFoundError(
            f'非法的模型名 "{model_name}"：不允许包含 ".." 路径分量。'
            f'可用模型: {available_models()}'
        )


def _load_engine_class(model_name: str) -> type:
    with _cache_lock:
        if model_name in _engine_class_cache:
            return _engine_class_cache[model_name]

    _validate_model_name(model_name)

    engine_path = MODELS_DIR / model_name / 'engine.py'
    if not engine_path.is_file():
        raise ModelNotFoundError(
            f'models/{model_name}/engine.py 不存在。可用模型: {available_models()}'
        )

    spec = importlib.util.spec_from_file_location(
        f'unichess_server_models.{model_name}.engine', engine_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    engine_cls = getattr(module, 'GameEngine', None)
    if engine_cls is None:
        raise ModelNotImplementedError(
            f'models/{model_name}/engine.py 未定义 GameEngine 类'
        )

    missing = [m for m in _REQUIRED_METHODS if not callable(getattr(engine_cls, m, None))]
    if missing:
        raise ModelNotImplementedError(
            f'models/{model_name}/engine.py 的 GameEngine 缺少方法: {missing}'
        )

    with _cache_lock:
        _engine_class_cache[model_name] = engine_cls
    return engine_cls


def _load_config(model_name: str) -> dict[str, dict[str, Any]]:
    if model_name in _config_cache:
        return _config_cache[model_name]

    config_path = MODELS_DIR / model_name / 'config.json'
    if not config_path.is_file():
        config = {}
    else:
        with config_path.open('r', encoding='utf-8') as f:
            config = json.load(f)

    _config_cache[model_name] = config
    return config


def list_presets(model_name: str) -> list[str]:
    """列出 models/{model_name}/config.json 中的可用预设名（arg_name）。"""
    return sorted(_load_config(model_name).keys())


def is_implemented(model_name: str) -> bool:
    """引擎是否已按契约接入（类属性 IMPLEMENTED，默认 True）。"""
    engine_cls = _load_engine_class(model_name)
    return bool(getattr(engine_cls, 'IMPLEMENTED', True))


def not_implemented_reason(model_name: str) -> str:
    engine_cls = _load_engine_class(model_name)
    return str(getattr(engine_cls, 'NOT_IMPLEMENTED_REASON', '') or '')


def describe_model(model_name: str) -> dict[str, Any]:
    """供 /api/models 使用：如实上报状态，不把必然 501 的引擎广告成 available。

    返回 {"status": "available"|"not_implemented", "presets": [...], "reason": str}。
    加载/校验失败时 status 为 "error" 并附 message（不影响其它模型列出）。
    """
    try:
        engine_cls = _load_engine_class(model_name)
    except ModelNotFoundError:
        return {'status': 'not_found', 'presets': [], 'reason': ''}
    except Exception as e:
        return {'status': 'error', 'presets': [], 'reason': f'{type(e).__name__}: {e}'}

    try:
        presets = list_presets(model_name)
    except Exception:
        presets = []

    if not bool(getattr(engine_cls, 'IMPLEMENTED', True)):
        return {
            'status': 'not_implemented',
            'presets': presets,
            'reason': str(getattr(engine_cls, 'NOT_IMPLEMENTED_REASON', '') or ''),
        }
    return {'status': 'available', 'presets': presets, 'reason': ''}


def resolve_kwargs(model_name: str, arg_name: str | None) -> dict[str, Any]:
    """按 <model_name>/<arg_name> 查 config.json 得到 kwargs。

    arg_name 为 None 或空字符串时返回空 kwargs（使用 GameEngine 默认参数）。
    """
    if not arg_name:
        return {}

    config = _load_config(model_name)
    if arg_name not in config:
        raise ArgPresetNotFoundError(
            f'models/{model_name}/config.json 中不存在预设 "{arg_name}"。'
            f'可用预设: {sorted(config.keys())}'
        )
    return dict(config[arg_name])


def engine_class(model_name: str) -> type:
    """按契约校验并返回 GameEngine 类（不实例化；供 jobs 解析 KIT_FACTORY 等类属性）。"""
    engine_cls = _load_engine_class(model_name)
    if not bool(getattr(engine_cls, 'IMPLEMENTED', True)):
        raise ModelNotImplementedError(
            str(getattr(engine_cls, 'NOT_IMPLEMENTED_REASON', ''))
            or f'{model_name} 引擎尚未接入，暂不可对局。'
        )
    return engine_cls


def create_engine(model_name: str, arg_name: str | None = None):
    """创建一个 GameEngine 实例：加载 models/{model_name}/engine.py，
    按 arg_name 查 config.json 得到 kwargs，实例化并返回。"""
    engine_cls = _load_engine_class(model_name)
    if not bool(getattr(engine_cls, 'IMPLEMENTED', True)):
        raise ModelNotImplementedError(
            str(getattr(engine_cls, 'NOT_IMPLEMENTED_REASON', ''))
            or f'{model_name} 引擎尚未接入，暂不可对局。'
        )
    kwargs = resolve_kwargs(model_name, arg_name)
    return engine_cls(**kwargs)
