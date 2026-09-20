# UniChessServer

定位：统一棋类对局接口的 FastAPI 服务。任意棋类模型只需在 `models/{model_name}/engine.py` 暴露 `GameEngine` 类即可被前端调用。进程内最多同时持有 4 个对局实例（滑动窗口 LRU，淘汰最旧并调用 `cleanup()`）。走法合法性判定权威在服务层（python-chess）。

## 目录结构

- `app.py` 路由与调度
- `session_manager.py` 会话生命周期 + 合法性权威 + 调度时序
- `models/__init__.py` 模型发现/加载/预设查表
- `models/{model_name}/engine.py` + `config.json`
- `static/` 前端页面（`index.html` 对局页）
- `tests/test_server.py` unittest 验收套件

## HTTP API（全部返回 JSON）

- `GET /api/health` → `{"status":"ok","models":[...]}`
- `GET /api/models` → `{"models": {name: {"status": "available"|"not_implemented"|"error", "presets": [str], "reason": str}}}`
- `POST /api/new`，body `{"model_name": str, "arg_name": str|null, "fen": str|null, "engine_white": bool}` → `{"session_id": str, "state": {...}}`
- `POST /api/games/{session_id}/move`，body `{"uci": str}` → `{"session_id","result","state"}`
- `GET /api/games/{session_id}/state` → `{"session_id","state"}`
- `POST /api/games/{session_id}/undo` → `{"session_id","result","state"}`
- `DELETE /api/games/{session_id}` → `{"session_id","closed":true}`
- `GET /api/games` → `{"sessions":[{"session_id","model_name","arg_name","engine_white"}]}`
- `GET /` 与 `/static/<file>` 静态页面

错误码：404 模型名非法/不存在或会话不存在（已被淘汰）；400 预设不存在、FEN 非法、走法不合法或轮到引擎；501 引擎未接入（`IMPLEMENTED=False`）；500 其它异常（detail 形如 `"ExceptionType: message"`）。

`state()` 保证字段：`fen`（服务层权威 Board，引擎返回值不覆盖）、`legal_moves`、`is_game_over`、`engine_white`。引擎可额外提供：`last_move`、`san_history`、`eval{win,draw,loss,pov}`、`source`、`engine_ms`、`in_check`。

调度时序：`engine_white=true` 时新局由服务层触发引擎开局第一步；人类每走一步后轮到引擎则自动应答；悔棋回退双方各一步后若轮到引擎会重新触发引擎走子。

## GameEngine 契约

每个 `models/{model_name}/engine.py` 必须暴露 `class GameEngine`，实现六个方法：

- `__init__(**kwargs)`：kwargs 来自 `config.json` 中 `arg_name` 对应条目
- `setup(fen=None)`
- `human_move(uci)`：服务层已校验合法，不触发思考
- `engine_move()`：不接受参数，必须返回含 `"engine_move"` UCI 字符串的 dict，该字段为强制
- `state()`
- `undo()`
- `cleanup()`：释放 GPU/搜索树资源，淘汰时调用

类属性 `IMPLEMENTED=False` + `NOT_IMPLEMENTED_REASON="..."` 表示占位未接入（`/api/models` 如实报告 `not_implemented`，`/api/new` 返回 501）。

`config.json` 格式 `{"<arg_name>": {**kwargs}}`，缺省 `{}`。

安全约束：`model_name` 仅允许字母数字 `. _ -`（拒绝 `/`、`..`、点开头），从请求参数层面杜绝路径穿越——`_load_engine_class` 会 `exec_module` 找到的 `engine.py`，这是唯一的远程攻击面控制点。

`models/<name>` **允许是符号链接**：用于按原样集成外部 git 项目（如 `models/T -> /home/jeefy/UniChess/Transformer`），保持单一事实源、避免拷贝双份代码后漂移。符号链接是运营者信任配置：创建/修改它需要主机文件系统写权限，远程请求无法做到，与"直接往 `models/` 放真实目录"信任级相同。示例：`ln -s /home/jeefy/UniChess/Transformer models/T`。

## 部署与运维（远端 5070 Ti 主机 jeefy@172.16.2.12）

- 工作目录 `/home/jeefy/UniChess/Server`；入口：

```bash
python app.py --host 127.0.0.1 --port 8000
```

仅接受 `--host/--port`。

- conda python：`/home/jeefy/miniconda3/envs/unichess/bin/python`
- systemd 用户级服务：`unichess-server.service`（`Restart=always` + `StartLimitIntervalSec=300/Burst=5`，日志走 journald）
- 查看日志：`journalctl --user -u unichess-server -f`；重启：`systemctl --user restart unichess-server`
- 改 unit 后必须 `systemctl --user daemon-reload`
- 公网暴露：`unichess-tunnel.service`（`ssh -R 8800:localhost:8000` 到中继机 36.151.145.113，`Requires/BindsTo=unichess-server.service`）→ 中继 nginx 反代 `chess.jeefy.top` → 隧道 → 本机 8000

已知踩坑记录（重要）：

1. systemd `append:` 日志路径所在目录不存在时 unit 报 209/STDOUT 起不来（历史上 `/home/jeefy/UniChess/logs` 不存在导致隧道起不来）；
2. 手动测试进程占用 8000/8091 时新进程绑定失败静默退出，请求打到旧代码进程——重启前先 `ss -ltnp` 确认端口；
3. Windows 用户名是 `jeefy`（非 `jeffy`）；
4. 远端无外网、无 pytest，用标准库 unittest runner；
5. Windows 新建文件传到远端会丢 `+x`（本服务用 python 直接跑，不受影响）。

## 模型清单与命名

| 模型名 | 实现 | 状态 |
|---|---|---|
| `T` | 符号链接 → `/home/jeefy/UniChess/Transformer/engine.py`（Transformer 项目真实引擎，跨会话共享权重单例） | available，预设 `max_mcts` |
| `resnet` | `models/resnet/engine.py` 占位桩 | not_implemented，等待 ResNet 项目接入 |

命名约定：模型名保持简短（`T` 而非 `transformer`），避免冗长；接入新项目时用符号链接 + 简短名，例如 `ln -s /home/jeefy/UniChess/ResNet models/R`。曾经的 `models/transformer/` 占位桩已被 `T` 取代并删除（备份见 `~/UniChess/_legacy_server_20260920/placeholder_transformer_20260920/`）。

## 当前状态

`T` 已可对局（Transformer 20M + 开局库/表库/MCTS 800）。`resnet` 仍为占位（`IMPLEMENTED=False`）：`/api/models` 报告 `not_implemented`，`/api/new` 返回 501「暂不可对局」。

## 已归档内容（2026-09-20 审查后移除，勿再 Serve）

- `/board`、`/status` 路由及 `static/board.html`、`static/status.html`：旧训练看板页面，只调用本服务不存在的 `/api/board`、`/api/status*` 端点，加载后必然空白。页面已移至 `~/UniChess/_legacy_server_20260920/`，路由一并移除；`index.html` 顶部原"训练看板"链接已删除。若未来需要看板，需先按本文件 API 契约实现 `/api/status*` 端点。
- `static/` 下的 `*.bak-*`、`*.before-*` 备份：旧版前端源码，曾可被公网下载，已移出静态目录；`/static` 另有中间件拦截 `.bak`/`.before-*`/点文件/`~` 结尾文件（404）。
- `_legacy_app_reference/`（旧 app.py、旧 unit、install_services.sh 等）：已移至 `~/UniChess/_legacy_server_20260920/`，避免重跑安装脚本覆盖新 unit。

## 测试

```bash
/home/jeefy/miniconda3/envs/unichess/bin/python -m unittest discover -s tests -v
```

在 Server 目录下运行。
