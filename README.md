# UniChessServer

定位：统一棋类对局接口的 FastAPI 服务。任意棋类模型只需在 `models/{model_name}/engine.py` 暴露 `GameEngine` 类即可被前端调用。进程内最多同时持有 4 个对局实例（滑动窗口 LRU，淘汰最旧并调用 `cleanup()`）。走法合法性判定权威在服务层（python-chess）。

## 目录结构

- `app.py` 路由与调度
- `session_manager.py` 会话生命周期 + 合法性权威 + 调度时序（终局用 UniChessKit 的 `classify`）
- `jobs.py` 批量对弈与网页观战：提交 UniChessKit 后台 job（独立进程组 + GPU 租约），读 job 目录同步进度
- `kit_env.py` 定位 UniChessKit（默认 `../Kit`，可用 `UNICHESS_KIT_ROOT` 覆盖；kit 不 pip 安装）
- `arena_storage.py` 对弈记录 / 批次 SQLite（`data/arena/arena_history.db`）
- `models/__init__.py` 模型发现/加载/预设查表
- `models/{model_name}/engine.py` + `config.json`
- `static/` 前端页面（`index.html` 对局页）
- `tests/test_server.py` 对局接口验收；`tests/test_jobs.py` 批量对弈 / 观战 / 存储 / 管理员鉴权

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

## 观战与批量对弈（UniChessKit job）

两者都不在服务进程里跑引擎，而是写 `data/jobs/<id>/job.json` 后启动 `python -m unichess_kit.jobs`
（独立进程组，cwd 为 job 目录，启动前按 `gpu_mib` 申请 GPU 租约，显存不足直接 `gpu_busy` 退出）。
Server 只读 job 目录（`status.json` / `live.json` / `results.jsonl`），后台 Monitor 线程每 2 秒把进度入库。

- 引擎解析：`GameEngine` 声明类属性 `KIT_FACTORY = "包.模块:函数"`（模块须在模型目录内，以 `preset=<arg>` 调用）
  时走 kit 原生 Player（单进程 8 局并发、跨局攒批，如 R）；否则由 `unichess_kit.serving` 把六方法包装成 Player
  （每进程一局，4 进程并行，如 T 的 C++ MCTS、M6）。
- 观战 `POST /api/arena/new`（`white_model/white_arg/black_model/black_arg/fen`）→ 一局 game job 在后台连续下完；
  `POST /api/arena/games/{id}/step` 按序揭示下一步，引擎还没走出时最多等 20 秒，仍无则 `step.pending=true`；
  `step.turn` 为刚走棋的一方，`step.eval` 为白方视角。最多 2 局同时观战（超出停最旧一局并记 `stopped`）；
  `DELETE` 结束并入库；没人单步的局下完后由 Monitor 自动入库。
- 批量对弈 `POST /api/arena/batch/start` / `POST /api/arena/batch/stop` **仅管理员**（`X-Admin-Token` 或直连 localhost；
  隧道流量靠转发头识别为公网）。轮数为 2..200 的偶数，同开局换色成对；开局取 kit 自带开局库。统计按模型 A/B
  （`tally.a_win/b_win/draw`），`GET /api/arena/batch/state` 的 `batch.summary` 带 Elo±95%CI、五项分布、重复局率等。
  每局以 `<batch_id>-g<n>` 入库，`GET /api/arena/records?batch_id=<id>` 按批次筛选（`__batch__` = 所有批次局）。
- 错误码：409 已有批次在跑；503 GPU 租约拒绝（训练等占用显存）；其余同对局接口。
- 服务重启：job 进程随 cgroup 一起结束；启动时 running 批次若 job 已写完就照常收尾，否则记 `interrupted`。

## GameEngine 契约

每个 `models/{model_name}/engine.py` 必须暴露 `class GameEngine`，实现六个方法：

- `__init__(**kwargs)`：kwargs 来自 `config.json` 中 `arg_name` 对应条目
- `setup(fen=None)`
- `human_move(uci)`：服务层已校验合法，不触发思考
- `engine_move()`：不接受参数，必须返回含 `"engine_move"` UCI 字符串的 dict，该字段为强制
- `state()`
- `undo()`
- `cleanup()`：释放 GPU/搜索树资源，淘汰时调用

可选类属性 `KIT_FACTORY`（见上节）。类属性 `IMPLEMENTED=False` + `NOT_IMPLEMENTED_REASON="..."` 表示占位未接入（`/api/models` 如实报告 `not_implemented`，`/api/new` 返回 501）。

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
| `T` | 符号链接 → `/home/jeefy/UniChess/Transformer`（Transformer 项目真实引擎，跨会话共享权重单例） | available，预设 `max_mcts` |
| `R` | 符号链接 → `/home/jeefy/UniChess/ResNet`（ResNet 项目引擎） | available，预设 `policy`, `fast`, `max_mcts`, `cpu` |
| `M6` | 符号链接 → `/home/jeefy/UniChess/M6`（M6 预览包：`chess_ai` 1M 监督 Transformer + `NeuralMCTS`，约 2.7M 参数 / 10MB 权重） | available，预设 `default`, `preview` |

命名约定：模型名保持简短（`T` 而非 `transformer`，`R` 而非 `resnet`），避免冗长；接入新项目时用符号链接 + 简短名，例如 `ln -s /home/jeefy/UniChess/ResNet models/R`。旧有的占位桩目录（`models/transformer/`、`models/resnet/`）已被对应的符号链接取代并清理。

### M6 引擎说明

- 部署物：`chess_ai_M6_preview_windows_v3.zip`（Windows x64 源码 + CPU 推理权重包），已解压到 `/home/jeefy/UniChess/M6` 并按 `SHA256SUMS.txt` 逐文件核验（26/26 通过），`Server/models/M6` 以符号链接接入。
- 适配层：`/home/jeefy/UniChess/M6/engine.py`，实现 GameEngine 六方法契约；包内模块全部命名空间在 `chess_ai` 之下，与 T/R 的裸顶层包名（`core`/`model`/`search`...）无冲突，不需要 `sys.modules` 隔离。
- **仅 GPU 推理**：`device` 只接受 `cuda`/`auto`，CUDA 不可用直接报错，不提供 CPU 回退（CPU 上一次搜索要数秒，无法用于对弈服务）。
- **默认配置即 M6 代码默认配置中最强一档**：`simulations=64`（`chess_ai/search/mcts.py` 的 `MCTSConfig` 默认值，代码中最大的默认模拟预算）、`eval_batch_size=8` 与 `reuse_tree=True`（`play_preview.py` 的生产默认）、`c_puct=1.5`。GPU 上约 0.1s/步。`preview` 预设是原 Windows 包的 16 模拟演示档位。
- 权重按 `(ckpt, device)` 进程级共享（约 24MB 显存），每个会话持有独立的 `NeuralMCTS` 搜索树；`cleanup()` 只释放会话级搜索树。
- 已知边界：预览版权重来自 1M 样本监督基线（未测 Elo），包内自带 SHA-256 基线可选核验（构造参数 `require_source_sha256`）。

## 当前状态

`T`、`R`、`M6` 均已接入并可对局（API 报告 `available`）。

## 已归档内容（2026-09-20 审查后移除，勿再 Serve）

- `/board`、`/status` 路由及 `static/board.html`、`static/status.html`：旧训练看板页面，只调用本服务不存在的 `/api/board`、`/api/status*` 端点，加载后必然空白。页面已移至 `~/UniChess/_legacy_server_20260920/`，路由一并移除；`index.html` 顶部原"训练看板"链接已删除。若未来需要看板，需先按本文件 API 契约实现 `/api/status*` 端点。
- `static/` 下的 `*.bak-*`、`*.before-*` 备份：旧版前端源码，曾可被公网下载，已移出静态目录；`/static` 另有中间件拦截 `.bak`/`.before-*`/点文件/`~` 结尾文件（404）。
- `_legacy_app_reference/`（旧 app.py、旧 unit、install_services.sh 等）：已移至 `~/UniChess/_legacy_server_20260920/`，避免重跑安装脚本覆盖新 unit。

## 测试

```bash
/home/jeefy/miniconda3/envs/unichess/bin/python -m unittest discover -s tests -v
```

在 Server 目录下运行。
