# UniChessServer

定位：统一棋类对局接口的 FastAPI 服务。任意棋类模型只需在 `models/{model_name}/engine.py` 暴露 `GameEngine` 类即可被前端调用。进程内最多同时持有 4 个对局实例（滑动窗口 LRU，淘汰最旧并调用 `cleanup()`）。走法合法性判定权威在服务层（python-chess）。

## 目录结构

- `app.py` 路由与调度
- `session_manager.py` 会话生命周期 + 合法性权威 + 调度时序（终局用 UniChessKit 的 `classify`）
- `jobs.py` 批量对弈与网页观战：提交 UniChessKit 后台 job（独立进程组 + GPU 租约），读 job 目录同步进度
- `kit_env.py` 定位 **import 根**（默认上级目录 `~/UniChess`；kit 不 pip 安装。`UNICHESS_IMPORT_ROOT` 可覆盖，`UNICHESS_KIT_ROOT` 仍兼容但语义已变为 import 根）
- `arena_storage.py` 对弈记录 / 批次 SQLite（`data/arena/arena_history.db`）
- `models/__init__.py` 模型发现/加载/预设查表
- `models/{model_name}/engine.py` + `config.json`
- `tools/fetch_stockfish.py` 下载并校验 Stockfish 官方二进制到 `tools/`（二进制本身不入 git）
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
- `GET /favicon.ico` → 站点图标（文件是 `static/favicon.jpg`，与 blog.jeefy.top 同一个文件，
  上游是 JPEG 所以按 `image/jpeg` 提供；三个页面都用 `<link rel="icon">` 指向它）

错误码：401 访问口令缺失/错误（见下节）；404 模型名非法/不存在或会话不存在（已被淘汰）；400 预设不存在、FEN 非法、走法不合法或轮到引擎；501 引擎未接入（`IMPLEMENTED=False`）；500 其它异常（detail 形如 `"ExceptionType: message"`）。

## 鉴权：两道彼此独立的门禁

两道门禁的口令**互不通用**：拿到一个不等于拿到另一个。

| 门禁 | 请求头 | 口令来源 | 覆盖范围 |
|---|---|---|---|
| 访问口令 | `X-Access-Token` | `UNICHESS_ACCESS_TOKEN`，否则 `~/.config/unichess/access_token`（600，不入库） | **所有** `/api/*` |
| 管理员口令 | `X-Admin-Token` | `UNICHESS_ADMIN_TOKEN`，否则 `~/.config/unichess/admin_token`（600，不入库） | 仅在 `/api/arena/batch/start`、`/stop` 之上叠加 |

- 实现在 `access_auth.py`（访问口令）与 `app.py` 的 `require_admin`（管理员口令）。两份代码刻意不共用常量与函数，
  免得日后加转发头时"改了 admin 忘了 access"；`tests/test_access_auth.py` 有对拍，两者漂移会红。
- **本机直连免口令**：对端是 `127.0.0.1`/`::1`/`localhost` 且不带任何代理转发头时，两道门禁都放行
  （运维脚本、systemd 看门狗、单测直接调函数都靠这条）。隧道把公网流量也送到 127.0.0.1，
  所以带 `x-forwarded-for`/`x-real-ip`/`cf-connecting-ip`/`forwarded`/`cf-ray` 任一头的请求一律不算本机。
- **未配置口令 = 拒绝远程**：两个门禁都是 fail-closed，没配口令时公网一律 401/403，只有本机直连能用。
- 口令比对：HTTP 规范会去掉头值首尾空白，其余字符必须完全一致（常量时间比较，截断/前缀都不行）。
- 静态页面（`/`、`/arena`、`/arena/batch`、`/static/*`）**不设门禁**：浏览器打开文档时带不了自定义请求头。
  访客口令由 `static/access-gate.js` 负责——它包住 `window.fetch` 注入请求头，收到 401 弹遮罩索要口令，
  拿到后存 `localStorage` 并自动重发原请求（最多重试 3 次）。三个页面都引入了它，
  `tests/test_access_auth.py` 会检查这一点，新增页面别忘了加 `<script src="/static/access-gate.js"></script>`。
  `/favicon.ico` 与 `/static/*` 同理不设门禁。
- `/docs`、`/redoc`、`/openapi.json` 已关闭：全部接口都要口令，没必要对外暴露接口清单。

`state()` 保证字段：`fen`（服务层权威 Board，引擎返回值不覆盖）、`legal_moves`、`is_game_over`、`engine_white`。引擎可额外提供：`last_move`、`san_history`、`eval{win,draw,loss,pov}`、`source`、`engine_ms`、`in_check`。

调度时序：`engine_white=true` 时新局由服务层触发引擎开局第一步；人类每走一步后轮到引擎则自动应答；悔棋回退双方各一步后若轮到引擎会重新触发引擎走子。

## 观战与批量对弈（UniChessKit job）

两者都不在服务进程里跑引擎，而是写 `data/jobs/<id>/job.json` 后启动 `python -m Kit.jobs`
（独立进程组，cwd 为 job 目录，启动前按 `gpu_mib` 申请 GPU 租约，显存不足直接 `gpu_busy` 退出）。
Server 只读 job 目录（`status.json` / `live.json` / `results.jsonl`），后台 Monitor 线程每 2 秒把进度入库。

- 引擎解析：`GameEngine` 声明类属性 `KIT_FACTORY = "包.模块:函数"`（模块须在模型目录内，以 `preset=<arg>` 调用）
  时走 kit 原生 Player（单进程 8 局并发、跨局攒批，如 R）；否则由 `Kit.serving` 把六方法包装成 Player
  （每进程一局，4 进程并行，如 T 的 C++ MCTS、M3）。
- 观战 `POST /api/arena/new`（`white_model/white_arg/black_model/black_arg/fen`）→ 一局 game job 在后台连续下完；
  `POST /api/arena/games/{id}/step` 按序揭示下一步，引擎还没走出时最多等 20 秒，仍无则 `step.pending=true`；
  `step.turn` 为刚走棋的一方，`step.eval` 为白方视角。最多 2 局同时观战（超出停最旧一局并记 `stopped`）；
  `DELETE` 结束并入库；没人单步的局下完后由 Monitor 自动入库。
- 批量对弈 `POST /api/arena/batch/start` / `POST /api/arena/batch/stop` **仅管理员**（`X-Admin-Token`；
   公网还需先过 `X-Access-Token` 访问门禁，见「鉴权」一节。本机直连两道都豁免）。
   轮数为 2..200 的偶数，同开局换色成对；开局取 kit 自带开局库。统计按模型 A/B
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
| `T` | 符号链接 → `/home/jeefy/UniChess/Transformer`（Transformer 项目真实引擎，跨会话共享权重单例） | available，预设 `max_mcts`, `max_t` |
| `R` | 符号链接 → `/home/jeefy/UniChess/ResNet`（ResNet 项目引擎） | available，预设 `policy`, `fast`, `max_mcts`, `cpu` |
| `S` | 符号链接 → `/home/jeefy/UniChess/SSM`（状态序列模型主线，stage B 自对弈 RL） | available，预设 `champion` |
| `M3` | 符号链接 → `/home/jeefy/UniChess/M3`（上游冻结可玩包：`src/chess_ai` 30M finalist + `NeuralMCTS`，3,695,244 参数 / 14.8MB 权重） | available，预设 `default`, `preview`, `policy` |
| `M2` | 真实目录 `models/M2/`（适配层）+ `/home/jeefy/UniChess/M2`（上游 chess_ai neural v2.0.0 原包：策略网 `ChessCNN` + 价值网 `ResidualValueModel` + negamax/PUCT 搜索） | available，预设 `default`, `fast`, `deep`, `policy` |
| `DS` | 符号链接 → `/home/jeefy/UniChess/DS`（`DS` 仓库 = jeefies/UniChessLLM，百炼 OpenAI 兼容端点大模型引擎；`.env` 在 DS 仓库内、不入库） | available，预设 `default`, `fast` |
| `SF` | 真实目录 `models/SF/`（仓内适配层）+ `tools/stockfish`（Stockfish 19 官方二进制，**不入 git**，见下） | available（装好二进制后），预设 `default`, `fast`, `strong`, `depth12`, `weak`, `elo_1500` |

命名约定：模型名保持简短（`T` 而非 `transformer`，`R` 而非 `resnet`），避免冗长；接入新项目时用符号链接 + 简短名，例如 `ln -s /home/jeefy/UniChess/ResNet models/R`。旧有的占位桩目录（`models/transformer/`、`models/resnet/`）已被对应的符号链接取代并清理。

### M3 引擎说明

M3 对应的是**上游冻结产物**（上游模型名 M6 30M finalist，注册名为 M3），不由本目录维护：
升级 = 用新发布的可玩包整体替换 `/home/jeefy/UniChess/M3`（不合并、不改代码），
替换后跑 `tests/test_m3_engine.py` 做接入验收。

- 部署物：`M6-30M-UniChessServer-playable.zip`（上游模型叫 M6，这里注册成 M3），
  解压到 `/home/jeefy/UniChess/M3` 并按包内 `SHA256SUMS` 逐文件核验（25/25 通过），
  `Server/models/M3` 以符号链接接入。
  zip 本身含 `.pt` 权重，**不放仓库**（已加 `.gitignore`），
  原件留在 `~/UniChess/.m6-backup-20260927/`。
- 包结构：`engine.py`（适配层）+ `src/chess_ai/`（网络与推理闭包）+ `weights/`（单一推理权重）
  + `config.json` + `SHA256SUMS`。`engine.py` 自己把 `src/` 插进 `sys.path`，
  因此引入的是 `chess_ai.*`，与 T/R 的裸顶层包名（`core`/`model`/`search`...）无冲突。
- 适配层实现 GameEngine 六方法契约；构造只接受 `simulations` / `eval_batch_size` /
  `reuse_tree` / `c_puct` / `device` 五个参数，**没有 `ckpt` 参数**（权重路径包内硬编码，
  换权重只能换包）。`state()` 的 `eval` 是白方视角 WDL 字典（`{win,draw,loss,pov}`），
  终局为 `None`。
- **仅 GPU 推理**：`device` 只接受 `cuda`/`auto`/`cuda:0`，CUDA 不可用直接报错，不提供 CPU 回退。
- **默认配置即上游 M6 模型代码默认配置中最强一档**：`simulations=64`、`eval_batch_size=8`、
  `reuse_tree=True`、`c_puct=1.5`。GPU 上约 0.1–0.2s/步。`preview` 是 16 模拟演示档位。
- 权重按 `(artifact, device)` 进程级共享（约 24MB 显存），每个会话持有独立的 `NeuralMCTS`
  搜索树；`cleanup()` 只释放会话级搜索树。
- 已知边界：包只做人机对弈适配，不带开局库/UCI/时间控制/批量优化；上游未跑完整对局与
  性能扫描，Elo 未知。

### M2 引擎说明

同样是上游冻结产物，但**结构与 M3 不同**，接线方式也不一样：

- 部署物：`kesiweim/chess_ai` 的 release `v2.0.0`（资产 `neural-v2.0.0.zip`，
  GitHub API digest `e42b92a4…d18ef6`），解压到 `/home/jeefy/UniChess/M2`。
  包是一堆**裸顶层模块**（`model_cnn` / `search_engine` / `neural_fast` …）+ 两个权重
  （策略网 `chess_model_balanced.pt` 13,157,801B、价值网
  `value_full_runs/*/value_full_epoch2.pt` 2,590,579B），
  SHA-256 与上游 README 声明值逐一核对一致。zip 原件在 `~/UniChess/.m2-package-20260927/`。
- **适配层在仓库里**：`models/M2/engine.py`（+ `config.json`）。因为上游没带
  UniChessServer 适配层，这是 Server 侧的接线代码，归本仓库维护；
  包目录本身不放 git、也不改一个字节。包根默认取 import 根下的 `M2`，
  可用 `UNICHESS_M2_ROOT` 覆盖（测试/异地部署用）。
- **sys.path 必须常驻**：模块互相用裸名 import，且 `search_engine.policy_order`
  里有函数内惰性 import，所以适配层把包目录插在 `sys.path[0]` 后不能撤。
  因为 M3 用的是 `src/chess_ai` 命名空间包，两者不冲突（有共存用例兜底）。
- **线程数（关键）**：本机 20 逻辑核上 torch 默认开满会让 19×8×8 的微型 CNN 慢 450 倍
  （评估 0.119ms → 54.5ms，2 秒只够 depth 1）。适配层默认压到 4 线程
  （与上游 `play_v4.py` 一致），`threads` 参数可调，`None` 表示不动全局设置。
  这是进程级全局设置，服务进程里 T/R/S/M3 都跑 GPU、不受影响。
- **CPU 引擎**：不需要 CUDA（`FastEvaluator` 反而要求 CPU 模型）。
- **每会话一份评估器**：`FastEvaluator` 复用输入缓冲区、上游文档明言不可并发共用，
  因此权重进程级共享，但 `TracedEvaluator` 与搜索树每会话各一份。
- **`eval` 恒为 None**：前端只渲染 WDL 字典，而 M2 只有一个 tanh 标量，
  不伪造分布；原生分（白方视角，行棋方视角取负）放在 `state()['value_tanh']`。
- 搜索是**时间预算 + 深度上限**（不是模拟数）：`seconds` 为硬墙、`depth` 为迭代加深
  上限，`qdepth` 是静态搜索深度。`claim_draw=True` 时上游可能回报"建议申领和棋"
  而不是走法，此时适配层显式报错（服务层 classify 本就把可申领和棋判终局）。
- 已知边界：上游 README 自称"v4 原版稳定基线"，未跑完整对局与性能扫描，Elo 未知。

### SF 引擎说明

SF = **Stockfish 19**（2026-09-05 发布，GPLv3，官方仓库
`official-stockfish/Stockfish`，NNUE + alpha-beta）。开源国际象棋引擎里没有
比它更强的：它是 CCRL/TCEC/CCC 等榜单的常任第一，fishtest 社区持续回归验证。
接入它是为了给 T/R/S/M2/M3/DS 一个**同一规则、同一裁决口径下的可复现 baseline**：
人机对弈可以直接下，批量对弈 `/api/arena/batch/start` 选 `SF` 对任一自研引擎
即可量 Elo 差距（预算固定时 `depth12` 档的复现性最好）。

接线方式与 M2 同构：**适配层在仓库里，引擎本体是第三方二进制**。

- 部署物：官方 release `sf_19` 的预编译二进制（约 100MB，GPLv3，**不入 git**）。
  安装：在 Server 目录跑 `python tools/fetch_stockfish.py`（自动识别平台、下载、
  sha256 校验、解压安装到 `tools/stockfish[.exe]`；`--check` 只校验已安装版本）。
  也可 `UNICHESS_STOCKFISH_BIN=/path/to/stockfish` 指向系统里已有的一份。
- 首批实测（2026-10-08，4 局/2 局小样本，仅示意"同一口径下怎么读"）：
  `SF(fast)` 2:0 `M2(fast)`（均将杀）；`SF(weak)`（Skill 3 + 0.3s）0:4 `T(max_mcts)`
  ——自研 T 已强过被大幅放水的 SF；`SF(default)`（1s 满力）4:0 `T(max_mcts)`
  ——满力 SF 仍显著在 T 之上。合计 Elo 差距请跑几十轮再看（`elo_ci95` 才有意义）。
- 二进制找不到时**模块 import 不受影响**：`/api/models` 照常列出 SF，只有
  `/api/new` 真正建会话时报错（detail 带安装提示），不会把模型清单带红。
- 每个会话一个 UCI 子进程（stdin/stdout，经 python-chess 的 `SimpleEngine` 包装），
  进程间零共享。`cleanup()` 先 `quit` 后兜底强杀：SF 常驻搜索线程必须回收，
  否则批量对弈反复建会话会把机器啃干净。
- **Threads/Hash 默认保守**（1 线程 / 16MB）：进程内最多 4 个对局会话，批量对弈
  还有 4 个 worker 进程，每份都开满线程会跟同机的 GPU 训练抢核；要更强再加
  `threads` / `hash_mb`（构造参数，预设档里已给 `strong` = 2 线程 64MB）。
- 不声明 `KIT_FACTORY`：观战 / 批量对弈走 `Kit.serving:game_engine_player_factory`
  包装路径（每 worker 进程一局，4 进程并行），SF 用 CPU 不占显存，GPU 租约照常申请。
- 参数（白名单，未知 kwarg 抛 `TypeError`）：`binary` / `movetime_ms` / `depth` /
  `nodes` / `skill_level`(0–20) / `uci_elo`(1320–3190) / `hash_mb` / `threads` /
  `show_wdl` / `move_overhead_ms`。搜索预算三选一，优先级 `depth > nodes > movetime_ms`；
  `skill_level` 与 `uci_elo` 互斥。
- **eval 是真 WDL**：默认开 `UCI_ShowWDL`，`state()['eval']` 是引擎自报的
  `{win, draw, loss, pov: 'white'}`（千分比归一化，白方视角）；关掉 `show_wdl`
  或引擎不回报时 `eval` 为 `None`，原始分在 `score_cp` / `score_mate` / `pv` /
  `nodes` / `depth` / `seldepth` / `nps` 里。人类走子后旧分数作废（不像 M2 有
  廉价价值网可重算），置空等引擎应答再给分。
- 已知边界：`uci_elo` 与 `skill_level` 的"弱化"是引擎自带的拟人化抽样，
  不代表真实 Elo；`movetime_ms` 是硬墙，搜索尾段可能略微超预算。

### DS 引擎说明

DS 是**仓库内实现**的大模型引擎，不依赖本地权重文件，靠 `.env` 读取阿里云百炼
（OpenAI 兼容端点）配置：

- 接入仓库：`jeefies/UniChessLLM`（本地目录 `DS/`），`Server/models/DS` 以符号链接
  指向 `~/UniChess/DS`。
- 依赖：仅标准库 `urllib` + `chess`；**不新增任何依赖**。
- 构造参数白名单：`model / timeout_s / max_attempts / temperature / max_tokens /
  history_plies / thinking / extra_request`；未知 kwargs 抛 `TypeError`。
- `thinking` 档：请求体带 `enable_thinking` 开关；端点返回 HTTP 400 且错误文本疑似不认
  该参数时，同 attempt 去参重发并将"不支持"结论缓存，后续不再带参。
- 回复解析容错：裸 UCI / `MOVE:` 行 / 全部 UCI token 取最后一个合法者 / SAN 兜底 /
  `reasoning_content` 兜底；全部需要 `chess.Move.from_uci` + `move in board.legal_moves`。
- 失败语义（用户确认）：重试耗尽直接抛错，不兜底。普通对局返回 500；观战/批量对弈 job
  以 error 终止。
- `eval` 恒为 `None`（LLM 不产出可信 WDL，不伪造）；`llm` 字段记录上一步引擎调用的元信息
  （`model / attempts / raw / reasoning_chars / thinking / thinking_param_supported`）。
- **不声明 `KIT_FACTORY`**：观战 / 批量对弈走 `Kit.serving:game_engine_player_factory`
  包装六方法（4 worker、每进程一局）。
- 密钥纪律：`.env` 不入库；异常/日志/状态返回均不含 API key。

### 给模型加/改预设：config.json 与 *.local.json

每个模型的预设来自两处，后者覆盖前者（同名预设按字段合并）：

1. `models/<模型名>/config.json` —— 模型自带。**软链进来的冻结包不要改这里**：
   M3/R/S/T 的 config.json 被包内 `SHA256SUMS` 罩着，改一个字节完整性自检就红，
   也丢了"与上游一致"的凭证。
2. `models/<模型名>.local.json` —— **仓库内的覆盖文件**，跟 git 走、不进上游包。

`list_presets()` / `resolve_kwargs()` / `describe_model()` 三处都会自动合并，
所以 `/api/models` 与 `/api/new?arg_name=…` 直接就能看到覆盖后的预设。
M3 的 `policy` 档就是这么加的（`models/M3.local.json`）。

## 当前状态

`T`、`R`、`S`、`M3`、`M2`、`DS` 均已接入并可对局（API 报告 `available`），其中 M2、M3 各有一个 `policy`（仅策略网）档。
`SF`（Stockfish 19 baseline）也已接入，见「SF 引擎说明」：二进制需先跑
`tools/fetch_stockfish.py` 安装，装好后 API 报告 `available`。

## 已归档内容（2026-09-20 审查后移除，勿再 Serve）

- `/board`、`/status` 路由及 `static/board.html`、`static/status.html`：旧训练看板页面，只调用本服务不存在的 `/api/board`、`/api/status*` 端点，加载后必然空白。页面已移至 `~/UniChess/_legacy_server_20260920/`，路由一并移除；`index.html` 顶部原"训练看板"链接已删除。若未来需要看板，需先按本文件 API 契约实现 `/api/status*` 端点。
- `static/` 下的 `*.bak-*`、`*.before-*` 备份：旧版前端源码，曾可被公网下载，已移出静态目录；`/static` 另有中间件拦截 `.bak`/`.before-*`/点文件/`~` 结尾文件（404）。
- `_legacy_app_reference/`（旧 app.py、旧 unit、install_services.sh 等）：已移至 `~/UniChess/_legacy_server_20260920/`，避免重跑安装脚本覆盖新 unit。

## 测试

```bash
/home/jeefy/miniconda3/envs/unichess/bin/python -m unittest discover -s tests -v
```

在 Server 目录下运行。

- 远端 169 项全绿（含 SF 22 项；前提是先跑过 `tools/fetch_stockfish.py`）。
- Windows 本机：4 项必失败（3 项符号链接权限 + 1 项 `../M2` 包不在本机，
  `describe_model('M2')` 报 error），都与门禁无关，干净树上同样失败；
  SF 的用例在装了 `tools/stockfish.exe` 的本机照常跑，没装二进制时整体 skip
  （与 DS 的 skip 口径一致），不会新增必失败项。
