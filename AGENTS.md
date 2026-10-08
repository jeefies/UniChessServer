# Server AGENTS.md

面向 AI 编码 agent。项目群总览与隐私纪律见上级 `../AGENTS.md`（最高优先级）。

## 环境与命令

- Python：`/home/jeefy/miniconda3/envs/unichess/bin/python`（无 pytest，全部用标准库 `unittest`）
- 无 `pyproject.toml` / `requirements.txt` / CI：直接 `python app.py` 运行
- 启动：`python app.py --host 127.0.0.1 --port 8000`（仅接受 `--host/--port`）
- 测试：`python -m unittest discover -s tests`（在 Server 目录下跑；190 项，
  其中访问门禁 30 项 / M3 19 项 / M2 24 项 / DS 15 项 / jobs 34 项 / server 28 项 /
  SF 对弈 22 项 / SF 分析 18 项）。
  Windows 本机有 4 项必失败：3 项符号链接权限 + 1 项 `../M2` 包不在本机
  （`describe_model('M2')` 报 error），都与门禁无关，干净树上同样失败；
  远端 190 项全绿（SF 前提：装好二进制，见「SF 接线」）
- **批量对弈汇总里的 `inf` 不能进 HTTP 响应**：`Kit.stats` 的 Elo 哨兵是 `ELO_INF`
  （`float('inf')`），一方全胜/还没下完时汇总里必然出现（`elo` / `elo_ci95` /
  `elo_pentanomial*`），job 的 status.json 原样带出来。starlette 的 JSONResponse
  用 `allow_nan=False`，一个 inf 让整条 `/api/arena/batch/state` 变 500
  （自研引擎实力接近时极少全胜，SF 这种 baseline 次次踩中，2026-10-08 修）。
  出口在 `jobs._json_finite()`（snapshot 与观战 step 两个出口都过它），
  新增透出 job 状态的路由必须同样处理，回归测试见 `tests/test_jobs.py`
  `test_snapshot_sanitizes_infinite_elo`。
- 远端部署：systemd 用户级服务 `unichess-server.service` + `unichess-tunnel.service`
- 接口清单与模型插件契约见 `README.md`（刷新区块务必同步两份）

## Kit 依赖加载（易踩坑）

`Kit` 不 pip 安装。扁平布局（2026-09-25）后 `kit_env.py` 把 **import 根**追加到
`sys.path`，之后 `import Kit` / `import Transformer` 才可用：

- 默认 import 根 = Server 目录的父目录（`~/UniChess`）；`UNICHESS_IMPORT_ROOT` 可覆盖，
  旧名 `UNICHESS_KIT_ROOT` 仍兼容（语义已从"Kit 仓库目录"变为"import 根"）
- **追加而非前置**：Kit 仓库本身也有 `tests/` 包，前置会遮蔽 Server 自己的 `tests` 模块
- `jobs.py` 启动子进程时，通过 `kit_env.subprocess_env()` 设置 `PYTHONPATH` 和
  `PYTHONIOENCODING=utf-8`
- 模型目录 `models/<短名>` 是符号链接，指向各引擎仓库；各引擎的 `engine.py` 自己会再挂
  一次 import 根（两边都追加、不重复，顺序不受影响）

## 六方法契约与引擎接线

- 引擎实现 `setup / human_move / engine_move / state / undo / cleanup` 六个方法；
  `engine_move` 返回的 `info` 里的 `q`（行棋方视角）会换算成白方视角的 `eval` 供前端展示，
  引擎自带的 `eval` 字段原样透传
- `engine.py` 声明 `KIT_FACTORY = "<包>.kit:make_player_factory"` 时，观战与批量对弈走
  kit 原生 Player（单进程 8 局并发、跨局攒批）；否则由 `Kit.serving` 把六方法包装成 Player，
  每个 worker 进程一局，4 进程并行
- 新引擎接入：`ln -s /home/jeefy/UniChess/<项目> ~/UniChess/Server/models/<短名>`，
  并在 `config.json` 里配 preset
- `models/M3`（上游 M6 30M finalist 冻结包，勿重构）：升级只能整体替换
  `~/UniChess/M3`（zip 原件留在 `~/UniChess/.m6-backup-20260927/`，不在任何仓库里），
  替换后跑 `tests/test_m3_engine.py` 验收。它自己没有 git 仓库，
  `Server/.gitignore` 用权重/压缩包规则挡住误提交
- M2（上游 `chess_ai` neural v2.0.0，CPU 引擎）**适配层在仓库里**：
  `models/M2/engine.py` + `config.json`，包在 import 根下 `M2/`（不在 git，
  `UNICHESS_M2_ROOT` 可覆盖包根）。与 M3 的结构差别：M3 是 `src/chess_ai`
  命名空间包 + 自带适配层；M2 是一堆裸顶层模块、没有适配层，由本仓接线。
- M2 的两个坑：① 20 逻辑核上 torch 线程开满会让微型 CNN 慢 450 倍，
  适配层默认压 4 线程（进程级全局，GPU 引擎不受影响）；② `FastEvaluator`
  复用输入缓冲区不可并发共用，权重共享但评估器/搜索树每会话一份。
- M2 的 `eval` 恒为 None（只有 tanh 标量、不伪造 WDL），原生分在 `value_tanh`。
 - M2/M3 各有一个 `policy` 档：M2 是真·仅策略（`policy_only=true`，不进搜索、
   不加载价值网，约 1ms/步）；M3 只能把 `simulations` 压到冻结包下限 1（约 40–65ms/步）。
 - **DS（`jeefies/UniChessLLM`，百炼大模型引擎）**：仓库内实现，不依赖本地权重；
   通过 `.env` 读取百炼 OpenAI 兼容端点配置。构造参数白名单 `model / timeout_s /
   max_attempts / temperature / max_tokens / history_plies / thinking / extra_request`；
   `thinking` 档带 `enable_thinking` 开关，端点 400 疑似不认时自适应降级。
   不声明 `KIT_FACTORY`，观战 / 批量对弈走 `Kit.serving` 包装路径。
   `eval` 恒为 `None`；`llm` 字段记录上一步引擎调用元信息。
- **软链进来的冻结包不要改它的 `config.json`**（被包内 SHA256SUMS 罩着）：
  加/改预设写 `models/<模型名>.local.json`，`_load_config` 会把两处按预设名合并，
  `list_presets`/`resolve_kwargs`/`describe_model` 都会看到合并结果。
- **SF = Stockfish 19（开源最强引擎，baseline）**：适配层在 `models/SF/`（仓内），
  引擎本体是官方 release `sf_19` 预编译二进制（约 100MB、GPLv3、**不入 git**），
  装在 `Server/tools/stockfish[.exe]`。安装：`python tools/fetch_stockfish.py`
  （按平台下载 + sha256 校验 + 解压；`--check` 校验已装版本）。定位顺序：
  `UNICHESS_STOCKFISH_BIN` → `tools/stockfish[.exe]` → PATH。二进制缺失只让
  `/api/new` 报错（detail 带提示），`/api/models` 与 import 不受影响——适配层的
  二进制解析必须在 `__init__` 里做，模块级不能碰。
- **远端部署三连**（新机器/换机同样适用；`git pull` 后二进制不会跟着来）：
  `git pull origin rebuild` → `python tools/fetch_stockfish.py`（下完记得
  `--check`）→ `systemctl --user restart unichess-server`，然后
  `curl -s localhost:8000/api/models` 确认 SF 是 `available`（本机直连免口令）。
- SF 每个会话一个 UCI 子进程（python-chess `SimpleEngine`），进程间零共享。
  Threads 默认 1、Hash 默认 16MB：进程内最多 4 会话 + 批量 4 worker，
  不能跟同机 GPU 训练抢核（要更强用 `strong` 档或加 `threads`/`hash_mb`）。
  `cleanup()` 先 `quit` 后兜底 `close()` 强杀，SF 常驻搜索线程必须回收。
- SF 不声明 `KIT_FACTORY`（观战/批量走 `Kit.serving:game_engine_player_factory`，
  每 worker 一局，CPU 引擎不占显存）。默认开 `UCI_ShowWDL`，`state()['eval']`
  是引擎自报的白方视角 WDL；关掉后 `eval=None`、原始分在 `score_cp`/`score_mate`/
  `pv`/`nodes`/`depth`。构造参数白名单见 `models/SF/engine.py` 的 `_allowed_keys`，
  搜索预算 `depth > nodes > movetime_ms`，`skill_level` 与 `uci_elo` 互斥。
- SF 用例（`tests/test_sf_engine.py`）在二进制缺失时整体 skip，与 DS 同口径；
  装上后远端 22 项应全绿。
- M3 的 `engine.py` 只认 `simulations`/`eval_batch_size`/`reuse_tree`/`c_puct`/`device`
  五个参数（无 `ckpt`，权重路径包内硬编码），`state()` 的 `eval` 是白方视角 WDL 字典
- M3 的 `state()` 用 `en_passant="fen"`（双步进兵后总写给区格），服务层 `state()` 用
  python-chess 默认口径并**用自己的 fen 覆盖**，故前端看到的始终是服务层口径；
  比对两边局面时比着法序列，别比 fen 字符串（`test_m3_engine._histories`）

## 观战揭示的一个不变量（踩过坑）

`jobs.ArenaWatch.step` 一步揭示一个**完整**的着法记录：着法与它的 detail 必须原子给出。
worker 是先追加着法再补 detail 的，只等着法就揭示会让前端评估条先看到 `eval=None` 再跳变；
而对局已终局（detail 不会再长）时才允许拖着法单收，否则最后一步会永久卡住。
`test_jobs.TestArenaService.test_game_plays_to_end_and_saves` 锁的就是这条。

## 后台 job

- 观战与批量对弈都不在服务进程里跑引擎，而是写 `data/jobs/<id>/job.json` 后启动
  `python -m Kit.jobs <job_dir>`
- Server 只读写 `data/jobs/<id>/`，不直连进程；服务重启后凭目录接管或收尾
- 退出码：0 完成、1 出错、2 显存不足、3 被停、4 已在运行
- 批量对弈 `/api/arena/batch/start`、`/stop` 仅限管理员：请求头 `X-Admin-Token`，
  口令在远端 `~/.config/unichess/admin_token`（600 权限，不入库；也可用环境变量
  `UNICHESS_ADMIN_TOKEN`）

## 鉴权：两道彼此独立的门禁（2026-09-28 起全部接口都要口令）

| 门禁 | 请求头 | 口令来源 | 覆盖范围 |
|---|---|---|---|
| 访问口令 | `X-Access-Token` | `UNICHESS_ACCESS_TOKEN`，否则 `~/.config/unichess/access_token` | **所有** `/api/*` |
| 管理员口令 | `X-Admin-Token` | `UNICHESS_ADMIN_TOKEN`，否则 `~/.config/unichess/admin_token` | 只在 batch start/stop 之上叠加 |

- 访问门禁实现在 `access_auth.py`（`require_access_token`），管理员门禁在 `app.py`
  （`require_admin`）。两份常量与函数刻意不共用：转发头清单会随部署变化，共用就会出现
  "改了 admin 忘了 access"。`tests/test_access_auth.py` 有对拍（互相冒充口令、环回口径、
  路由表完备性），漂移会红。
- **新增 /api 路由必须挂门禁**：`dependencies=[Depends(require_access_token)]`，
  `test_every_api_route_is_gated` 会扫路由表兜底；新增静态页面要引入
  `<script src="/static/access-gate.js"></script>`（`test_pages_load_the_gate` 兜底）。
- **页面路由（`/`、`/arena`、`/arena/batch`）不能挂门禁**：浏览器打开文档时带不了自定义
  请求头，口令只能由 `static/access-gate.js` 注入（401 弹遮罩 → 存 localStorage → 自动重发）。
- **本机直连免口令**（对端 127.0.0.1/::1/localhost 且无任何代理转发头）：运维与单测靠这条。
  隧道把公网流量也送到 127.0.0.1，所以带 `x-forwarded-for`/`x-real-ip`/`cf-connecting-ip`/
  `forwarded`/`cf-ray` 任一头的一律不算本机。
- **fail-closed**：没配口令时公网一律 401/403，只有本机直连可用。
- `/docs`、`/redoc`、`/openapi.json` 已关闭（`docs_url=None` 等），未匹配的 /api 路径仍是 404。

## 隧道与代理

- 链路：Cloudflare → nginx（36.151.145.113）→ 隧道 8800 → `127.0.0.1:8000`
- 隧道流量源地址也是 127.0.0.1，靠代理转发头区分真实客户端，**勿删 `app.py` 里的转发头判断**
- 自愈看门狗：`unichess-tunnel-ensure.timer`（30s 幂等），异常时它会重建隧道

## 已知取舍：重启后首局 S 会 522

每次重启 server 后，**第一次**开 S 对局会返回 522：该进程内首次要 import torch/mamba、
加载 101MB 的 champion.pt、跑 triton autotune，本地实测约 60s，超过 Cloudflare 的代理预算
（T/M3 轻得多，重启后 1.6–2.0s 就回）。热路径正常：建局 2.9–4.5s、每步 <0.2s。

**2026-09-27 决定不做启动预热**：预热会让服务启动多背约 3s 和常驻 2.8G 显存，
且已明确不改 S 的加载方式。若哪天要消除这个 522，唯一该做的改法是启动时预热一次
（而不是动引擎或搜索参数）。
