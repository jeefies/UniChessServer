# Server AGENTS.md

面向 AI 编码 agent。项目群总览与隐私纪律见上级 `../AGENTS.md`（最高优先级）。

## 环境与命令

- Python：`/home/jeefy/miniconda3/envs/unichess/bin/python`（无 pytest，全部用标准库 `unittest`）
- 无 `pyproject.toml` / `requirements.txt` / CI：直接 `python app.py` 运行
- 启动：`python app.py --host 127.0.0.1 --port 8000`（仅接受 `--host/--port`）
- 测试：`python -m unittest discover -s tests`（在 Server 目录下跑；72 项）
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
- `models/M6` 是他人项目，**勿重构**；它现有 3 个失败用例与 Server 无关

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

## 隧道与代理

- 链路：Cloudflare → nginx（36.151.145.113）→ 隧道 8800 → `127.0.0.1:8000`
- 隧道流量源地址也是 127.0.0.1，靠代理转发头区分真实客户端，**勿删 `app.py` 里的转发头判断**
- 自愈看门狗：`unichess-tunnel-ensure.timer`（30s 幂等），异常时它会重建隧道
