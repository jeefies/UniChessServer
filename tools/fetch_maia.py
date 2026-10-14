#!/usr/bin/env python3
"""下载并校验 9 个 Maia 档位权重到 <import根>/Maia/（权重不入 git）。

背景：Maia 是 UniChess 的**人类 Elo 标尺**（`Kit/maia.py` + `Kit/players/maia.py`，
Server 侧适配层 `models/MAIA/`）。Stockfish 的 UCI_Elo 限力档实测被压缩 2.3~3.9 倍、
名义值不可当 Elo 读（见 `SSM/docs/representation-and-strength-audit.md` §5.4）；
Maia 的档位标签本身就是人类 Elo（Lichess 对应分段训练），而且下人类开局、是神经网络，
能同时解决"标尺未校准"和"开局库偏差"。

权重来源：``CSSLab/maia-chess`` v1.0 release（9 个 ``.pb.gz``，各约 1.3MB，
lc0 ``net.proto`` 格式）。GPL，按项目纪律**不进任何 git 仓库**；这个脚本只负责
"从官方 release 拉正确的那几份"。

用法（在 Server 目录下）：

    python tools/fetch_maia.py            # 下载缺的档位 + 校验 + 安装
    python tools/fetch_maia.py --check    # 只校验已安装的
    python tools/fetch_maia.py --dest DIR # 装到别处
    python tools/fetch_maia.py --tiers 1100 1500 1900   # 只装指定档位

网络说明：release 资产直连 ``github.com/.../releases/download/...``；某些网络
（如 ulab）只放行 ``api.github.com``，脚本会自动退回 API 的 asset 端点
（``/repos/{owner}/{repo}/releases/assets/{id}`` + ``Accept: application/octet-stream``）。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

_OWNER = 'CSSLab'
_REPO = 'maia-chess'
_TAG = 'v1.0'

# (档位, GitHub release asset id, sha256)。asset id 与 sha256 来自
# https://github.com/CSSLab/maia-chess/releases/tag/v1.0 ；换版本必须同步改这张表。
_TIERS = [
    (1100, 30665573, 'e1cf1cd0c96b8a4fa6a275f4b9fd54ed1ffebf9fe44641b9fceded310e9619c4'),
    (1200, 30665575, 'ead4ba953f233ae732999ebc1e2b675378148527ebcfad2f0acbc5e4c224d98e'),
    (1300, 30665576, '36195f87bf4761834baa0bf87472b18509a7261a9d7d6f1a8443261369a733f2'),
    (1400, 30665577, 'd5353ea6766356dad2d28920c6692f37a5f30963767f1a3105d33b4d0af011e8'),
    (1500, 30665578, '35ab6f20421d59e1df3b17c5a5016947af4c6761368ef84044a9a9c7619a9a00'),
    (1600, 30665579, 'd2c9e5948581acf4b9fc0b1e720c5dc0fe64ce80cfc4a239d3f8a42e1176c876'),
    (1700, 30665580, 'd277eacd792d340a30abb464dc65127254e65cac57abca17facc469889b96478'),
    (1800, 30665581, '0031ad7c4256b1fd09fbebd28418d644d68b26cd2a45df4967ccf5c7ec9c4965'),
    (1900, 30665582, 'e2f565f42d7cd9f122557e6dc4eb84e5bbaedceda1d404dc485d3611c7c97a12'),
]

_SERVER_ROOT = Path(__file__).resolve().parent.parent
_UA = 'unichess-maia-deploy'


def _default_dest() -> Path:
    """默认装到 import 根下的 ``Maia/``（与 M2/M3 同级、不在 git）。"""
    root = os.environ.get('UNICHESS_IMPORT_ROOT') or os.environ.get('UNICHESS_KIT_ROOT')
    if root:
        return Path(root) / 'Maia'
    return _SERVER_ROOT.parent / 'Maia'


def _fetch(elo: int, asset_id: int) -> bytes:
    name = f'maia-{elo}.pb.gz'
    direct = (f'https://github.com/{_OWNER}/{_REPO}/releases/download/'
              f'{_TAG}/{name}')
    api = (f'https://api.github.com/repos/{_OWNER}/{_REPO}/releases/'
           f'assets/{asset_id}')
    attempts = ((direct, 'application/vnd.github+json'),
                (api, 'application/octet-stream'))
    last = None
    for url, accept in attempts:
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': _UA, 'Accept': accept})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            if data[:2] == b'\x1f\x8b':
                return data
            last = f'{url} 返回的不是 gzip（{len(data)} 字节）'
        except urllib.error.HTTPError as e:
            last = f'{url} -> HTTP {e.code}'
        except Exception as e:                       # noqa: BLE001
            last = f'{url} -> {type(e).__name__}: {e}'
    raise SystemExit(f'下载 {name} 失败：{last}。'
                     f'可手动从 {direct} 下载后放到目标目录。')


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='只校验已安装的权重')
    ap.add_argument('--dest', default=None, help='安装目录（默认 import 根/Maia）')
    ap.add_argument('--tiers', type=int, nargs='*', default=None,
                    help='只处理这些档位（默认全部 9 个）')
    args = ap.parse_args()

    dest = Path(args.dest) if args.dest else _default_dest()
    tiers = args.tiers if args.tiers else [t for t, _, _ in _TIERS]
    unknown = sorted(set(tiers) - {t for t, _, _ in _TIERS})
    if unknown:
        raise SystemExit(f'未知档位 {unknown}；可用：{[t for t, _, _ in _TIERS]}')

    if args.check:
        bad = 0
        for elo, _, digest in _TIERS:
            if elo not in tiers:
                continue
            p = dest / f'maia-{elo}.pb.gz'
            if not p.exists():
                print(f'  maia-{elo}: 缺失（{p}）')
                bad += 1
                continue
            got = _sha256(p)
            ok = got == digest
            print(f'  maia-{elo}: {"OK" if ok else "sha256 不匹配"}  {p}')
            if not ok:
                print(f'      期望 {digest}')
                print(f'      实得 {got}')
                bad += 1
        raise SystemExit(1 if bad else 0)

    dest.mkdir(parents=True, exist_ok=True)
    print(f'目标目录: {dest}')
    ok = fail = 0
    for elo, asset_id, digest in _TIERS:
        if elo not in tiers:
            continue
        p = dest / f'maia-{elo}.pb.gz'
        if p.exists() and _sha256(p) == digest:
            print(f'  maia-{elo}: 已是最新，跳过')
            ok += 1
            continue
        data = _fetch(elo, asset_id)
        got = hashlib.sha256(data).hexdigest()
        if got != digest:
            print(f'  maia-{elo}: sha256 不匹配，拒绝写入（期望 {digest} 实得 {got}）')
            fail += 1
            continue
        tmp = p.with_suffix('.pb.gz.tmp')
        tmp.write_bytes(data)
        tmp.replace(p)
        print(f'  maia-{elo}: {len(data)} 字节 -> {p}')
        ok += 1
    print(f'完成：{ok} 就绪，{fail} 失败')
    if fail:
        raise SystemExit(1)
    print()
    print('用法示例（Kit match 配置）：')
    print('  {"factory": "Kit.players.maia:make_maia_player_factory",')
    print('   "root": "<import 根>",')
    print('   "kwargs": {"weights": "%s", "device": "cpu"}}'
          % (dest / 'maia-1500.pb.gz'))


if __name__ == '__main__':
    sys.exit(main())
