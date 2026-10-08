#!/usr/bin/env python3
"""下载并校验官方 Stockfish 二进制到 Server/tools/（二进制本身不入 git）。

背景：SF 是 UniChessServer 的 baseline 引擎（`models/SF`），GPLv3、单文件约 100MB，
按项目纪律**不进任何 git 仓库**；这个脚本只负责"从官方 release 拉正确的那一份"。

用法（在 Server 目录下）：

    python tools/fetch_stockfish.py            # 下载 + 校验 + 安装到 tools/
    python tools/fetch_stockfish.py --check    # 只校验已安装的二进制
    python tools/fetch_stockfish.py --dest DIR # 装到别处（再配 UNICHESS_STOCKFISH_BIN）

sha256 与 GitHub release（官方仓库 official-stockfish/Stockfish）资产 digest 一致：
换版本时先核对 release 页，再同步下面的 _RELEASE / _ASSETS。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

_RELEASE = 'sf_19'                    # GitHub release tag（Stockfish 19）
_EXPECT_MAJOR = '19'                  # 安装后自报版本的主号，--check 用它做判据
_BASE = f'https://github.com/official-stockfish/Stockfish/releases/download/{_RELEASE}/'
_ASSETS = {
    # (文件名): sha256
    'stockfish-linux-x86-64-universal.tar.gz': '9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f',
    'stockfish-linux-arm64-universal.tar.gz': 'fe26cfd1d9db4c8af3d21e24d9ff34cacb31c1f940085a7583da11796f2bac01',
    'stockfish-macos-universal.tar.gz': 'a1f0e3bcc5a6927a11fe6fc8e54a779754645f3c2bae2cf13420fd1957adaa77',
    'stockfish-windows-x86-64-universal.zip': '3c8bf1f9ea66a09350a40df4f632288285ac206d99f33ab5842c408fc30b48a7',
    'stockfish-windows-arm64-universal.zip': '8372ad3f0d7276deb2c70f801f541ec7db463219fc6d9c7592864e542aa4f401',
}

_SERVER_ROOT = Path(__file__).resolve().parent.parent


def _target() -> str:
    machine = platform.machine().lower()
    system = platform.system().lower()
    if system == 'windows':
        if machine in ('amd64', 'x86_64'):
            return 'stockfish-windows-x86-64-universal.zip'
        if machine in ('arm64', 'aarch64'):
            return 'stockfish-windows-arm64-universal.zip'
    elif system == 'linux':
        if machine in ('x86_64', 'amd64'):
            return 'stockfish-linux-x86-64-universal.tar.gz'
        if machine in ('aarch64', 'arm64'):
            return 'stockfish-linux-arm64-universal.tar.gz'
    elif system == 'darwin':
        return 'stockfish-macos-universal.tar.gz'
    raise SystemExit(f'暂不支持的平台 {system}/{machine}；请手动下载 {_BASE} 后放至 '
                     f'tools/stockfish，或设 UNICHESS_STOCKFISH_BIN 指向已有二进制')


def _dest_name() -> str:
    return 'stockfish.exe' if os.name == 'nt' else 'stockfish'


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _probe_version(binary: Path) -> str | None:
    """跑一次 `uci` 拿 `id name`：跨平台通用的"装的是不是这一版"判据。

    解压出来的可执行文件与 release 资产 digest 不是同一个对象（资产是压缩包），
    所以校验分两层：下载时比对压缩包 sha256（下表 _ASSETS），安装后比对引擎自报版本。
    """
    import subprocess
    try:
        out = subprocess.run([str(binary), 'uci'], capture_output=True,
                             text=True, timeout=30, check=True).stdout
    except Exception as exc:
        print(f'无法运行 {binary}: {exc}')
        return None
    for line in out.splitlines():
        if line.startswith('id name'):
            return line[len('id name'):].strip()
    return None


def _top_level_files(archive: Path):
    """列出压缩包里 `stockfish/` 顶层目录下的普通文件（README/src/wiki/scripts 都在这里）。

    release 资产的可执行文件名带平台后缀（`stockfish-linux-x86-64-universal`、
    `stockfish-windows-x86-64-universal.exe`），不是裸 `stockfish`。
    """
    root = 'stockfish'
    entries: list[tuple[str, int, int]] = []
    if archive.suffix == '.zip':
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                parts = PurePosixPath(info.filename).parts
                if len(parts) == 2 and parts[0] == root:
                    entries.append((info.filename, info.file_size, 0o644))
    else:
        with tarfile.open(archive, 'r:gz') as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                parts = PurePosixPath(member.name).parts
                if len(parts) == 2 and parts[0] == root:
                    entries.append((member.name, member.size, member.mode))
    return entries


def _pick_binary(entries) -> str:
    """从顶层文件里挑出引擎可执行文件。

    Windows zip 没有权限位，先认 `.exe`；tar 认执行位；最后兜底取最大的
    （二进制 100MB+，同目录其余文件都是几十 KB 的文本）。
    """
    if not entries:
        raise SystemExit('压缩包里没找到 stockfish 顶层文件（结构变了？）')
    executables = [e for e in entries if e[0].endswith('.exe')]
    if executables:
        return max(executables, key=lambda e: e[1])[0]
    for name, _, mode in entries:
        if mode & 0o111 and not name.endswith('.sh'):
            return name
    for name, size, _ in sorted(entries, key=lambda e: -e[1]):
        if not name.endswith('.sh'):
            return name
    raise SystemExit('压缩包里没找到可执行文件（结构变了？）')


def _extract_binary(archive: Path, workdir: Path) -> Path:
    entries = _top_level_files(archive)
    target = _pick_binary(entries)
    if archive.suffix == '.zip':
        with zipfile.ZipFile(archive) as zf:
            out = workdir / PurePosixPath(target).name
            with zf.open(target) as src, out.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            return out
    with tarfile.open(archive, 'r:gz') as tf:
        tf.extract(target, workdir)
        return workdir / target


def _download(url: str, dest: Path) -> None:
    print(f'下载 {url}')
    with urllib.request.urlopen(url) as resp, dest.open('wb') as out:
        shutil.copyfileobj(resp, out)


def main() -> int:
    parser = argparse.ArgumentParser(description='安装官方 Stockfish 二进制（下载+校验）')
    parser.add_argument('--dest', default=str(_SERVER_ROOT / 'tools'),
                        help='安装目录（默认 Server/tools）')
    parser.add_argument('--check', action='store_true',
                        help='只校验已安装的二进制，不下载')
    args = parser.parse_args()

    dest_dir = Path(args.dest).resolve()
    installed = dest_dir / _dest_name()
    want = _ASSETS[_target()]

    if args.check:
        if not installed.is_file():
            print(f'未安装：{installed}')
            return 1
        name = _probe_version(installed)
        if name is None or not name.startswith(f'Stockfish {_EXPECT_MAJOR}'):
            print(f'校验失败：{installed} 自报引擎 {name!r}（期望 Stockfish {_EXPECT_MAJOR}，'
                  f'release {_RELEASE}）')
            return 1
        print(f'校验通过：{installed} → {name}（release {_RELEASE}）')
        return 0

    if installed.is_file() and _probe_version(installed) == f'Stockfish {_EXPECT_MAJOR}':
        print(f'已是最新：{installed}（Stockfish {_EXPECT_MAJOR}）')
        return 0

    dest_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        archive = tmpdir / _target()
        _download(_BASE + _target(), archive)
        got = _sha256(archive)
        if got != want:
            print(f'下载内容校验失败：期望 {want}，实际 {got}（丢弃）')
            return 1
        print(f'sha256 校验通过：{got}')
        binary = _extract_binary(archive, tmpdir)
        if binary.name != _dest_name():
            binary = binary.rename(tmpdir / _dest_name())
        installed.unlink(missing_ok=True)
        shutil.copy2(binary, installed)
    if os.name != 'nt':
        installed.chmod(0o755)
    print(f'安装完成：{installed}（release {_RELEASE}）')
    print('提示：二进制不入 git；换机器部署时重跑本脚本即可，'
          '或用 UNICHESS_STOCKFISH_BIN 指向已有二进制。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
