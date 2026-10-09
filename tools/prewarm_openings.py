"""标准扩展级开局谱预热工具：将常用开局前 6~10 步批量推演并持久化至 SQLite 磁盘缓存。

特性：
- 涵盖 Kit/data/openings.txt 与 SSM/data/openings_200.txt（逾 200 条标准 ECO 开局体系）；
- 去重后提取各开局前 N 步走法；
- 批量使用 lightning 模式（深度 22）推演并存入 sf_cache.sqlite；
- 零 RAM 常驻：数据直接落盘，运行时按需单页检索，不增加服务常驻内存；
- 推演完成后主动释放 Stockfish 进程，释放全部 1.5GB 内存。
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import chess

# 保证可直接加载 Server 根目录与 Kit
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
IMPORT_ROOT = ROOT.parent
sys.path.insert(0, str(IMPORT_ROOT))

import sf_analyzer


def load_opening_lines() -> list[list[str]]:
    """加载并合并全部标准开局着法序列（UCI 列表）。"""
    lines: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    sources = [
        ROOT.parent / "Kit" / "data" / "openings.txt",
        ROOT.parent / "SSM" / "data" / "openings_200.txt",
    ]

    for p in sources:
        if not p.is_file():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line or raw_line.startswith("#"):
                    continue
                clean = raw_line.split("#")[0].strip()
                tokens = clean.split()
                if not tokens:
                    continue

                # 兼容 SAN 与 UCI
                board = chess.Board()
                uci_moves: list[str] = []
                valid = True
                for tok in tokens:
                    move = None
                    try:
                        cand = chess.Move.from_uci(tok)
                        if cand in board.legal_moves:
                            move = cand
                    except Exception:
                        pass
                    if move is None:
                        try:
                            move = board.parse_san(tok)
                        except Exception:
                            valid = False
                            break
                    if move and move in board.legal_moves:
                        uci_moves.append(move.uci())
                        board.push(move)
                    else:
                        valid = False
                        break

                if valid and uci_moves:
                    tup = tuple(uci_moves)
                    if tup not in seen:
                        seen.add(tup)
                        lines.append(uci_moves)

    print(f"成功加载并解析去重后标准开局谱线: 共 {len(lines)} 条")
    return lines


def extract_positions(
    lines: list[list[str]], max_plies: int = 8
) -> list[tuple[list[str], str]]:
    """从各开局线中提取前 max_plies 步的历史与目标着法二元组 (moves_history, played_move)。"""
    seen: set[tuple[tuple[str, ...], str]] = set()
    pairs: list[tuple[list[str], str]] = []

    for line in lines:
        limit = min(len(line), max_plies)
        for i in range(limit):
            history = line[:i]
            played = line[i]
            key = (tuple(history), played)
            if key not in seen:
                seen.add(key)
                pairs.append((history, played))

    print(f"提取前 {max_plies} 步去重局面节点数: 共 {len(pairs)} 个")
    return pairs


def prewarm(max_plies: int = 8, limit: int | None = None, dry_run: bool = False):
    lines = load_opening_lines()
    pairs = extract_positions(lines, max_plies=max_plies)

    if limit is not None and limit > 0:
        pairs = pairs[:limit]
        print(f"根据限制截取前 {len(pairs)} 个待预热局面")

    analyzer = sf_analyzer.get_analyzer()
    disk_cache = analyzer._disk_cache
    print(f"SQLite 缓存数据库: {disk_cache.db_path} (当前已有 {disk_cache.count()} 条记录)")

    if dry_run:
        print("Dry run 模式，跳过实际推演。")
        return

    hit_count = 0
    computed_count = 0
    t0 = time.perf_counter()

    print(f"\n开始批量预热推演 (Profile: lightning, depth=22)...")
    try:
        for idx, (history, played) in enumerate(pairs, 1):
            cache_key = (
                "analyze_move",
                "",
                tuple(history),
                played,
                22,
                None,
                1,
                sf_analyzer.DEFAULT_MAX_PV_PLIES,
                analyzer._get_identity(),
            )

            # 先检查磁盘缓存是否已有
            if disk_cache.get(cache_key) is not None:
                hit_count += 1
                if idx % 20 == 0 or idx == len(pairs):
                    print(f"[{idx}/{len(pairs)}] 已存在 (跳过) | 历史/实战: {len(history)}步 {played}")
                continue

            # 未命中则调用 analyzer 进行推演
            step_t0 = time.perf_counter()
            analyzer.analyze_move(
                initial_fen=None,
                moves=history,
                played_move=played,
                profile="lightning",
                depth=22,
            )
            step_ms = (time.perf_counter() - step_t0) * 1000
            computed_count += 1

            if computed_count % 5 == 0 or idx == len(pairs):
                elapsed = time.perf_counter() - t0
                print(
                    f"[{idx}/{len(pairs)}] 耗时 {step_ms:.0f}ms | 历史: {len(history)}步, 招法: {played} | "
                    f"总推演: {computed_count}, 已跳过: {hit_count}, 累计: {elapsed:.1f}s"
                )
    finally:
        # 预热结束必须主动释放 Stockfish 引擎，归还全部 1.5GB 内存！
        print("\n预热完毕或中断，正在释放 Stockfish 引擎子进程...")
        freed = analyzer.release_idle_engines(force=True)
        print(f"Stockfish 引擎已释放: {freed}，当前磁盘缓存总条数: {disk_cache.count()}")

    total_time = time.perf_counter() - t0
    print("\n" + "=" * 50)
    print("           开局持久化缓存预热完成")
    print("=" * 50)
    print(f"处理局面总数:    {len(pairs)}")
    print(f"本次新推演入库:  {computed_count}")
    print(f"命中已有缓存:    {hit_count}")
    print(f"当前磁盘数据库:  {disk_cache.db_path}")
    print(f"数据库记录总数:  {disk_cache.count()} 条")
    print(f"总耗时:          {total_time:.1f} s")
    print(f"常驻 RAM 占用:   0 MB (完全落盘，引擎进程已关闭)")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prewarm chess openings into SQLite disk cache.")
    parser.add_argument("--max-plies", type=int, default=8, help="最大半步深度 (默认 8 半步 / 4 回合)")
    parser.add_argument("--limit", type=int, default=None, help="最大处理局面数上限")
    parser.add_argument("--dry-run", action="store_true", help="仅分析统计局面，不执行推演")
    args = parser.parse_args()

    prewarm(max_plies=args.max_plies, limit=args.limit, dry_run=args.dry_run)
