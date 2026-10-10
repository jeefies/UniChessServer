#!/usr/bin/env python3
"""深度测试方案 1：在目标深度严格保持 22 层的前提下，
寻找 Core Ultra 7 265K（20 物理核）下最强并发参数组合。

测试变量：
1. 并发 Worker 数量 (Workers: 4, 6, 8, 9, 10, 16)
2. 每个 Worker 的线程分配 (Threads: 1, 2, 3, 4)
3. 长尾防死锁软上限 (max_time_ms: None 纯深度22 / 3000ms / 2000ms / 1500ms)
"""

import copy
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import chess
import chess.pgn

# 确保能导入 Server 模块
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import sf_analyzer


def run_review_cold(
    analyzer: sf_analyzer.StockfishAnalyzer,
    moves: List[str],
    depth: int = 22,
    workers: int = 4,
    threads_per_worker: int = 4,
    max_time_ms: int | None = None,
) -> Dict[str, Any]:
    """使用独立空缓存执行全量冷启动全盘复盘，精准测量算力与耗时。"""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tf:
        tmp_db = tf.name
        
    old_disk_cache = analyzer._disk_cache
    old_cache = analyzer._cache
    
    try:
        import collections
        analyzer._disk_cache = sf_analyzer.DiskCache(tmp_db)
        analyzer._cache = collections.OrderedDict()
        
        # 覆写并发配置
        os.environ["UNICHESS_SF_REVIEW_CONCURRENCY"] = str(workers)
        os.environ["UNICHESS_SF_REVIEW_THREADS"] = str(threads_per_worker)
        
        t0 = time.perf_counter()
        orig_prof = copy.deepcopy(sf_analyzer.PROFILES.get("lightning", {}))
        sf_analyzer.PROFILES["lightning"]["depth"] = depth
        sf_analyzer.PROFILES["lightning"]["maxTimeMs"] = max_time_ms
        
        res = analyzer.review_game(
            initial_fen=None,
            moves=moves,
            profile="lightning",
            concurrency=workers,
            threads_per_worker=threads_per_worker,
        )
        total_sec = time.perf_counter() - t0
        
        depths = [m["depth"] for m in res["moves"]]
        latencies = [m["elapsedMs"] for m in res["moves"]]
        
        return {
            "workers": workers,
            "threads": threads_per_worker,
            "total_threads": workers * threads_per_worker,
            "max_time_ms": max_time_ms,
            "total_sec": total_sec,
            "tps": len(moves) / total_sec,
            "min_depth": min(depths) if depths else 0,
            "avg_depth": sum(depths) / len(depths) if depths else 0,
            "max_move_ms": max(latencies) if latencies else 0,
            "avg_move_ms": sum(latencies) / len(latencies) if latencies else 0,
            "accuracy_w": res["summary"]["whiteAccuracy"],
            "accuracy_b": res["summary"]["blackAccuracy"],
        }
    finally:
        sf_analyzer.PROFILES["lightning"] = orig_prof
        analyzer._disk_cache = old_disk_cache
        analyzer._cache = old_cache
        if os.path.isfile(tmp_db):
            try:
                os.remove(tmp_db)
            except Exception:
                pass


def main():
    pgn_path = ROOT / "data" / "sample_human_games.pgn"
    with open(pgn_path, "r", encoding="utf-8") as f:
        g1 = chess.pgn.read_game(f)
        g2 = chess.pgn.read_game(f)  # 89 步王兵印度防御
        
    moves_89 = [m.uci() for m in g2.mainline_moves()]
    print(f"测试对局: {g2.headers.get('White')} vs {g2.headers.get('Black')} (总长: {len(moves_89)} 步)", flush=True)
    print(f"宿主 CPU: Intel Core Ultra 7 265K (物理核心: {os.cpu_count()})", flush=True)
    print("=" * 85, flush=True)
    print("目标：深度严格锁定 22 层，对比不同 Worker 数、单 Worker 线程数与长尾软截断耗时", flush=True)
    print("=" * 85, flush=True)
    
    analyzer = sf_analyzer.get_analyzer()
    
    # 评测矩阵设计：
    # 1. 纯深度 22（不限时），探索不同 W 与 T 组合：
    #    (4W x 4T = 16 线程, 基线)
    #    (6W x 3T = 18 线程)
    #    (8W x 2T = 16 线程)
    #    (9W x 2T = 18 线程)
    #    (10W x 2T = 20 线程, 100% 满核)
    #    (16W x 1T = 16 线程, 单线程零 SMP 开销)
    # 2. 探索不同长尾保护时限 (2.0s / 1.5s) 在最佳并发下的边际收益
    combinations = [
        {"workers": 4, "threads": 4, "max_time_ms": None, "desc": "基线: 4W x 4T (16核, 纯深22)"},
        {"workers": 6, "threads": 3, "max_time_ms": None, "desc": "6W x 3T (18核, 纯深22)"},
        {"workers": 8, "threads": 2, "max_time_ms": None, "desc": "8W x 2T (16核, 纯深22)"},
        {"workers": 9, "threads": 2, "max_time_ms": None, "desc": "9W x 2T (18核, 纯深22)"},
        {"workers": 10, "threads": 2, "max_time_ms": None, "desc": "10W x 2T (20核满载, 纯深22)"},
        {"workers": 12, "threads": 1, "max_time_ms": None, "desc": "12W x 1T (12核单线程, 纯深22)"},
        {"workers": 16, "threads": 1, "max_time_ms": None, "desc": "16W x 1T (16核单线程, 纯深22)"},
        {"workers": 18, "threads": 1, "max_time_ms": None, "desc": "18W x 1T (18核单线程, 纯深22)"},
        {"workers": 9, "threads": 2, "max_time_ms": 2000, "desc": "9W x 2T (18核, 2.0s 截断)"},
        {"workers": 16, "threads": 1, "max_time_ms": 2000, "desc": "16W x 1T (16核, 2.0s 截断)"},
        {"workers": 18, "threads": 1, "max_time_ms": 2000, "desc": "18W x 1T (18核, 2.0s 截断)"},
        {"workers": 18, "threads": 1, "max_time_ms": 1500, "desc": "18W x 1T (18核, 1.5s 截断)"},
    ]
    
    results = []
    base_time = None
    
    for c in combinations:
        w = c["workers"]
        th = c["threads"]
        t_limit = c["max_time_ms"]
        desc = c["desc"]
        print(f"\n>>> 正在运行: {desc} ...", flush=True)
        
        stat = run_review_cold(
            analyzer,
            moves=moves_89,
            depth=22,
            workers=w,
            threads_per_worker=th,
            max_time_ms=t_limit,
        )
        stat["desc"] = desc
        results.append(stat)
        if base_time is None:
            base_time = stat["total_sec"]
            
        speedup = base_time / stat["total_sec"]
        print(f"    完成! 耗时: {stat['total_sec']:5.1f}s | 吞吐: {stat['tps']:4.2f} 步/s | "
              f"最慢单步: {stat['max_move_ms']:5d}ms | 平均深度: {stat['avg_depth']:4.1f} (最低 {stat['min_depth']}) | 提速: {speedup:4.2f}x",
              flush=True)
        
    print("\n" + "=" * 105, flush=True)
    print(f"{'配置方案描述':<32} | {'总耗时(秒)':<9} | {'吞吐量(步/s)':<11} | {'最慢单步':<9} | {'平均深度':<8} | {'白方Acc':<7} | {'黑方Acc':<7} | {'相对提速':<6}", flush=True)
    print("-" * 105, flush=True)
    
    for r in results:
        speedup = base_time / r["total_sec"]
        print(f"{r['desc']:<32} | {r['total_sec']:8.1f}s | {r['tps']:9.2f} 步/s | {r['max_move_ms']:7d}ms | "
              f"{r['avg_depth']:7.1f}  | {r['accuracy_w']:6.1f}% | {r['accuracy_b']:6.1f}% | {speedup:5.2f}x", flush=True)
    print("=" * 105, flush=True)
    
    out_file = ROOT / "data" / "review_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n完整测试数据已落盘至: {out_file}", flush=True)


if __name__ == "__main__":
    main()
