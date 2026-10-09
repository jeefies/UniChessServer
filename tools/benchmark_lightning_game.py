"""Benchmark Stockfish lightning profile across complete chess games."""
from __future__ import annotations

import io
import json
import statistics
import time
import urllib.request
import chess
import chess.pgn

TOKEN = open("/home/jeefy/.config/unichess/access_token").read().strip()
BASE = "http://127.0.0.1:8000"

def post_json(path: str, data: dict) -> tuple[dict, float]:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(data).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Access-Token": TOKEN,
        },
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        elapsed = (time.perf_counter() - t0) * 1000
        res = json.loads(resp.read().decode("utf-8"))
        return res, elapsed

def load_game_moves_from_pgn(pgn_path: str, game_index: int = 0) -> list[str]:
    with open(pgn_path, "r", encoding="utf-8") as f:
        for idx in range(game_index + 1):
            game = chess.pgn.read_game(f)
            if game is None:
                raise ValueError(f"Game index {game_index} not found in {pgn_path}")
            if idx == game_index:
                moves = []
                node = game
                while node.variations:
                    next_node = node.variation(0)
                    moves.append(next_node.move.uci())
                    node = next_node
                return moves
    return []

def run_benchmark():
    # 选取 Transformer/logs/match_verification_6g.pgn 第 2 盘完整对局 (94 步 / 47 回合)
    pgn_path = "/home/jeefy/UniChess/Transformer/logs/match_verification_6g.pgn"
    moves = load_game_moves_from_pgn(pgn_path, game_index=1)
    total_plies = len(moves)
    print(f"=== 加载完整实战对局: 共 {total_plies} 半步 ({total_plies // 2} 回合) ===")
    print("走法前 10 步:", moves[:10])

    records = []
    print(f"\n开始逐步推演分析 (Profile: lightning, 目标深度 22)...")
    print(f"{'Ply':>4} | {'Move':>6} | {'耗时(ms)':>9} | {'CommonDepth':>11} | {'Best':>6} | {'Played':>6} | {'Nodes':>9} | {'NPS(k)':>8} | {'Stage2':>6} | {'Theory':>6}")
    print("-" * 88)

    for i in range(total_plies):
        history = moves[:i]
        played = moves[i]
        req = {
            "requestId": f"bench-light-{i+1}",
            "position": {
                "initialFen": None,
                "moves": history,
            },
            "playedMove": played,
            "profile": "lightning",
        }
        res, wall_ms = post_json("/sf/v1/analyze-move", req)
        
        best_mv = str((res.get("best") or {}).get("move") or "-")
        played_mv = str((res.get("played") or {}).get("move") or "-")
        opening = res.get("opening", {}) or {}
        comp = res.get("comparison", {}) or {}
        stats = res.get("stats", {}) or {}
        
        cd = comp.get("commonDepth", 0)
        nodes = stats.get("nodes") or 0
        nps = (stats.get("nps") or 0) // 1000
        stage2 = (played_mv != best_mv)
        theory = bool(opening.get("theory"))

        rec = {
            "ply": i + 1,
            "move": played,
            "wall_ms": wall_ms,
            "server_ms": stats.get("elapsedMs", 0),
            "common_depth": cd,
            "best_move": best_mv,
            "played_move": played_mv,
            "nodes": nodes,
            "nps": stats.get("nps") or 0,
            "tbhits": stats.get("tbhits") or 0,
            "is_best": (played == best_mv),
            "stage2_needed": stage2,
            "theory": theory,
        }
        records.append(rec)

        print(f"{i+1:>4} | {played:>6} | {wall_ms:>9.1f} | {cd:>11} | {best_mv:>6} | {played_mv:>6} | {nodes:>9} | {nps:>8} | {str(stage2):>6} | {str(theory):>6}", flush=True)

    # 统计数据汇总
    wall_times = [r["wall_ms"] for r in records]
    server_times = [r["server_ms"] for r in records]
    depths = [r["common_depth"] for r in records]
    nodes_list = [r["nodes"] for r in records if r["nodes"] > 0]
    
    # 分阶段统计
    opening_times = [r["wall_ms"] for r in records if r["ply"] <= 20]
    midgame_times = [r["wall_ms"] for r in records if 20 < r["ply"] <= 60]
    endgame_times = [r["wall_ms"] for r in records if r["ply"] > 60]

    # Best-hit vs non-best-hit
    best_hit_times = [r["wall_ms"] for r in records if r["is_best"]]
    non_best_times = [r["wall_ms"] for r in records if not r["is_best"]]

    print("\n" + "=" * 50)
    print("           LIGHTNING 完整对局 BENCHMARK 统计")
    print("=" * 50)
    print(f"总步数 (Plies):          {len(records)}")
    print(f"平均单步耗时 (Mean):      {statistics.mean(wall_times):.1f} ms")
    print(f"中位数耗时 (Median):     {statistics.median(wall_times):.1f} ms")
    print(f"P90 耗时:                {sorted(wall_times)[int(len(wall_times)*0.9)]:.1f} ms")
    print(f"P95 耗时:                {sorted(wall_times)[int(len(wall_times)*0.95)]:.1f} ms")
    print(f"最小耗时 (Min):          {min(wall_times):.1f} ms")
    print(f"最大耗时 (Max):          {max(wall_times):.1f} ms")
    print(f"整盘对局总耗时:          {sum(wall_times)/1000.0:.2f} s")
    print("-" * 50)
    print(f"开局阶段 (Plies 1-20):   平均 {statistics.mean(opening_times):.1f} ms (N={len(opening_times)})")
    print(f"中局阶段 (Plies 21-60):  平均 {statistics.mean(midgame_times):.1f} ms (N={len(midgame_times)})")
    print(f"残局阶段 (Plies 61+):    平均 {statistics.mean(endgame_times):.1f} ms (N={len(endgame_times)})")
    print("-" * 50)
    print(f"实战即最佳招 (Stage 2 早退): 平均 {statistics.mean(best_hit_times):.1f} ms (占比 {len(best_hit_times)}/{len(records)} = {len(best_hit_times)/len(records)*100:.1f}%)")
    print(f"实战非最佳招 (Stage 2 跑满): 平均 {statistics.mean(non_best_times):.1f} ms (占比 {len(non_best_times)}/{len(records)} = {len(non_best_times)/len(records)*100:.1f}%)")
    print("-" * 50)
    print(f"深度达标情况:")
    d22_count = sum(1 for d in depths if d >= 22)
    print(f"深度 >= 22 比例:         {d22_count}/{len(depths)} ({d22_count/len(depths)*100:.1f}%)")
    non_22 = [r for r in records if r["common_depth"] < 22]
    if non_22:
        print(f"未满 22 步数详情 (检查是否为将杀/残局截断):")
        for item in non_22:
            print(f"  Ply {item['ply']} ({item['move']}): depth={item['common_depth']}")
    else:
        print("所有步数 100% 达到深度 22 层！")
    print("=" * 50)

    # 导出 json 供深度分析
    with open("/tmp/lightning_benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_plies": len(records),
                "mean_ms": statistics.mean(wall_times),
                "median_ms": statistics.median(wall_times),
                "p90_ms": sorted(wall_times)[int(len(wall_times)*0.9)],
                "min_ms": min(wall_times),
                "max_ms": max(wall_times),
                "total_seconds": sum(wall_times)/1000.0,
                "opening_mean_ms": statistics.mean(opening_times),
                "midgame_mean_ms": statistics.mean(midgame_times),
                "endgame_mean_ms": statistics.mean(endgame_times),
                "best_hit_mean_ms": statistics.mean(best_hit_times),
                "non_best_mean_ms": statistics.mean(non_best_times),
            },
            "records": records,
        }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    run_benchmark()
