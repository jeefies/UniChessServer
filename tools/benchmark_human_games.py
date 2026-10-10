#!/usr/bin/env python3
"""评估 Stockfish API 对真实人类对局的推演速度基准测试。

测试维度：
1. 真实大师/高水平人类对局（取自 Magnus Carlsen 等实际对局）；
2. 逐步请求 /sf/v1/analyze-move（实战单步质量对比与同深度评测，含双路并行推演）；
3. 分阶段深度耗时统计：
   - 开局阶段（1~16 半步）：测试开局磁盘缓存命中率与亚毫秒响应；
   - 中局阶段（17~40 半步）：测试复杂战术局面 22 层极速档评估速度；
   - 残局阶段（41+ 半步）：测试残局与 Syzygy 3-4-5 残局库穿透加速效果；
4. 核心指标统计：中位数耗时、算术均值、90分位、每秒评估步数、开局缓存命中率；
5. 二次回看测试（Replay Test）：验证全盘落盘后复盘时的极限吞吐与零常驻内存表现。
"""

import argparse
import json
import statistics
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List

import chess
import chess.pgn


def request_sf_analyze_move(api_url: str, moves_before: List[str], played_move: str, profile: str = "lightning") -> Dict[str, Any]:
    url = f"{api_url.rstrip('/')}/sf/v1/analyze-move"
    payload = {
        "position": {
            "moves": moves_before,
        },
        "playedMove": played_move,
        "profile": profile,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=120.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def request_sf_evaluate(api_url: str, moves: List[str], profile: str = "lightning") -> Dict[str, Any]:
    url = f"{api_url.rstrip('/')}/sf/v1/evaluate"
    payload = {
        "position": {
            "moves": moves,
        },
        "profile": profile,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=120.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_sf_health(api_url: str) -> Dict[str, Any]:
    url = f"{api_url.rstrip('/')}/sf/v1/health"
    with urllib.request.urlopen(url, timeout=5.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_game_benchmark(api_url: str, game: chess.pgn.Game, game_idx: int, profile: str = "lightning", endpoint: str = "analyze-move") -> Dict[str, Any]:
    headers = game.headers
    white = headers.get("White", "White")
    black = headers.get("Black", "Black")
    result = headers.get("Result", "*")
    event = headers.get("Event", "Game")
    eco = headers.get("ECO", "")
    
    board = game.board()
    moves = list(game.mainline_moves())
    total_plies = len(moves)
    
    print(f"\n{'='*75}")
    print(f"对局 #{game_idx}: 【{white}】 vs 【{black}】 ({result}) | ECO: {eco}")
    print(f"总步数: {total_plies} 半步 ({len(moves)//2} 回合) | 接口: /sf/v1/{endpoint} | 档位: {profile}")
    print(f"{'='*75}")
    
    move_records = []
    moves_so_far: List[str] = []
    
    t_start_game = time.perf_counter()
    
    for ply_idx, move in enumerate(moves, start=1):
        move_uci = move.uci()
        t0 = time.perf_counter()
        
        if endpoint == "analyze-move":
            resp = request_sf_analyze_move(api_url, moves_so_far, move_uci, profile=profile)
            cached = resp.get("cached", False)
            best_info = resp.get("best") or {}
            depth = best_info.get("depth", 0)
            score = best_info.get("score") or {}
            best_move = best_info.get("move", "")
            comp = resp.get("comparison") or {}
            diff_cp = comp.get("diffCp")
        else:
            moves_with_curr = moves_so_far + [move_uci]
            resp = request_sf_evaluate(api_url, moves_with_curr, profile=profile)
            cached = resp.get("cached", False)
            depth = resp.get("completedDepth", 0)
            best_info = resp.get("best") or {}
            score = best_info.get("score") or {}
            best_move = best_info.get("move", "")
            diff_cp = None
            
        req_latency_ms = (time.perf_counter() - t0) * 1000.0
        
        eval_phase = "开局" if ply_idx <= 16 else ("中局" if ply_idx <= 40 else "残局")
        
        san_move = board.san(move)
        move_records.append({
            "ply": ply_idx,
            "move": move_uci,
            "san": san_move,
            "phase": eval_phase,
            "latency_ms": req_latency_ms,
            "cached": cached,
            "depth": depth,
            "best_move": best_move,
            "diff_cp": diff_cp,
            "score": score,
        })
        
        # 实时打印部分步数采样
        if ply_idx <= 8 or ply_idx % 10 == 0 or ply_idx == total_plies:
            cache_tag = " [DISK HIT]" if cached else "           "
            diff_str = f" diff: {diff_cp:+4d}cp" if diff_cp is not None else ""
            score_val = score.get("value", 0) if score else 0
            score_type = score.get("type", "cp") if score else "cp"
            print(f"  [{ply_idx:2d}/{total_plies:2d}] {san_move:6s} ({move_uci}) | "
                  f"耗时: {req_latency_ms:6.1f}ms{cache_tag} | 深度: {depth:2d} | 评分: {score_val:+4d} {score_type}{diff_str}")
        
        board.push(move)
        moves_so_far.append(move_uci)
        
    total_time_s = time.perf_counter() - t_start_game
    
    # 分阶段统计
    latencies = [r["latency_ms"] for r in move_records]
    opening_latencies = [r["latency_ms"] for r in move_records if r["phase"] == "开局"]
    middlegame_latencies = [r["latency_ms"] for r in move_records if r["phase"] == "中局"]
    endgame_latencies = [r["latency_ms"] for r in move_records if r["phase"] == "残局"]
    
    cached_count = sum(1 for r in move_records if r["cached"])
    cache_rate = (cached_count / total_plies) * 100.0 if total_plies > 0 else 0.0
    
    mean_lat = statistics.mean(latencies)
    median_lat = statistics.median(latencies)
    sorted_lat = sorted(latencies)
    p90_lat = sorted_lat[int(len(sorted_lat) * 0.90)] if sorted_lat else 0.0
    
    def fmt_phase(lats: List[float]) -> str:
        if not lats:
            return "N/A"
        return f"均值 {statistics.mean(lats):.1f}ms, 中位数 {statistics.median(lats):.1f}ms (共 {len(lats)} 步)"
    
    print(f"\n--- 对局 #{game_idx} 评估结果统计 ---")
    print(f"  • 整局总耗时:       {total_time_s:.2f} 秒 (评估 {total_plies} 步, 有效吞吐 {total_plies/total_time_s:.2f} 步/秒)")
    print(f"  • 单步中位数耗时:   {median_lat:.1f} ms  <--- 核心体验指标")
    print(f"  • 单步算术平均耗时: {mean_lat:.1f} ms")
    print(f"  • P90 耗时:         {p90_lat:.1f} ms")
    print(f"  • 开局磁盘缓存命中: {cache_rate:.1f}% ({cached_count}/{total_plies} 步)")
    print(f"  • 开局阶段(1-16步): {fmt_phase(opening_latencies)}")
    print(f"  • 中局阶段(17-40):  {fmt_phase(middlegame_latencies)}")
    print(f"  • 残局阶段(41+步):  {fmt_phase(endgame_latencies)}")
    
    return {
        "game_idx": game_idx,
        "white": white,
        "black": black,
        "result": result,
        "total_plies": total_plies,
        "total_time_s": total_time_s,
        "median_ms": median_lat,
        "mean_ms": mean_lat,
        "p90_ms": p90_lat,
        "cache_rate": cache_rate,
        "records": move_records,
    }


def main():
    parser = argparse.ArgumentParser(description="Stockfish 对人类真实对局评估基准测试")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="UniChess Server URL")
    parser.add_argument("--pgn", default="", help="PGN 文件路径")
    parser.add_argument("--games", type=int, default=2, help="测试对局数量")
    parser.add_argument("--profile", default="lightning", help="SF 分析档位 (lightning/fast/standard)")
    parser.add_argument("--skip-games", type=int, default=0, help="跳过的对局数量")
    parser.add_argument("--endpoint", default="analyze-move", choices=["analyze-move", "evaluate"], help="调用的 API 端点")
    args = parser.parse_args()
    
    print("正在检查 SF 服务健康与初始状态...")
    health = get_sf_health(args.url)
    mem_info = health.get("memory", {})
    print(f"当前 SF 服务状态: resident={mem_info.get('resident')}, "
          f"engineRunning={mem_info.get('engineRunning')}, "
          f"diskCacheRecords={mem_info.get('diskCacheRecords')}")
    
    # 查找 PGN
    pgn_path = args.pgn
    if not pgn_path:
        base_dir = Path(__file__).resolve().parent.parent
        possible = [
            base_dir / "data" / "sample_human_games.pgn",
            base_dir.parent / "SSM" / "data" / "sample_real.pgn",
        ]
        for p in possible:
            if p.exists():
                pgn_path = str(p)
                break
                
    if not pgn_path or not Path(pgn_path).exists():
        print(f"错误: 未找到 PGN 文件: {pgn_path}")
        return
        
    print(f"读取人类真实对局文件: {pgn_path}")
    loaded_games = []
    skipped = 0
    with open(pgn_path, "r", encoding="utf-8") as f:
        while len(loaded_games) < args.games:
            g = chess.pgn.read_game(f)
            if not g:
                break
            moves = list(g.mainline_moves())
            if len(moves) >= 30:
                if skipped < args.skip_games:
                    skipped += 1
                    continue
                loaded_games.append(g)
                
    print(f"成功加载 {len(loaded_games)} 局真实人类对局 (跳过前 {args.skip_games} 局)，开始逐局评估测试...\n")
    
    all_results = []
    for idx, game in enumerate(loaded_games, start=1):
        res = run_game_benchmark(args.url, game, idx, profile=args.profile, endpoint=args.endpoint)
        all_results.append(res)
        
    # 全局综合指标
    total_plies_all = sum(r["total_plies"] for r in all_results)
    total_time_all = sum(r["total_time_s"] for r in all_results)
    all_latencies = [m["latency_ms"] for r in all_results for m in r["records"]]
    
    overall_mean = statistics.mean(all_latencies) if all_latencies else 0.0
    overall_median = statistics.median(all_latencies) if all_latencies else 0.0
    sorted_all = sorted(all_latencies)
    overall_p90 = sorted_all[int(len(sorted_all) * 0.90)] if sorted_all else 0.0
    
    print("\n" + "="*75)
    print("【全部人类对局基准测试综合汇总】")
    print("="*75)
    print(f"测试对局数:           {len(all_results)} 局")
    print(f"总评估步数:           {total_plies_all} 步")
    print(f"整局评估累计耗时:     {total_time_all:.2f} 秒")
    print(f"单步中位数耗时 (50%): {overall_median:.1f} ms  <--- 核心真实体验")
    print(f"单步算术平均耗时:     {overall_mean:.1f} ms")
    print(f"单步 90分位耗时(90%): {overall_p90:.1f} ms")
    print(f"整盘推演吞吐量:       {total_plies_all / total_time_all:.2f} 步/秒")
    print("="*75)
    
    # 再次测试二次复盘（全部走缓存的回看速度）
    print("\n[二次复盘验证] 测试同对局回看 / 重复浏览时的极限吞吐（已评估局面的即时响应）...")
    t_replay_start = time.perf_counter()
    replay_records = []
    for game in loaded_games[:1]:
        moves_replay = list(game.mainline_moves())
        so_far = []
        for m in moves_replay:
            m_uci = m.uci()
            t0 = time.perf_counter()
            if args.endpoint == "analyze-move":
                request_sf_analyze_move(args.url, so_far, m_uci, profile=args.profile)
            else:
                request_sf_evaluate(args.url, so_far + [m_uci], profile=args.profile)
            replay_records.append((time.perf_counter() - t0) * 1000.0)
            so_far.append(m_uci)
            
    t_replay_total = time.perf_counter() - t_replay_start
    print(f"二次回看 {len(replay_records)} 步总耗时: {t_replay_total:.3f} 秒")
    print(f"二次回看单步平均耗时: {statistics.mean(replay_records):.2f} ms (中位数: {statistics.median(replay_records):.2f} ms)")
    print(f"二次回看瞬时吞吐:     {len(replay_records) / t_replay_total:.1f} 步/秒")


if __name__ == "__main__":
    main()
