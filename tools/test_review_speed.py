#!/usr/bin/env python3
"""测试 /sf/v1/review 批量高并发复盘接口在人类对局下的性能与吞吐。"""

import json
import time
import urllib.request
from pathlib import Path
import chess.pgn

pgn_path = Path(__file__).resolve().parent.parent / "data" / "sample_human_games.pgn"
print("读取人类真实对局...")
with open(pgn_path, "r", encoding="utf-8") as f:
    # 读第一局或第二局
    g1 = chess.pgn.read_game(f)
    g2 = chess.pgn.read_game(f)

# 使用第二局（89 步，King's Indian）
moves_89 = [m.uci() for m in g2.mainline_moves()]
print(f"对局: {g2.headers.get('White')} vs {g2.headers.get('Black')} | 总步数: {len(moves_89)} 步")

url = "http://127.0.0.1:8000/sf/v1/review"
payload = {
    "moves": moves_89,
    "profile": "lightning",
    "concurrency": 4,
}

print(f"\n请求 POST {url} 进行 4-Worker 并发全盘复盘...")
t0 = time.perf_counter()
req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(req, timeout=180.0) as resp:
    data = json.loads(resp.read().decode("utf-8"))

total_elapsed = time.perf_counter() - t0
print("\n" + "="*60)
print("【/sf/v1/review 全盘并发复盘结果】")
print("="*60)
print(f"总半步数:             {data['totalPlies']} 步")
print(f"已分析步数:           {data['analyzedPlies']} 步")
print(f"磁盘缓存命中:         {data['cacheHits']} 步")
print(f"服务端报告耗时:       {data['elapsedMs'] / 1000.0:.2f} 秒")
print(f"客户端总耗时 (含网络): {total_elapsed:.2f} 秒")
print(f"复盘有效吞吐量:       {data['effectivePliesPerSecond']:.2f} 步/秒")

summary = data.get("summary", {})
print("\n--- 对局质量评估摘要 ---")
print(f"白方 ({g2.headers.get('White')}): 准确率 {summary.get('whiteAccuracy')}% | 平均厘兵损失 (ACPL): {summary.get('whiteAcpl')} cp")
print(f"  招法分布: {summary.get('whiteJudgments')}")
print(f"黑方 ({g2.headers.get('Black')}): 准确率 {summary.get('blackAccuracy')}% | 平均厘兵损失 (ACPL): {summary.get('blackAcpl')} cp")
print(f"  招法分布: {summary.get('blackJudgments')}")

print("\n采样前 5 步详细输出:")
for m in data["moves"][:5]:
    cache_tag = "[CACHE]" if m["cached"] else f"[{m['elapsedMs']}ms]"
    print(f"  第 {m['ply']} 步: {m['san']:5s} ({m['move']}) | 最佳: {m['bestMove']} | diff: {m['diffCp']}cp | "
          f"评语: {m['judgment']:10s} | 准确率: {m['accuracy']:5.1f}% | 深度: {m['depth']} | {cache_tag}")

print("="*60)
