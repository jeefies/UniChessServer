import json
import time
import urllib.request
from pathlib import Path
import chess.pgn

pgn_path = Path(__file__).resolve().parent.parent / "data" / "sample_human_games.pgn"
with open(pgn_path, "r", encoding="utf-8") as f:
    g1 = chess.pgn.read_game(f)
    g2 = chess.pgn.read_game(f)
    g3 = chess.pgn.read_game(f)

# 用第三局进行完全干净的无缓存测试
moves_g3 = [m.uci() for m in g3.mainline_moves()]
print(f"测试对局 Game 3: {g3.headers.get('White')} vs {g3.headers.get('Black')} | 总步数: {len(moves_g3)} 步")

url = "http://127.0.0.1:8000/sf/v1/review"

for workers in [4, 6, 8]:
    t0 = time.perf_counter()
    req = urllib.request.Request(
        url,
        data=json.dumps({"moves": moves_g3, "profile": "lightning", "concurrency": workers}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer sf-token-123"},
        method="POST",
    )
    # 先清空当前局面缓存以保证每次都是全量冷推演
    # 或者用不同前缀测试
    with urllib.request.urlopen(req, timeout=120.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - t0
    print(f"Workers: {workers:2d} | 耗时: {elapsed:5.2f}s | 吞吐: {data['effectivePliesPerSecond']:5.2f} 步/秒 | 缓存命中: {data['cacheHits']}/{data['totalPlies']}")
