import chess
import chess.engine
import time
from pathlib import Path

binary_path = "/home/jeefy/UniChess/Server/tools/stockfish"

# 取一个典型的中局局面（复杂的西西里防御中局）
test_fen = "r1bqkb1r/pp3ppp/2n1pn2/2pp4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 6"
board = chess.Board(test_fen)

print("="*60)
print("测试 Intel Core Ultra 7 265K (20 Cores) 下不同线程数对 Stockfish 19 速度的影响")
print("测试局面: 西西里复杂中局，目标深度: 22 层")
print("="*60)

thread_candidates = [4, 6, 8, 12, 14, 16, 18]

for th in thread_candidates:
    engine = chess.engine.SimpleEngine.popen_uci(binary_path)
    engine.configure({
        "Threads": th,
        "Hash": 1024,
    })
    
    # 预热一次
    engine.analyse(board, chess.engine.Limit(depth=10))
    
    t0 = time.perf_counter()
    info = engine.analyse(board, chess.engine.Limit(depth=22), multipv=1)
    if isinstance(info, list):
        info = info[0]
    elapsed = (time.perf_counter() - t0) * 1000
    
    nodes = info.get("nodes", 0)
    nps = info.get("nps", 0)
    depth = info.get("depth", 0)
    
    print(f"Threads: {th:2d} | 耗时: {elapsed:6.1f} ms | NPS: {nps/1e6:5.2f} Mnps | Nodes: {nodes:8d} | Depth: {depth}")
    engine.quit()

print("="*60)
