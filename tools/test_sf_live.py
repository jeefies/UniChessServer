import json
import urllib.request
import os

token_file = os.path.expanduser("~/.config/unichess/access_token")
token = ""
if os.path.isfile(token_file):
    with open(token_file, "r", encoding="utf-8") as f:
        token = f.read().strip()

data = {
    "requestId": "live-test-001",
    "position": {
        "initialFen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "moves": ["e2e4", "e7e5", "g1f3", "b8c6"]
    },
    "playedMove": "f1c4",
    "profile": "fast",
    "limits": {
        "depth": 128,
        "maxTimeMs": 1500
    },
    "multiPv": 2,
    "maxPvPlies": 10
}

req = urllib.request.Request(
    "http://127.0.0.1:8000/sf/v1/analyze-move",
    data=json.dumps(data).encode("utf-8"),
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}" if token else ""
    }
)

with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode("utf-8"))
    print("STATUS:", resp.status)
    print("REQUEST ID:", res.get("requestId"))
    print("BEST:", res.get("best", {}).get("move"), "depth:", res.get("best", {}).get("depth"), "score:", res.get("best", {}).get("score"))
    print("PLAYED:", res.get("played", {}).get("move"), "depth:", res.get("played", {}).get("depth"), "score:", res.get("played", {}).get("score"))
    print("COMMON DEPTH:", res.get("comparison", {}).get("commonDepth"))
    print("DIFF CP:", res.get("comparison", {}).get("diffCp"))
    print("STATS:", res.get("stats"))
