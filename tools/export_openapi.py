"""导出 Stockfish 分析 API OpenAPI 3.1 描述文件到 Server/openapi.json。"""
from __future__ import annotations

import json
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from sf_router import get_sf_openapi  # noqa: E402


def main() -> None:
    schema = get_sf_openapi()
    out_file = root / "openapi.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    print(f"Exported OpenAPI spec to {out_file} ({out_file.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
