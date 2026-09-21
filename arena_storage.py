"""竞技场历史对弈持久化存储（SQLite 实现）。"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "data" / "arena" / "arena_history.db"


class ArenaStorage:
    def __init__(self, db_path: Path | str = DB_PATH):
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS arena_records (
                            id TEXT PRIMARY KEY,
                            created_at TEXT NOT NULL,
                            end_time TEXT,
                            white_model TEXT NOT NULL,
                            white_arg TEXT,
                            black_model TEXT NOT NULL,
                            black_arg TEXT,
                            moves TEXT,
                            ply_count INTEGER DEFAULT 0,
                            result TEXT,
                            winner TEXT,
                            termination_reason TEXT
                        )
                        """
                    )
            finally:
                conn.close()

    def save_record(self, record: dict[str, Any]) -> None:
        """保存或更新对弈记录。"""
        moves_val = record.get("moves")
        if isinstance(moves_val, (list, tuple)):
            moves_val = " ".join(moves_val)

        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO arena_records (
                            id, created_at, end_time,
                            white_model, white_arg, black_model, black_arg,
                            moves, ply_count, result, winner, termination_reason
                        ) VALUES (
                            :id, :created_at, :end_time,
                            :white_model, :white_arg, :black_model, :black_arg,
                            :moves, :ply_count, :result, :winner, :termination_reason
                        )
                        ON CONFLICT(id) DO UPDATE SET
                            end_time = excluded.end_time,
                            white_model = excluded.white_model,
                            white_arg = excluded.white_arg,
                            black_model = excluded.black_model,
                            black_arg = excluded.black_arg,
                            moves = excluded.moves,
                            ply_count = excluded.ply_count,
                            result = excluded.result,
                            winner = excluded.winner,
                            termination_reason = excluded.termination_reason
                        """,
                        {
                            "id": record["id"],
                            "created_at": record.get("created_at"),
                            "end_time": record.get("end_time"),
                            "white_model": record.get("white_model", ""),
                            "white_arg": record.get("white_arg"),
                            "black_model": record.get("black_model", ""),
                            "black_arg": record.get("black_arg"),
                            "moves": moves_val or "",
                            "ply_count": record.get("ply_count", 0),
                            "result": record.get("result", "*"),
                            "winner": record.get("winner"),
                            "termination_reason": record.get("termination_reason"),
                        },
                    )
            finally:
                conn.close()

    def list_records(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """获取对弈记录列表，按创建时间倒序返回。"""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, created_at, end_time,
                           white_model, white_arg, black_model, black_arg,
                           moves, ply_count, result, winner, termination_reason
                    FROM arena_records
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                )
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.close()

    def get_record(self, record_id: str) -> dict[str, Any] | None:
        """根据 id 查询单条记录。"""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, created_at, end_time,
                           white_model, white_arg, black_model, black_arg,
                           moves, ply_count, result, winner, termination_reason
                    FROM arena_records
                    WHERE id = ?
                    """,
                    (record_id,),
                )
                row = cursor.fetchone()
                return dict(row) if row else None
            finally:
                conn.close()


# 模块级单例
storage = ArenaStorage()
