"""竞技场历史对弈持久化存储（SQLite 实现）。"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "data" / "arena" / "arena_history.db"
BATCH_ANY = "__batch__"


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
                            termination_reason TEXT,
                            batch_id TEXT
                        )
                        """
                    )
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS arena_batches (
                            id TEXT PRIMARY KEY,
                            created_at TEXT NOT NULL,
                            end_time TEXT,
                            white_model TEXT NOT NULL,
                            white_arg TEXT,
                            black_model TEXT NOT NULL,
                            black_arg TEXT,
                            rounds_planned INTEGER NOT NULL,
                            rounds_completed INTEGER DEFAULT 0,
                            status TEXT NOT NULL,
                            tally_json TEXT DEFAULT '{}',
                            error TEXT
                        )
                        """
                    )
                    self._migrate(conn)
            finally:
                conn.close()

    def _migrate(self, conn) -> None:
        """兼容旧库：为已有 arena_records 补加 batch_id 列。"""
        try:
            cols = {r['name'] for r in conn.execute("PRAGMA table_info(arena_records)").fetchall()}
        except Exception:
            return
        if 'batch_id' not in cols:
            try:
                conn.execute("ALTER TABLE arena_records ADD COLUMN batch_id TEXT")
            except Exception:
                pass

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
                            moves, ply_count, result, winner, termination_reason,
                            batch_id
                        ) VALUES (
                            :id, :created_at, :end_time,
                            :white_model, :white_arg, :black_model, :black_arg,
                            :moves, :ply_count, :result, :winner, :termination_reason,
                            :batch_id
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
                            termination_reason = excluded.termination_reason,
                            batch_id = excluded.batch_id
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
                            "batch_id": record.get("batch_id"),
                        },
                    )
            finally:
                conn.close()

    def list_records(self, limit: int = 50, offset: int = 0, batch_id: str | None = None) -> list[dict[str, Any]]:
        """获取对弈记录列表，按创建时间倒序返回。

        batch_id：None 不过滤；"__batch__" 只要批量对弈的局；"" 只要非批量（观战）的局；
        其它值按批次 id 精确匹配。
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                sql = """
                    SELECT id, created_at, end_time,
                           white_model, white_arg, black_model, black_arg,
                           moves, ply_count, result, winner, termination_reason,
                           batch_id
                    FROM arena_records
                """
                params: list[Any] = []
                if batch_id == BATCH_ANY:
                    sql += " WHERE batch_id IS NOT NULL AND batch_id != ''"
                elif batch_id == "":
                    sql += " WHERE batch_id IS NULL OR batch_id = ''"
                elif batch_id is not None:
                    sql += " WHERE batch_id = ?"
                    params.append(batch_id)
                sql += " ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                cursor.execute(sql, params)
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
                           moves, ply_count, result, winner, termination_reason,
                           batch_id
                    FROM arena_records
                    WHERE id = ?
                    """,
                    (record_id,),
                )
                row = cursor.fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    # ---------- 批量对弈持久化 ----------

    def save_batch(self, batch: dict[str, Any]) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO arena_batches (
                            id, created_at, end_time,
                            white_model, white_arg, black_model, black_arg,
                            rounds_planned, rounds_completed, status, tally_json, error
                        ) VALUES (
                            :id, :created_at, :end_time,
                            :white_model, :white_arg, :black_model, :black_arg,
                            :rounds_planned, :rounds_completed, :status, :tally_json, :error
                        )
                        ON CONFLICT(id) DO UPDATE SET
                            end_time = excluded.end_time,
                            rounds_completed = excluded.rounds_completed,
                            status = excluded.status,
                            tally_json = excluded.tally_json,
                            error = excluded.error
                        """,
                        {
                            "id": batch["id"],
                            "created_at": batch.get("created_at"),
                            "end_time": batch.get("end_time"),
                            "white_model": batch.get("white_model", ""),
                            "white_arg": batch.get("white_arg"),
                            "black_model": batch.get("black_model", ""),
                            "black_arg": batch.get("black_arg"),
                            "rounds_planned": batch.get("rounds_planned", 0),
                            "rounds_completed": batch.get("rounds_completed", 0),
                            "status": batch.get("status", "running"),
                            "tally_json": json.dumps(batch.get("tally", {}), ensure_ascii=False),
                            "error": batch.get("error"),
                        },
                    )
            finally:
                conn.close()

    def get_batch(self, batch_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM arena_batches WHERE id = ?", (batch_id,))
                row = cursor.fetchone()
                if row is None:
                    return None
                result = dict(row)
                result["tally"] = json.loads(result.pop("tally_json") or "{}")
                return result
            finally:
                conn.close()

    def list_batches(self, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, created_at, end_time,
                           white_model, white_arg, black_model, black_arg,
                           rounds_planned, rounds_completed, status, tally_json, error
                    FROM arena_batches
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                )
                rows = cursor.fetchall()
                results = []
                for row in rows:
                    item = dict(row)
                    item["tally"] = json.loads(item.pop("tally_json") or "{}")
                    results.append(item)
                return results
            finally:
                conn.close()


# 模块级单例
storage = ArenaStorage()
