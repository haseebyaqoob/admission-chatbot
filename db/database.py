"""
database.py
───────────
SQLite-backed chat-history store for the admission bot.
"""

import json
import sqlite3
from pathlib import Path

from config_loader import cfg

DB_PATH = Path(cfg["db_path"])


class DatabaseManager:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id        TEXT,
                    role              TEXT,
                    message           TEXT,
                    intent            TEXT,
                    sources_referenced TEXT,
                    timestamp         TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def save_message(self, session_id, role, message, intent=None, sources=None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO chat_history "
                "(session_id, role, message, intent, sources_referenced) "
                "VALUES (?,?,?,?,?)",
                (session_id, role, message, intent, json.dumps(sources or [])),
            )
            conn.commit()

    def get_recent_history(self, session_id: str, n: int = 4) -> list:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT role, message FROM chat_history "
                "WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, n * 2),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]
