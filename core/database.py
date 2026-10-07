"""Koneksi SQLite bersama (data/bot.db) untuk bobot sinyal, jurnal, dan state posisi."""

from __future__ import annotations

import sqlite3
from pathlib import Path

MEMORY = ":memory:"


def connect(path: str | Path) -> sqlite3.Connection:
    """Buka database SQLite dengan mode WAL agar aman dibaca sambil ditulis."""
    target = str(path)
    if target != MEMORY:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if target != MEMORY:
        conn.execute("PRAGMA journal_mode = WAL")
    return conn
