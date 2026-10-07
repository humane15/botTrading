"""Bobot confluence engine yang tersimpan di SQLite (bukan hardcode).

Nilai default di sini hanya dipakai untuk mengisi database saat pertama kali
dibuat. Setelah itu nilai di database yang berlaku, dan modul learning
(Fase 5) mengubahnya lewat WeightStore.update(), yang:

* membatasi perubahan maksimal 10% per siklus (untuk bobot bernilai 0,
  batasnya 10% dari rentang bobot tersebut);
* menjaga nilai tetap di dalam batas bawah dan atas;
* mencatat setiap perubahan ke tabel weight_history dan logs/learning.log.

Nama bobot memakai PERAN timeframe, bukan nama timeframe, supaya tetap valid
jika daftar timeframe diubah:
* tf.<peran>         bobot timeframe (trend=1h, structure=30m, setup=15m, trigger=5m)
* w.<peran>.<modul>  bobot modul di dalam timeframe tersebut
* penalty.<kondisi>  penalti skor (skala -1..1) jika kondisi berisiko terdeteksi
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType

from config.logging_setup import LEARNING_LOGGER
from core.database import connect

learning_log = logging.getLogger(LEARNING_LOGGER)

ROLES = ("trend", "structure", "setup", "trigger")

DEFAULT_TIMEFRAME_WEIGHTS: dict[str, float] = {"trend": 0.30, "structure": 0.20, "setup": 0.30, "trigger": 0.20}

# Modul yang dinilai pada tiap peran timeframe beserta bobot awalnya.
DEFAULT_MODULE_WEIGHTS: dict[str, dict[str, float]] = {
    "trend": {"ma": 1.0, "sr": 0.8, "structure": 0.4, "macd": 0.4, "rsi": 0.4},
    "structure": {"structure": 1.0, "ma": 0.6, "macd": 0.5, "rsi": 0.3},
    "setup": {"fib": 1.0, "sr": 0.8, "bb": 0.6, "rsi": 0.6, "ma": 0.5},
    "trigger": {"macd": 1.0, "volume": 0.8, "candle": 0.8, "rsi": 0.4, "bb": 0.3},
}

# Kondisi berisiko saat entry (sama dengan label mistake analyzer Fase 5).
# Penalti awal 0; modul learning menaikkannya jika kondisi itu sering berujung rugi.
PENALTY_CONDITIONS: dict[str, str] = {
    "melawan_tren_1h": "tren 1h tidak mendukung (bukan uptrend)",
    "dekat_resistance": "ruang ke resistance kurang dari 2x risiko",
    "volume_lemah": "volume candle trigger di bawah rata rata",
    "entry_terlambat": "RSI sudah tinggi saat entry",
    "btc_dump": "BTC turun lebih dari 1.5% dalam 1 jam",
}
DEFAULT_PENALTIES: dict[str, float] = {name: 0.0 for name in PENALTY_CONDITIONS} | {"btc_turun": 0.2}


@dataclass(frozen=True)
class WeightSpec:
    default: float
    min: float
    max: float
    description: str


def _build_specs() -> dict[str, WeightSpec]:
    specs: dict[str, WeightSpec] = {}
    for role, value in DEFAULT_TIMEFRAME_WEIGHTS.items():
        specs[f"tf.{role}"] = WeightSpec(value, 0.05, 1.0, f"bobot timeframe peran {role}")
    for role, modules in DEFAULT_MODULE_WEIGHTS.items():
        for module, value in modules.items():
            specs[f"w.{role}.{module}"] = WeightSpec(value, 0.0, 3.0, f"bobot modul {module} pada peran {role}")
    for name, value in DEFAULT_PENALTIES.items():
        description = PENALTY_CONDITIONS.get(name, "BTC dalam regime Trending Down")
        specs[f"penalty.{name}"] = WeightSpec(value, 0.0, 0.5, f"penalti: {description}")
    return specs


DEFAULT_WEIGHTS: dict[str, WeightSpec] = _build_specs()


def default_weight_values() -> dict[str, float]:
    return {name: spec.default for name, spec in DEFAULT_WEIGHTS.items()}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class WeightStore:
    """Akses bobot sinyal di tabel signal_weights."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._init_schema()

    @classmethod
    def open(cls, path: str | Path) -> WeightStore:
        return cls(connect(path))

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS signal_weights (
                    name TEXT PRIMARY KEY,
                    value REAL NOT NULL,
                    default_value REAL NOT NULL,
                    min_value REAL NOT NULL,
                    max_value REAL NOT NULL,
                    description TEXT,
                    updated_at TEXT NOT NULL
                )"""
            )
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS weight_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    old_value REAL,
                    new_value REAL NOT NULL,
                    reason TEXT,
                    changed_at TEXT NOT NULL
                )"""
            )
            now = _now()
            for name, spec in DEFAULT_WEIGHTS.items():
                # Bobot baru diisi default; bobot lama tetap nilainya, hanya batasnya yang diperbarui.
                self.conn.execute(
                    """INSERT INTO signal_weights (name, value, default_value, min_value, max_value, description, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(name) DO UPDATE SET
                           default_value = excluded.default_value,
                           min_value = excluded.min_value,
                           max_value = excluded.max_value,
                           description = excluded.description,
                           value = MIN(MAX(signal_weights.value, excluded.min_value), excluded.max_value)""",
                    (name, spec.default, spec.default, spec.min, spec.max, spec.description, now),
                )

    def all(self) -> dict[str, float]:
        rows = self.conn.execute("SELECT name, value FROM signal_weights ORDER BY name").fetchall()
        return {row["name"]: float(row["value"]) for row in rows}

    def snapshot(self) -> Mapping[str, float]:
        """Salinan bobot yang tidak bisa diubah, untuk dipakai satu siklus analisis."""
        return MappingProxyType(self.all())

    def get(self, name: str) -> float:
        row = self.conn.execute("SELECT value FROM signal_weights WHERE name = ?", (name,)).fetchone()
        if row is None:
            raise KeyError(f"bobot tidak dikenal: {name}")
        return float(row["value"])

    def update(self, changes: Mapping[str, float], reason: str, max_change: float = 0.10) -> dict[str, tuple[float, float]]:
        """Ubah bobot menuju nilai target dengan batas perubahan per siklus.

        Mengembalikan {nama: (nilai lama, nilai baru)} untuk bobot yang benar benar berubah.
        """
        applied: dict[str, tuple[float, float]] = {}
        now = _now()
        with self.conn:
            for name, target in changes.items():
                row = self.conn.execute(
                    "SELECT value, min_value, max_value FROM signal_weights WHERE name = ?", (name,)
                ).fetchone()
                if row is None:
                    raise KeyError(f"bobot tidak dikenal: {name}")
                old, low, high = float(row["value"]), float(row["min_value"]), float(row["max_value"])
                step = abs(old) * max_change if old != 0 else (high - low) * max_change
                new = min(max(float(target), old - step), old + step)
                new = round(min(max(new, low), high), 6)
                if abs(new - old) < 1e-9:
                    continue
                self.conn.execute("UPDATE signal_weights SET value = ?, updated_at = ? WHERE name = ?", (new, now, name))
                self.conn.execute(
                    "INSERT INTO weight_history (name, old_value, new_value, reason, changed_at) VALUES (?, ?, ?, ?, ?)",
                    (name, old, new, reason, now),
                )
                applied[name] = (old, new)
        for name, (old, new) in applied.items():
            learning_log.info("Bobot %s %s %.4f -> %.4f (%s)", name, "dinaikkan" if new > old else "diturunkan", old, new, reason)
        return applied

    def reset_defaults(self, reason: str = "reset ke default") -> None:
        now = _now()
        with self.conn:
            for name, spec in DEFAULT_WEIGHTS.items():
                old = self.get(name)
                if abs(old - spec.default) > 1e-12:
                    self.conn.execute("UPDATE signal_weights SET value = ?, updated_at = ? WHERE name = ?", (spec.default, now, name))
                    self.conn.execute(
                        "INSERT INTO weight_history (name, old_value, new_value, reason, changed_at) VALUES (?, ?, ?, ?, ?)",
                        (name, old, spec.default, reason, now),
                    )
        learning_log.info("Semua bobot dikembalikan ke default (%s)", reason)

    def history(self, name: str | None = None, limit: int = 50) -> list[dict[str, object]]:
        query = "SELECT name, old_value, new_value, reason, changed_at FROM weight_history"
        params: tuple[object, ...] = ()
        if name is not None:
            query += " WHERE name = ?"
            params = (name,)
        query += " ORDER BY id DESC LIMIT ?"
        rows = self.conn.execute(query, (*params, limit)).fetchall()
        return [dict(row) for row in rows]
