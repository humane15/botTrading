"""Posisi trading dan penyimpanannya di SQLite (data/bot.db).

Tabel:
* positions        : status setiap posisi (entry, TP1, stop, breakeven, PnL, snapshot sinyal)
* orders           : semua order yang dikirim bot (audit dan recovery)
* equity_snapshots : nilai akun di awal hari/minggu untuk batas rugi harian/mingguan

Siklus status posisi:
    pending_entry -> open -> tp1_hit -> closed
    pending_entry -> failed   (order entry tidak terisi)
    pending_entry -> dust     (terisi terlalu sedikit untuk bisa dijual, perlu dicek manual)
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.database import connect
from core.orders import OrderResult

PENDING = "pending_entry"
OPEN = "open"
TP1_HIT = "tp1_hit"
CLOSED = "closed"
FAILED = "failed"
DUST = "dust"
ACTIVE_STATUSES = (PENDING, OPEN, TP1_HIT)


# Jam yang dipakai untuk mencatat waktu posisi. Backtest memasang jam simulasi
# lewat simulated_clock() sehingga opened_at/closed_at mengikuti waktu candle.
_CLOCK: ContextVar[Callable[[], datetime] | None] = ContextVar("position_clock", default=None)


def utc_now_iso() -> str:
    clock = _CLOCK.get()
    now = clock() if clock is not None else datetime.now(timezone.utc)
    return now.isoformat(timespec="seconds")


@contextmanager
def simulated_clock(clock: Callable[[], datetime]) -> Iterator[None]:
    """Selama blok ini, semua waktu posisi, order, dan ekuitas dicatat memakai `clock`."""
    token = _CLOCK.set(clock)
    try:
        yield
    finally:
        _CLOCK.reset(token)


@dataclass
class Position:
    symbol: str
    mode: str                      # paper, live, atau backtest
    status: str = PENDING
    id: int | None = None
    planned_qty: float = 0.0
    entry_price: float = 0.0       # harga rata rata terisi
    initial_qty: float = 0.0       # jumlah bersih setelah fee, yang bisa dijual
    dust_qty: float = 0.0          # sisa coin di bawah step size yang tidak bisa dijual (tetap di akun)
    qty: float = 0.0               # sisa jumlah yang masih dipegang
    initial_stop: float = 0.0
    stop_price: float = 0.0
    tp1_price: float = 0.0
    tp1_qty: float = 0.0
    atr: float = 0.0
    highest_price: float = 0.0     # untuk trailing stop dan MFE
    lowest_price: float = 0.0      # untuk MAE
    breakeven: bool = False
    cost_quote: float = 0.0        # USDT yang dibayar untuk entry (termasuk fee quote)
    proceeds_quote: float = 0.0    # USDT bersih yang diterima dari exit
    fees_quote: float = 0.0
    realized_pnl: float = 0.0
    risk_amount: float = 0.0       # rugi (USDT) jika stop awal tersentuh
    size_multiplier: float = 1.0
    exit_reason: str = ""
    btc_corr: float | None = None
    score: float = 0.0
    reasons: str = ""
    pattern: str = ""
    regime: str = ""
    conditions: str = ""
    features: dict[str, float] = field(default_factory=dict)
    tp_mode: str = "oco"           # oco (order bursa) atau software (dipantau bot)
    entry_client_id: str = ""
    oco_client_id: str = ""
    tp_client_id: str = ""
    stop_a_client_id: str = ""     # stop leg OCO (melindungi bagian TP1)
    stop_b_client_id: str = ""     # stop untuk sisa posisi (runner)
    tp_filled: float = 0.0         # jumlah terisi yang sudah dicatat per order aktif
    stop_a_filled: float = 0.0
    stop_b_filled: float = 0.0
    opened_at: str | None = None
    closed_at: str | None = None
    updated_at: str | None = None

    @property
    def base(self) -> str:
        return self.symbol.split("/")[0]

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    @property
    def cost_per_unit(self) -> float:
        """Harga pokok per coin yang diterima. Dust ikut dibagi rata sehingga biayanya tidak
        dibebankan ke PnL trade (dust tetap bernilai dan tetap ada di akun)."""
        received = self.initial_qty + self.dust_qty
        return self.cost_quote / received if received > 0 else 0.0

    def breakeven_price(self, fee_rate: float) -> float:
        """Harga jual agar sisa posisi tidak rugi (menutup fee beli dan fee jual)."""
        return self.cost_per_unit / (1 - fee_rate)

    def record_exit(self, qty: float, price: float, fee_quote: float, reason: str, when: str | None = None) -> float:
        """Catat penjualan sebagian/seluruh posisi. Mengembalikan PnL bagian ini."""
        qty = min(qty, self.qty)
        gross = qty * price
        pnl = gross - fee_quote - qty * self.cost_per_unit
        self.proceeds_quote += gross - fee_quote
        self.fees_quote += fee_quote
        self.realized_pnl += pnl
        self.qty = max(self.qty - qty, 0.0)
        self.exit_reason = reason
        self.updated_at = when or utc_now_iso()
        return pnl

    @property
    def r_multiple(self) -> float:
        return self.realized_pnl / self.risk_amount if self.risk_amount > 0 else 0.0

    def unrealized_pnl(self, price: float, fee_rate: float) -> float:
        return self.qty * price * (1 - fee_rate) - self.qty * self.cost_per_unit

    def update_extremes(self, high: float, low: float) -> None:
        self.highest_price = max(self.highest_price, high)
        self.lowest_price = min(self.lowest_price, low) if self.lowest_price > 0 else low

    @property
    def mfe_pct(self) -> float:
        """Kenaikan terbaik selama posisi terbuka, relatif terhadap harga entry."""
        return self.highest_price / self.entry_price - 1 if self.entry_price > 0 else 0.0

    @property
    def mae_pct(self) -> float:
        """Penurunan terburuk selama posisi terbuka, relatif terhadap harga entry."""
        return self.lowest_price / self.entry_price - 1 if self.entry_price > 0 and self.lowest_price > 0 else 0.0


_COLUMN_TYPES = {float: "REAL", int: "INTEGER", bool: "INTEGER", str: "TEXT"}
_POSITION_FIELDS = [f for f in fields(Position) if f.name != "id"]


def _sql_type(annotation: Any) -> str:
    text = str(annotation)
    if "dict" in text:
        return "TEXT"
    for py_type, sql in _COLUMN_TYPES.items():
        if py_type.__name__ in text:
            return sql
    return "TEXT"


def _to_row(position: Position) -> dict[str, Any]:
    data = asdict(position)
    data.pop("id")
    data["features"] = json.dumps(data["features"], allow_nan=True)
    data["breakeven"] = int(data["breakeven"])
    return data


def _from_row(row: sqlite3.Row) -> Position:
    data = dict(row)
    data["features"] = json.loads(data["features"] or "{}")
    data["breakeven"] = bool(data["breakeven"])
    return Position(**data)


class PositionStore:
    """CRUD posisi, order, dan snapshot ekuitas di SQLite."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._init_schema()

    @classmethod
    def open(cls, path: str | Path) -> PositionStore:
        return cls(connect(path))

    def _init_schema(self) -> None:
        columns = ",\n".join(f"{f.name} {_sql_type(f.type)}" for f in _POSITION_FIELDS)
        with self.conn:
            self.conn.execute(f"CREATE TABLE IF NOT EXISTS positions (id INTEGER PRIMARY KEY AUTOINCREMENT,\n{columns})")
            existing = {row["name"] for row in self.conn.execute("PRAGMA table_info(positions)")}
            for f in _POSITION_FIELDS:  # migrasi ringan: tambahkan kolom baru jika skema lama
                if f.name not in existing:
                    self.conn.execute(f"ALTER TABLE positions ADD COLUMN {f.name} {_sql_type(f.type)}")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(mode, status)")
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS orders (
                    client_id TEXT PRIMARY KEY,
                    position_id INTEGER,
                    role TEXT,
                    symbol TEXT,
                    side TEXT,
                    type TEXT,
                    status TEXT,
                    amount REAL,
                    filled REAL,
                    average REAL,
                    price REAL,
                    stop_price REAL,
                    fee_base REAL,
                    fee_quote REAL,
                    created_at TEXT,
                    updated_at TEXT
                )"""
            )
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS equity_snapshots (
                    mode TEXT NOT NULL,
                    period TEXT NOT NULL,
                    equity REAL NOT NULL,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (mode, period)
                )"""
            )

    # ------------------------------------------------------------------
    # Posisi
    # ------------------------------------------------------------------
    def insert(self, position: Position) -> Position:
        now = utc_now_iso()
        position.opened_at = position.opened_at or now
        position.updated_at = now
        row = _to_row(position)
        names = ", ".join(row)
        marks = ", ".join(f":{name}" for name in row)
        with self.conn:
            cursor = self.conn.execute(f"INSERT INTO positions ({names}) VALUES ({marks})", row)
        position.id = int(cursor.lastrowid)
        return position

    def save(self, position: Position) -> None:
        if position.id is None:
            raise ValueError("posisi belum disimpan (id kosong)")
        position.updated_at = utc_now_iso()
        row = _to_row(position)
        assignments = ", ".join(f"{name} = :{name}" for name in row)
        with self.conn:
            self.conn.execute(f"UPDATE positions SET {assignments} WHERE id = :id", {**row, "id": position.id})

    def get(self, position_id: int) -> Position | None:
        row = self.conn.execute("SELECT * FROM positions WHERE id = ?", (position_id,)).fetchone()
        return _from_row(row) if row else None

    def active(self, mode: str | None = None) -> list[Position]:
        return self.with_status(ACTIVE_STATUSES, mode)

    def with_status(self, statuses: Sequence[str], mode: str | None = None) -> list[Position]:
        marks = ", ".join("?" for _ in statuses)
        query = f"SELECT * FROM positions WHERE status IN ({marks})"
        params: list[Any] = list(statuses)
        if mode is not None:
            query += " AND mode = ?"
            params.append(mode)
        return [_from_row(row) for row in self.conn.execute(query + " ORDER BY id", params)]

    def closed(self, mode: str | None = None, since: str | None = None, limit: int | None = None) -> list[Position]:
        query = "SELECT * FROM positions WHERE status = ?"
        params: list[Any] = [CLOSED]
        if mode is not None:
            query += " AND mode = ?"
            params.append(mode)
        if since is not None:
            query += " AND closed_at >= ?"
            params.append(since)
        query += " ORDER BY closed_at DESC, id DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        return [_from_row(row) for row in self.conn.execute(query, params)]

    def dust_by_asset(self, mode: str | None = None) -> dict[str, float]:
        """Total dust (sisa coin di bawah step size) yang ditinggalkan posisi bot, per coin."""
        query = "SELECT symbol, SUM(dust_qty) AS dust FROM positions WHERE dust_qty > 0"
        params: list[Any] = []
        if mode is not None:
            query += " AND mode = ?"
            params.append(mode)
        totals: dict[str, float] = {}
        for row in self.conn.execute(query + " GROUP BY symbol", params):
            base = str(row["symbol"]).split("/")[0]
            totals[base] = totals.get(base, 0.0) + float(row["dust"] or 0.0)
        return totals

    # ------------------------------------------------------------------
    # Order
    # ------------------------------------------------------------------
    def record_order(self, position_id: int | None, role: str, order: OrderResult) -> None:
        now = utc_now_iso()
        with self.conn:
            self.conn.execute(
                """INSERT INTO orders (client_id, position_id, role, symbol, side, type, status, amount, filled,
                                       average, price, stop_price, fee_base, fee_quote, created_at, updated_at)
                   VALUES (:client_id, :position_id, :role, :symbol, :side, :type, :status, :amount, :filled,
                           :average, :price, :stop_price, :fee_base, :fee_quote, :now, :now)
                   ON CONFLICT(client_id) DO UPDATE SET
                       status = excluded.status, filled = excluded.filled, average = excluded.average,
                       fee_base = excluded.fee_base, fee_quote = excluded.fee_quote, updated_at = excluded.updated_at""",
                {
                    "client_id": order.client_id, "position_id": position_id, "role": role, "symbol": order.symbol,
                    "side": order.side, "type": order.type, "status": order.status, "amount": order.amount,
                    "filled": order.filled, "average": order.average, "price": order.price,
                    "stop_price": order.stop_price, "fee_base": order.fee_base, "fee_quote": order.fee_quote, "now": now,
                },
            )

    def orders(self, position_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM orders WHERE position_id = ? ORDER BY created_at, rowid", (position_id,))
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Ekuitas awal periode (untuk batas rugi harian dan mingguan)
    # ------------------------------------------------------------------
    def period_equity(self, mode: str, period: str) -> float | None:
        row = self.conn.execute(
            "SELECT equity FROM equity_snapshots WHERE mode = ? AND period = ?", (mode, period)
        ).fetchone()
        return float(row["equity"]) if row else None

    def ensure_period_equity(self, mode: str, period: str, equity: float) -> float:
        """Simpan ekuitas awal periode jika belum ada; kembalikan nilai yang tersimpan."""
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO equity_snapshots (mode, period, equity, recorded_at) VALUES (?, ?, ?, ?)",
                (mode, period, equity, utc_now_iso()),
            )
        stored = self.period_equity(mode, period)
        return equity if stored is None else stored
