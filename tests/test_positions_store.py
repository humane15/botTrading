"""Test penyimpanan posisi di SQLite: simpan/baca ulang, query status, order, ekuitas, migrasi skema."""

from __future__ import annotations

import sqlite3

import pytest

from core.database import connect
from core.orders import OrderResult
from risk.positions import (
    CLOSED,
    FAILED,
    OPEN,
    PENDING,
    TP1_HIT,
    Position,
    PositionStore,
)


def sample(**overrides):
    values = dict(symbol="SOL/USDT", mode="paper", status=OPEN, qty=1.5, initial_qty=2.0, entry_price=100.0,
                  cost_quote=200.0, breakeven=True, btc_corr=0.85, features={"rsi": 45.0, "15m.score": 0.4})
    values.update(overrides)
    return Position(**values)


def test_simpan_dan_baca_ulang_lengkap(tmp_path):
    path = tmp_path / "bot.db"
    store = PositionStore.open(path)
    position = store.insert(sample())
    position.stop_price = 101.5
    store.save(position)
    store.conn.close()

    reopened = PositionStore.open(path).get(position.id)  # bertahan setelah bot restart
    assert reopened.features == {"rsi": 45.0, "15m.score": 0.4}
    assert reopened.breakeven is True and reopened.btc_corr == 0.85 and reopened.stop_price == 101.5
    assert reopened.opened_at and reopened.updated_at


def test_query_posisi_aktif_dan_tertutup():
    store = PositionStore(connect(":memory:"))
    for status in (PENDING, OPEN, TP1_HIT, CLOSED, FAILED):
        store.insert(sample(status=status, closed_at="2026-10-07T10:00:00+00:00" if status == CLOSED else None))
    store.insert(sample(mode="live"))
    assert [p.status for p in store.active("paper")] == [PENDING, OPEN, TP1_HIT]
    assert len(store.active()) == 4
    assert [p.status for p in store.closed("paper")] == [CLOSED]
    assert store.closed("paper", since="2026-10-08") == []


def test_order_dicatat_dan_diperbarui():
    store = PositionStore(connect(":memory:"))
    position = store.insert(sample())
    store.record_order(position.id, "stop_b", OrderResult("tb-sb-1-a", "SOL/USDT", "sell", "stop_loss", "open", amount=1.5))
    store.record_order(position.id, "stop_b", OrderResult("tb-sb-1-a", "SOL/USDT", "sell", "stop_loss", "closed",
                                                          amount=1.5, filled=1.5, average=97.9, fee_quote=0.15))
    orders = store.orders(position.id)
    assert len(orders) == 1 and orders[0]["status"] == "closed" and orders[0]["filled"] == 1.5


def test_ekuitas_awal_periode_hanya_dicatat_sekali():
    store = PositionStore(connect(":memory:"))
    assert store.ensure_period_equity("paper", "day:2026-10-07", 1000.0) == 1000.0
    assert store.ensure_period_equity("paper", "day:2026-10-07", 900.0) == 1000.0
    assert store.ensure_period_equity("live", "day:2026-10-07", 500.0) == 500.0  # mode terpisah


def test_migrasi_skema_lama_menambah_kolom_baru():
    conn = connect(":memory:")
    conn.execute("CREATE TABLE positions (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, mode TEXT, status TEXT)")
    store = PositionStore(conn)
    position = store.insert(sample())
    assert store.get(position.id).features == {"rsi": 45.0, "15m.score": 0.4}


def test_pnl_dan_r_multiple_posisi():
    position = sample(qty=2.0, initial_qty=2.0, cost_quote=200.0, risk_amount=4.0, realized_pnl=0.0)
    pnl = position.record_exit(1.0, 104.0, 0.104, "take_profit_1")
    assert pnl == pytest.approx(104 - 0.104 - 100)
    position.record_exit(1.0, 98.0, 0.098, "stop_loss")
    assert position.qty == 0 and position.realized_pnl == pytest.approx(3.896 - 2.098)
    assert position.r_multiple == pytest.approx(position.realized_pnl / 4.0)
    assert position.breakeven_price(0.001) == pytest.approx(100 / 0.999)
    with pytest.raises(ValueError):
        PositionStore(connect(":memory:")).save(sample())
    assert isinstance(connect(":memory:"), sqlite3.Connection)
