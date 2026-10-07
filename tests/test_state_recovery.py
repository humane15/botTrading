"""Test state recovery: kondisi setelah bot restart dicocokkan dengan bursa."""

from __future__ import annotations

import pytest
from helpers import make_settings, make_signal, paper_executor, sol_rules

from core.database import connect
from core.orders import make_client_id
from risk.position_manager import PositionManager
from risk.positions import CLOSED, FAILED, OPEN, TP1_HIT, Position, PositionStore
from risk.risk_manager import RiskManager
from risk.state_recovery import recover_state

SYMBOL = "SOL/USDT"


@pytest.fixture
def store():
    return PositionStore(connect(":memory:"))


async def open_one(store, executor):
    manager = PositionManager(store, executor)
    position = await manager.open_position(make_signal(), 2.0)
    return position


def label(position):
    return f"{SYMBOL}#{position.id}"


async def restart(store, executor, risk=None):
    """Bot hidup lagi: manager baru, database dan bursa tetap sama."""
    return await recover_state(PositionManager(store, executor), "USDT", risk)


async def test_entry_terkirim_sebelum_bot_mati_diselesaikan(store):
    ex = paper_executor()
    position = store.insert(Position(symbol=SYMBOL, mode="paper", planned_qty=2.0, initial_stop=98, stop_price=98, tp1_price=104, atr=1.0))
    position.entry_client_id = make_client_id("en", position.id)
    store.save(position)
    await ex.buy_limit_ioc(SYMBOL, 2.0, 100.3, position.entry_client_id)  # terkirim, lalu bot mati

    report = await restart(store, ex)
    assert report.entries_completed == [label(position)]
    recovered = store.get(position.id)
    assert recovered.status == OPEN and recovered.qty == 1.998
    assert len(await ex.open_orders()) == 3  # OCO (2 leg) + stop runner


async def test_entry_yang_tidak_pernah_terkirim_ditandai_gagal(store):
    ex = paper_executor()
    position = store.insert(Position(symbol=SYMBOL, mode="paper", planned_qty=2.0, stop_price=98, tp1_price=104))
    position.entry_client_id = make_client_id("en", position.id)
    store.save(position)
    report = await restart(store, ex)
    assert report.failed_entries == [label(position)]
    assert store.get(position.id).status == FAILED


async def test_stop_terisi_saat_bot_mati(store):
    ex = paper_executor()
    position = await open_one(store, ex)
    ex.process_candle(SYMBOL, 100, 100.2, 97, 97.5)  # harga jatuh saat bot mati
    report = await restart(store, ex)
    assert report.closed_while_offline == [label(position)]
    closed = store.get(position.id)
    assert closed.status == CLOSED and closed.exit_reason == "stop_loss"
    assert closed.r_multiple == pytest.approx(-1.0, rel=1e-6)


async def test_tp1_terisi_saat_bot_mati_stop_pindah_ke_breakeven(store):
    ex = paper_executor()
    position = await open_one(store, ex)
    ex.process_candle(SYMBOL, 100, 104.5, 99.9, 104.2)
    report = await restart(store, ex)
    assert report.filled_while_offline == [label(position)]
    recovered = store.get(position.id)
    assert recovered.status == TP1_HIT and recovered.breakeven
    assert recovered.stop_price >= recovered.breakeven_price(0.001)
    stops = [o for o in await ex.open_orders() if o.type == "stop_loss"]
    assert len(stops) == 1 and stops[0].amount == pytest.approx(recovered.qty)


async def test_stop_yang_dibatalkan_di_luar_bot_dipasang_ulang(store):
    ex = paper_executor()
    position = await open_one(store, ex)
    await ex.cancel_order(SYMBOL, position.stop_b_client_id)  # dibatalkan manual di aplikasi
    report = await restart(store, ex)
    assert report.reprotected == [label(position)] and report.resumed == [label(position)]
    covered = sum(o.amount for o in await ex.open_orders() if o.type == "stop_loss")
    assert covered == pytest.approx(store.get(position.id).qty)


async def test_coin_dijual_di_luar_bot(store):
    ex = paper_executor()
    position = await open_one(store, ex)
    await ex.cancel_oco(SYMBOL, position.oco_client_id)
    await ex.cancel_order(SYMBOL, position.stop_b_client_id)
    await ex.sell_market(SYMBOL, 1.998, "manual-sell")
    report = await restart(store, ex)
    assert report.closed_externally == [label(position)]
    closed = store.get(position.id)
    assert closed.status == CLOSED and closed.exit_reason == "ditutup_di_luar_bot"


async def test_stop_bot_tak_tercatat_dibatalkan_lalu_dipasang_ulang(store):
    ex = paper_executor()
    position = await open_one(store, ex)
    await ex.cancel_order(SYMBOL, position.stop_b_client_id)
    stray = make_client_id("sb", position.id)  # terkirim saat jaringan putus, hasilnya tidak sempat tercatat
    await ex.place_stop_loss(SYMBOL, 0.999, 98.0, stray)
    report = await restart(store, ex)
    assert report.resumed == [label(position)] and report.reprotected == [label(position)]
    assert not report.closed_externally and not report.errors
    recovered = store.get(position.id)
    assert recovered.status == OPEN and recovered.qty == pytest.approx(1.998)
    assert (await ex.fetch_order(SYMBOL, stray)).status == "canceled"
    covered = sum(o.amount for o in await ex.open_orders() if o.type == "stop_loss")
    assert covered == pytest.approx(1.998)


async def test_coin_terkunci_order_manual_tidak_dianggap_terjual(store):
    ex = paper_executor()
    position = await open_one(store, ex)
    await ex.cancel_order(SYMBOL, position.stop_b_client_id)
    await ex.place_stop_loss(SYMBOL, 0.999, 95.0, "web_manual_stop")  # order manual pengguna di aplikasi
    report = await restart(store, ex)
    assert not report.closed_externally and not report.reprotected
    assert len(report.errors) == 1 and "terkunci order di luar bot" in report.errors[0]
    recovered = store.get(position.id)
    assert recovered.status == OPEN and recovered.qty == pytest.approx(1.998)
    assert (await ex.fetch_order(SYMBOL, "web_manual_stop")).status == "open"  # order manual tidak disentuh


async def test_saldo_di_luar_bot_diblokir_dan_order_yatim_dibatalkan(tmp_path, store):
    ex = paper_executor(balance=1000)
    ex.add_rules(sol_rules(symbol="ETH/USDT", base="ETH"))
    ex.set_price("ETH/USDT", 2000.0)
    await ex.buy_limit_ioc("ETH/USDT", 0.1, 2010, "web_manual")          # dibeli manual, bukan oleh bot
    await ex.buy_limit_ioc(SYMBOL, 1.0, 100.3, "web_manual_2")
    await ex.place_stop_loss(SYMBOL, 0.5, 95.0, "tb-sb-999-orphan")        # sisa order bot lama
    risk = RiskManager(make_settings(tmp_path), store, "paper", kill_file=tmp_path / "STOP")

    report = await restart(store, ex, risk)
    assert set(report.untracked) == {"ETH/USDT", "SOL/USDT"}
    assert report.untracked["ETH/USDT"] == pytest.approx(0.0999 * 2000)
    assert {"ETH/USDT", "SOL/USDT"} <= risk.blocked_symbols
    assert report.orphan_orders_canceled == ["tb-sb-999-orphan"]
    assert await ex.open_orders() == []
    assert any("saldo di luar bot" in line for line in report.lines())


async def test_restart_tanpa_masalah(store):
    ex = paper_executor()
    position = await open_one(store, ex)
    report = await restart(store, ex)
    assert report.checked == 1 and report.resumed == [label(position)]
    assert not (report.reprotected or report.closed_externally or report.orphan_orders_canceled or report.errors)
