"""Test siklus posisi: entry, partial fill, OCO + runner, TP1, breakeven, trailing, stop, gap, exit manual."""

from __future__ import annotations

import pytest
from helpers import make_settings, make_signal, paper_executor, sol_rules

from core.database import connect
from core.orders import OrderError
from core.paper_exchange import PaperExecutor
from risk.position_manager import ManagerConfig, PositionManager
from risk.positions import CLOSED, DUST, FAILED, OPEN, TP1_HIT, PositionStore
from risk.risk_manager import RiskManager

SYMBOL = "SOL/USDT"


@pytest.fixture
def store():
    return PositionStore(connect(":memory:"))


def build(store, executor=None, **config):
    executor = executor or paper_executor()
    return PositionManager(store, executor, ManagerConfig(**config)), executor


async def candle(manager, executor, position, o, h, l, c, atr=1.0):
    """Satu candle 5m: bursa simulasi mengeksekusi order, lalu bot sync dan update trailing."""
    executor.process_candle(position.symbol, o, h, l, c)
    position = await manager.sync(position)
    return await manager.on_candle(position, h, l, c, atr)


async def test_entry_memasang_oco_dan_stop_runner(store):
    manager, ex = build(store)
    position = await manager.open_position(make_signal(), 2.0)
    assert position.status == OPEN
    assert position.entry_price == pytest.approx(100.05)       # ask + slippage 0.05%
    assert position.qty == position.initial_qty == 1.998        # 2 - fee 0.1% dalam coin
    assert position.tp1_qty == 0.999 and position.tp_mode == "oco"
    orders = {o.client_id: o for o in await ex.open_orders()}
    assert {o.type for o in orders.values()} == {"limit_maker", "stop_loss"}
    assert sum(o.amount for o in orders.values() if o.type == "stop_loss") == pytest.approx(1.998)
    assert orders[position.tp_client_id].price == 104.0
    assert position.risk_amount == pytest.approx(200.1 - 1.998 * 98 * 0.9995 * 0.999)
    assert any(e.startswith("[BUY]") for e in manager.events)


async def test_siklus_untung_tp1_breakeven_trailing(store):
    manager, ex = build(store)
    start_equity = ex.equity()
    position = await manager.open_position(make_signal(), 2.0)
    position = await candle(manager, ex, position, 100, 104.5, 99.8, 104.2)
    assert position.status == TP1_HIT and position.breakeven
    assert position.qty == pytest.approx(0.999)
    assert position.stop_price >= position.breakeven_price(0.001)  # tidak mungkin rugi lagi
    stops = [position.stop_price]
    for bar in [(104.2, 106, 103.9, 105.8), (105.8, 109, 105.5, 108.7), (108.7, 108.8, 107.5, 108.0)]:
        position = await candle(manager, ex, position, *bar)
        stops.append(position.stop_price)
    assert stops == sorted(stops)  # trailing stop tidak pernah turun
    position = await candle(manager, ex, position, 108.0, 108.2, 104.0, 104.5)
    assert position.status == CLOSED and position.exit_reason == "trailing_stop"
    assert position.realized_pnl > 0 and position.r_multiple > 1
    assert ex.equity() - start_equity == pytest.approx(position.realized_pnl, abs=1e-9)
    assert store.get(position.id).realized_pnl == pytest.approx(position.realized_pnl)


async def test_kena_stop_loss_rugi_satu_r(store):
    manager, ex = build(store)
    position = await manager.open_position(make_signal(), 2.0)
    position = await candle(manager, ex, position, 100, 100.5, 97.5, 98.2)
    assert position.status == CLOSED and position.exit_reason == "stop_loss"
    assert position.realized_pnl == pytest.approx(-position.risk_amount, rel=1e-6)
    assert position.r_multiple == pytest.approx(-1.0, rel=1e-6)
    assert await ex.open_orders() == []


async def test_tp_dan_stop_satu_candle_dianggap_rugi(store):
    manager, ex = build(store)
    position = await manager.open_position(make_signal(), 2.0)
    position = await candle(manager, ex, position, 100, 105, 97, 101)
    assert position.status == CLOSED and position.realized_pnl < 0


async def test_gap_di_bawah_stop_rugi_lebih_dari_satu_r(store):
    manager, ex = build(store)
    position = await manager.open_position(make_signal(), 2.0)
    position = await candle(manager, ex, position, 95, 95.5, 94, 94.5)
    assert position.status == CLOSED and position.r_multiple < -1


async def test_partial_fill_menyesuaikan_ukuran_sl_tp(store):
    ex = paper_executor()
    ex.fill_ratio[SYMBOL] = 0.5
    manager, _ = build(store, ex)
    position = await manager.open_position(make_signal(), 2.0)
    assert position.planned_qty == 2.0 and position.qty == 0.999  # terisi 1.0 dikurangi fee
    assert any(e.startswith("[PARTIAL FILL]") for e in manager.events)
    orders = await ex.open_orders()
    oco_qty = next(o.amount for o in orders if o.type == "limit_maker")
    runner_qty = next(o.amount for o in orders if o.client_id == position.stop_b_client_id)
    assert oco_qty + runner_qty == pytest.approx(position.qty)
    assert (oco_qty, runner_qty) == (0.499, 0.5)


async def test_entry_tidak_terisi_gagal_dan_tercatat(store):
    manager, ex = build(store, paper_executor(slippage=0.01), entry_max_slippage=0.0)
    position = await manager.open_position(make_signal(), 2.0)
    assert position.status == FAILED and position.exit_reason == "order entry tidak terisi"
    saved = store.get(position.id)
    assert saved.entry_client_id.startswith("tb-en-")  # write-ahead: client id dicatat sebelum order
    assert store.orders(position.id)[0]["role"] == "entry"
    assert ex.free["USDT"] == 1000


async def test_posisi_kecil_tidak_dibagi(store):
    manager, ex = build(store)
    position = await manager.open_position(make_signal(), 0.08)  # $8: separuhnya di bawah min notional
    assert position.tp1_qty == position.qty and not position.stop_b_client_id
    assert len(await ex.open_orders()) == 2  # hanya dua leg OCO


async def test_terisi_terlalu_sedikit_jadi_dust(store):
    ex = paper_executor()
    ex.fill_ratio[SYMBOL] = 0.04
    manager, _ = build(store, ex)
    position = await manager.open_position(make_signal(), 1.0)
    assert position.status == DUST


async def test_tanpa_oco_take_profit_dipantau_bot(store):
    ex = PaperExecutor({SYMBOL: sol_rules(oco_allowed=False)}, starting_balance=1000)
    ex.set_price(SYMBOL, 100)
    manager, _ = build(store, ex)
    position = await manager.open_position(make_signal(), 2.0)
    assert position.tp_mode == "software" and position.stop_a_client_id and position.stop_b_client_id
    position = await candle(manager, ex, position, 100, 104.5, 99.9, 104.2)
    assert position.status == TP1_HIT and position.qty == pytest.approx(0.999)
    assert position.exit_reason == "take_profit_1"


async def test_oco_dibatalkan_manual_tp1_dipantau_bot(store):
    manager, ex = build(store)
    position = await manager.open_position(make_signal(), 2.0)
    await ex.cancel_oco(SYMBOL, position.oco_client_id)  # dibatalkan manual di aplikasi Binance
    position = await manager.sync(position)
    assert position.status == OPEN and position.tp_mode == "software"
    stops = [o for o in await ex.open_orders() if o.type == "stop_loss"]
    assert sum(o.amount for o in stops) == pytest.approx(1.998)  # seluruh posisi tetap terlindungi stop
    position = await candle(manager, ex, position, 100, 104.5, 99.9, 104.2)
    assert position.status == TP1_HIT and position.breakeven and position.exit_reason == "take_profit_1"
    assert position.qty == pytest.approx(0.999)
    stops = [o for o in await ex.open_orders() if o.type == "stop_loss"]
    assert len(stops) == 1 and stops[0].amount == pytest.approx(0.999)


async def test_tp1_software_gagal_jual_proteksi_dikembalikan(store, monkeypatch):
    ex = PaperExecutor({SYMBOL: sol_rules(oco_allowed=False)}, starting_balance=1000)
    ex.set_price(SYMBOL, 100)
    manager, _ = build(store, ex)
    position = await manager.open_position(make_signal(), 2.0)

    async def bursa_menolak(*args, **kwargs):
        raise OrderError("bursa sedang maintenance")

    monkeypatch.setattr(ex, "sell_market", bursa_menolak)
    position = await candle(manager, ex, position, 100, 104.5, 99.9, 104.2)
    assert position.status == OPEN and position.qty == pytest.approx(1.998)
    assert any(e.startswith("[EXIT GAGAL]") for e in manager.events)
    stops = [o for o in await ex.open_orders() if o.type == "stop_loss"]
    assert sum(o.amount for o in stops) == pytest.approx(1.998) and stops[0].stop_price == 98.0
    monkeypatch.undo()
    position = await candle(manager, ex, position, 104.2, 104.8, 104.0, 104.6)  # dicoba lagi
    assert position.status == TP1_HIT and position.qty == pytest.approx(0.999)


async def test_trailing_di_atas_harga_langsung_jual_market(store):
    manager, ex = build(store)
    position = await manager.open_position(make_signal(), 2.0)
    position = await candle(manager, ex, position, 100, 104.5, 99.8, 104.2)
    position = await candle(manager, ex, position, 104.2, 120, 104.1, 105)  # ekor panjang: stop baru 118 > harga
    assert position.status == CLOSED and position.exit_reason == "trailing_stop"


async def test_tutup_manual(store):
    manager, ex = build(store)
    position = await manager.open_position(make_signal(), 2.0)
    position = await manager.close_position(position, "manual")
    assert position.status == CLOSED and position.exit_reason == "manual"
    assert await ex.open_orders() == [] and ex.locked.get("SOL", 0.0) == 0.0


async def test_try_open_lewat_risk_manager_dan_sizing(tmp_path, store):
    settings = make_settings(tmp_path)
    manager, ex = build(store)
    risk = RiskManager(settings, store, "paper", kill_file=tmp_path / "STOP")
    assert (await manager.try_open(make_signal(rejections=("skor_rendah",)), risk, settings)).reason == "sinyal bukan entry"

    (tmp_path / "STOP").write_text("")
    blocked = await manager.try_open(make_signal(resistance=(120, 121)), risk, settings)
    assert not blocked.opened and "kill switch" in blocked.reason
    (tmp_path / "STOP").unlink()

    ex.set_price(SYMBOL, 103.0)  # harga sudah naik mendekati resistance 104
    moved = await manager.try_open(make_signal(resistance=(104, 105)), risk, settings)
    assert not moved.opened and "harga sudah bergerak" in moved.reason

    ex.set_price(SYMBOL, 100.0)
    attempt = await manager.try_open(make_signal(resistance=(120, 121)), risk, settings, btc_corr=0.9)
    assert attempt.opened and attempt.position.btc_corr == 0.9
    assert attempt.position.planned_qty * 100 <= 1000 / 5 + 1e-6  # porsi satu slot
    assert attempt.position.features["rsi"] == 45.0 and attempt.position.pattern == "fib_618"


async def test_try_open_ditolak_jika_modal_terlalu_kecil(tmp_path, store):
    settings = make_settings(tmp_path)
    manager, _ = build(store, paper_executor(balance=8))
    attempt = await manager.try_open(make_signal(resistance=(120, 121)), RiskManager(settings, store, "paper", kill_file=tmp_path / "STOP"), settings)
    assert not attempt.opened and "terlalu kecil" in attempt.reason


# ----------------------------------------------------------------------
# Jalur live dengan mock ccxt: request yang dikirim ke Binance
# ----------------------------------------------------------------------
async def test_live_partial_fill_mengirim_oco_dan_stop_sesuai_jumlah_terisi(tmp_path, store, mock_exchange, fake_time):
    from helpers import install_markets
    from test_orders import binance_market

    from core.exchange import ExchangeClient
    from core.orders import LiveExecutor

    settings = make_settings(tmp_path, trading_mode="live", binance_api_key="kunci-palsu-123", binance_api_secret="rahasia-palsu-456")
    install_markets(mock_exchange, {SYMBOL: binance_market()})
    await mock_exchange.load_markets()
    mock_exchange.options = {}
    mock_exchange.fetch_ticker.return_value = {"bid": 99.99, "ask": 100.0, "last": 100.0}

    async def create_order(symbol, type_, side, amount, price, params):
        if side == "buy":  # IOC hanya terisi 1.2 dari 2.0, fee 0.1% dipotong dalam SOL
            return {"id": "1", "clientOrderId": params["newClientOrderId"], "symbol": symbol, "side": "buy", "type": "limit",
                    "status": "expired", "amount": amount, "filled": 1.2, "average": 100.02,
                    "fees": [{"cost": 0.0012, "currency": "SOL"}]}
        return {"id": "2", "clientOrderId": params["newClientOrderId"], "symbol": symbol, "side": side, "type": "stop_loss",
                "status": "open", "amount": amount, "filled": 0.0, "stopPrice": params.get("stopLossPrice")}

    mock_exchange.create_order.side_effect = create_order
    mock_exchange.private_post_orderlist_oco.return_value = {"orderListId": 5, "orderReports": []}
    client = ExchangeClient(settings, mock_exchange, sleep=fake_time.sleep, monotonic=fake_time.monotonic, jitter=0)
    manager = PositionManager(store, LiveExecutor(client, confirmed=True))

    position = await manager.open_position(make_signal(), 2.0)
    assert position.status == OPEN and position.qty == 1.198  # 1.2 - fee 0.0012
    entry_call = mock_exchange.create_order.await_args_list[0]
    assert entry_call.args[:5] == (SYMBOL, "limit", "buy", 2.0, 100.3)  # ask + 0.3%
    assert entry_call.args[5]["timeInForce"] == "IOC"
    oco = mock_exchange.private_post_orderlist_oco.await_args.args[0]
    assert (oco["quantity"], oco["abovePrice"], oco["belowStopPrice"], oco["belowType"]) == ("0.599", "104", "98", "STOP_LOSS")
    runner_call = mock_exchange.create_order.await_args_list[1]
    assert runner_call.args[:4] == (SYMBOL, "market", "sell", 0.599)
    assert runner_call.args[5]["stopLossPrice"] == 98.0
    assert store.get(position.id).stop_b_client_id == runner_call.args[5]["newClientOrderId"]


async def test_dust_dicatat_terpisah_dari_pnl(store):
    ex = PaperExecutor({SYMBOL: sol_rules(step_size=0.1, min_qty=0.1)}, starting_balance=1000)
    ex.set_price(SYMBOL, 100)
    manager, _ = build(store, ex)
    start_equity = ex.equity()
    position = await manager.open_position(make_signal(), 2.0)
    assert position.qty == 1.9 and position.dust_qty == pytest.approx(0.098)  # 2 - fee 0.002, dibulatkan ke step 0.1
    assert position.cost_per_unit == pytest.approx(position.cost_quote / 1.998)
    assert position.risk_amount == pytest.approx(1.9 * (position.cost_per_unit - 98 * 0.9995 * 0.999))
    position = await manager.close_position(position, "manual")
    dust_value = ex.equity() - ex.free["USDT"]
    assert dust_value == pytest.approx(0.098 * 100)  # dust tetap di akun dan tetap bernilai
    expected = position.realized_pnl + dust_value - position.dust_qty * position.cost_per_unit
    assert ex.equity() - start_equity == pytest.approx(expected, abs=1e-9)
