"""Test simulasi bursa: saldo virtual, fee, slippage, IOC partial fill, stop, OCO, urutan pesimis."""

from __future__ import annotations

import pytest
from helpers import paper_executor

from core.orders import OrderError

SYMBOL = "SOL/USDT"


async def test_entry_ioc_terisi_penuh_dengan_fee_dan_slippage():
    ex = paper_executor(balance=1000, price=100, fee_rate=0.001, slippage=0.0005)
    order = await ex.buy_limit_ioc(SYMBOL, 2.0, 100.3, "tb-en-1-a")
    assert (order.status, order.filled, order.average) == ("closed", 2.0, pytest.approx(100.05))
    assert order.fee_base == pytest.approx(0.002)  # fee beli dipotong dari coin
    balances = await ex.balances()
    assert balances["USDT"][0] == pytest.approx(1000 - 2.0 * 100.05)
    assert balances["SOL"][0] == pytest.approx(1.998)


async def test_entry_ioc_terisi_sebagian_atau_tidak_terisi():
    ex = paper_executor()
    ex.fill_ratio[SYMBOL] = 0.6
    partial = await ex.buy_limit_ioc(SYMBOL, 2.0, 100.3, "tb-en-1-a")
    assert (partial.status, partial.filled) == ("expired", 1.2)
    ex.fill_ratio[SYMBOL] = 1.0
    none = await ex.buy_limit_ioc(SYMBOL, 1.0, 99.9, "tb-en-2-a")  # batas harga di bawah ask
    assert (none.status, none.filled) == ("expired", 0.0)


async def test_entry_ditolak_jika_saldo_kurang_atau_di_bawah_min_notional():
    ex = paper_executor(balance=50)
    assert (await ex.buy_limit_ioc(SYMBOL, 1.0, 100.3, "tb-en-1-a")).status == "rejected"
    assert (await ex.buy_limit_ioc(SYMBOL, 0.01, 100.3, "tb-en-2-a")).status == "rejected"  # $1 < min notional
    with pytest.raises(OrderError, match="sudah dipakai"):
        await ex.buy_limit_ioc(SYMBOL, 0.1, 100.3, "tb-en-2-a")


async def test_jual_market_fee_dalam_usdt():
    ex = paper_executor(price=100, slippage=0.0)
    await ex.buy_limit_ioc(SYMBOL, 1.0, 100.3, "tb-en-1-a")
    order = await ex.sell_market(SYMBOL, 0.999, "tb-ex-1-a")
    assert order.status == "closed" and order.fee_quote == pytest.approx(0.0999)
    assert (await ex.sell_market(SYMBOL, 5.0, "tb-ex-2-a")).status == "rejected"


async def test_stop_loss_mengunci_saldo_dan_terpicu_dengan_gap():
    ex = paper_executor(price=100, slippage=0.001)
    await ex.buy_limit_ioc(SYMBOL, 1.0, 100.3, "tb-en-1-a")
    with pytest.raises(OrderError, match="langsung terpicu"):
        await ex.place_stop_loss(SYMBOL, 0.5, 100.5, "tb-sb-1-a")
    await ex.place_stop_loss(SYMBOL, 0.999, 98.0, "tb-sb-1-b")
    assert ex.locked["SOL"] == pytest.approx(0.999) and ex.free["SOL"] == pytest.approx(0.0)
    assert ex.process_candle(SYMBOL, 99.5, 99.8, 98.5, 98.7) == []  # belum menyentuh stop
    fills = ex.process_candle(SYMBOL, 97.0, 97.5, 96.0, 96.5)  # dibuka gap di bawah stop
    assert fills[0].average == pytest.approx(97.0 * 0.999)  # terisi di harga open, bukan di stop
    assert ex.locked["SOL"] == 0.0


async def test_batal_stop_mengembalikan_saldo():
    ex = paper_executor()
    await ex.buy_limit_ioc(SYMBOL, 1.0, 100.3, "tb-en-1-a")
    await ex.place_stop_loss(SYMBOL, 0.999, 98.0, "tb-sb-1-a")
    canceled = await ex.cancel_order(SYMBOL, "tb-sb-1-a")
    assert canceled.status == "canceled"
    assert ex.free["SOL"] == pytest.approx(0.999) and ex.locked["SOL"] == 0.0
    assert (await ex.fetch_order(SYMBOL, "tidak-ada")).status == "not_found"


@pytest.fixture
async def oco_setup():
    ex = paper_executor(price=100, slippage=0.0)
    await ex.buy_limit_ioc(SYMBOL, 1.0, 100.3, "tb-en-1-a")
    await ex.place_oco_exit(SYMBOL, 0.999, 104.0, 98.0, "tb-oc-1-a", "tb-tp-1-a", "tb-sa-1-a")
    return ex


async def test_oco_take_profit_membatalkan_stop(oco_setup):
    ex = oco_setup
    fills = ex.process_candle(SYMBOL, 100, 104.5, 99.5, 104.2)
    assert [f.client_id for f in fills] == ["tb-tp-1-a"]
    assert fills[0].average == 104.0
    assert (await ex.fetch_order(SYMBOL, "tb-sa-1-a")).status == "canceled"
    assert ex.locked["SOL"] == 0.0


async def test_oco_stop_membatalkan_take_profit(oco_setup):
    ex = oco_setup
    fills = ex.process_candle(SYMBOL, 100, 100.5, 97.5, 98.2)
    assert [f.client_id for f in fills] == ["tb-sa-1-a"]
    assert (await ex.fetch_order(SYMBOL, "tb-tp-1-a")).status == "canceled"


async def test_tp_dan_stop_dalam_satu_candle_dianggap_stop_dulu(oco_setup):
    ex = oco_setup
    fills = ex.process_candle(SYMBOL, 100, 105, 97, 101)
    assert [f.client_id for f in fills] == ["tb-sa-1-a"]  # asumsi pesimis


async def test_oco_ditolak_jika_harga_tidak_masuk_akal_dan_batal_oco():
    ex = paper_executor(price=100)
    await ex.buy_limit_ioc(SYMBOL, 1.0, 100.3, "tb-en-1-a")
    with pytest.raises(OrderError, match="take profit"):
        await ex.place_oco_exit(SYMBOL, 0.999, 99.0, 98.0, "tb-oc-1-a", "tb-tp-1-a", "tb-sa-1-a")
    await ex.place_oco_exit(SYMBOL, 0.999, 104.0, 98.0, "tb-oc-2-a", "tb-tp-2-a", "tb-sa-2-a")
    await ex.cancel_oco(SYMBOL, "tb-oc-2-a")
    assert ex.free["SOL"] == pytest.approx(0.999)
    assert await ex.open_orders() == []


async def test_ekuitas_mengikuti_harga():
    ex = paper_executor(balance=1000, price=100, slippage=0.0, fee_rate=0.0)
    await ex.buy_limit_ioc(SYMBOL, 2.0, 100.3, "tb-en-1-a")
    assert ex.equity() == pytest.approx(1000)
    ex.set_price(SYMBOL, 110)
    assert ex.equity() == pytest.approx(1020)
