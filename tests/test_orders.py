"""Test aturan presisi Binance, normalisasi order, client id, dan LiveExecutor (mock ccxt)."""

from __future__ import annotations

import re

import ccxt
import pytest
from helpers import install_markets, make_market, make_settings

from core.exchange import ExchangeClient
from core.orders import (
    LiveExecutor,
    OrderError,
    SymbolRules,
    _fmt,
    floor_to_step,
    is_bot_order,
    make_client_id,
    normalize_order,
)

BINANCE_FILTERS = [
    {"filterType": "PRICE_FILTER", "minPrice": "0.01000000", "maxPrice": "1000000.00000000", "tickSize": "0.01000000"},
    {"filterType": "LOT_SIZE", "minQty": "0.00100000", "maxQty": "9000.00000000", "stepSize": "0.00100000"},
    {"filterType": "MARKET_LOT_SIZE", "minQty": "0.00000000", "maxQty": "1000.00000000", "stepSize": "0.00000000"},
    {"filterType": "NOTIONAL", "minNotional": "5.00000000", "applyMinToMarket": True, "maxNotional": "9000000.00000000"},
]


def binance_market(order_types=None, oco=True):
    market = make_market("SOL")
    market["info"].update({
        "filters": BINANCE_FILTERS,
        "orderTypes": order_types or ["LIMIT", "LIMIT_MAKER", "MARKET", "STOP_LOSS", "STOP_LOSS_LIMIT"],
        "ocoAllowed": oco,
    })
    return market


# ----------------------------------------------------------------------
# Aturan presisi
# ----------------------------------------------------------------------
def test_aturan_dibaca_dari_filter_binance():
    rules = SymbolRules.from_market(binance_market())
    assert (rules.step_size, rules.tick_size, rules.min_qty, rules.max_qty, rules.min_notional) == (0.001, 0.01, 0.001, 9000.0, 5.0)
    assert rules.supports_stop_market and rules.oco_allowed


def test_aturan_cadangan_dari_precision_ccxt():
    market = make_market("ABC")
    market["precision"] = {"amount": 0.1, "price": 0.0001}
    market["limits"] = {"amount": {"min": 0.1, "max": None}, "cost": {"min": 10.0}}
    rules = SymbolRules.from_market(market)
    assert (rules.step_size, rules.tick_size, rules.min_qty, rules.min_notional) == (0.1, 0.0001, 0.1, 10.0)


def test_pembulatan_presisi_tanpa_galat_float():
    rules = SymbolRules.from_market(binance_market())
    assert rules.qty_down(1.23456) == 1.234
    assert rules.qty_down(0.0009) == 0.0
    assert floor_to_step(0.3, 0.1) == 0.3  # float biasa: 0.3 // 0.1 = 2.0
    assert rules.price_down(100.019) == 100.01
    assert rules.price_up(100.011) == 100.02
    assert rules.price_round(100.015) == 100.02


def test_cek_min_notional_dan_jumlah():
    rules = SymbolRules.from_market(binance_market())
    assert rules.check(0.04, 100) == "nilai order 4 di bawah min notional 5"
    assert "di bawah minimum" in rules.check(0.0005, 100_000)
    assert rules.check(0.05, 100) is None
    assert rules.is_sellable(0.06, 100, buffer=1.1) and not rules.is_sellable(0.05, 100, buffer=1.1)


# ----------------------------------------------------------------------
# Normalisasi order dan client id
# ----------------------------------------------------------------------
def test_normalisasi_order_dan_fee_per_mata_uang():
    order = {
        "id": "123", "clientOrderId": "tb-en-1-abc", "symbol": "SOL/USDT", "side": "buy", "type": "limit",
        "status": "expired", "amount": 2.0, "filled": 1.2, "average": 100.5, "price": 100.8,
        "fees": [{"cost": 0.0012, "currency": "SOL"}, {"cost": 0.01, "currency": "BNB"}],
    }
    result = normalize_order(order, "SOL", "USDT")
    assert (result.status, result.filled, result.average, result.fee_base) == ("expired", 1.2, 100.5, 0.0012)
    assert result.fee_other == {"BNB": 0.01}
    assert result.remaining == pytest.approx(0.8) and result.is_done
    sell = normalize_order({"id": "9", "clientOrderId": "x", "status": "closed", "filled": 1, "average": 99,
                            "fee": {"cost": 0.099, "currency": "USDT"}, "info": {"stopPrice": "98.50"}}, "SOL", "USDT")
    assert sell.fee_quote == 0.099 and sell.stop_price == 98.5


def test_client_id_unik_dan_sesuai_aturan_binance():
    ids = {make_client_id("sb", 123456) for _ in range(5000)}
    assert len(ids) == 5000  # dibuat dalam milidetik yang sama pun tetap unik
    for cid in ids:
        assert len(cid) <= 36
        assert re.fullmatch(r"[.A-Z:/a-z0-9_-]{1,36}", cid)
        assert is_bot_order(cid)
    assert not is_bot_order("web_abc") and not is_bot_order(None)


def test_format_angka_tanpa_notasi_ilmiah():
    assert _fmt(0.00001) == "0.00001"
    assert _fmt(100.0) == "100"
    assert _fmt(1234.5) == "1234.5"


# ----------------------------------------------------------------------
# LiveExecutor dengan mock ccxt
# ----------------------------------------------------------------------
@pytest.fixture
def live(tmp_path, mock_exchange, fake_time):
    settings = make_settings(tmp_path, trading_mode="live", binance_api_key="kunci-palsu-123", binance_api_secret="rahasia-palsu-456")
    install_markets(mock_exchange, {"SOL/USDT": binance_market()})
    mock_exchange.options = {}
    mock_exchange.parse_order.side_effect = lambda raw, market=None: {
        "id": str(raw["orderId"]), "clientOrderId": raw["clientOrderId"], "symbol": "SOL/USDT", "side": "sell",
        "type": raw["type"].lower(), "status": "open", "amount": float(raw["origQty"]), "filled": 0.0,
        "price": float(raw.get("price") or 0) or None, "info": raw,
    }
    client = ExchangeClient(settings, mock_exchange, sleep=fake_time.sleep, monotonic=fake_time.monotonic, jitter=0)
    return LiveExecutor(client, confirmed=True)


def ccxt_order(client_id, status="closed", filled=1.0, average=100.0, amount=1.0, fees=None):
    return {"id": "1", "clientOrderId": client_id, "symbol": "SOL/USDT", "side": "buy", "type": "limit",
            "status": status, "amount": amount, "filled": filled, "average": average, "fees": fees or []}


def test_live_executor_wajib_konfirmasi(tmp_path, mock_exchange):
    client = ExchangeClient(make_settings(tmp_path), mock_exchange)
    with pytest.raises(OrderError, match="confirm-live"):
        LiveExecutor(client, confirmed=False)


async def test_entry_limit_ioc_dengan_client_id(live, mock_exchange):
    mock_exchange.create_order.return_value = ccxt_order("tb-en-1-a", filled=1.5, amount=2.0, status="expired",
                                                          fees=[{"cost": 0.0015, "currency": "SOL"}])
    result = await live.buy_limit_ioc("SOL/USDT", 2.0, 100.3, "tb-en-1-a")
    mock_exchange.create_order.assert_awaited_once_with(
        "SOL/USDT", "limit", "buy", 2.0, 100.3, {"timeInForce": "IOC", "newClientOrderId": "tb-en-1-a"}
    )
    assert (result.filled, result.fee_base, result.status) == (1.5, 0.0015, "expired")


async def test_timeout_tidak_mengirim_ulang_order_tetapi_cek_lewat_client_id(live, mock_exchange):
    mock_exchange.create_order.side_effect = ccxt.RequestTimeout("binance timeout")
    mock_exchange.fetch_order.return_value = ccxt_order("tb-en-1-a", filled=2.0, amount=2.0)
    result = await live.buy_limit_ioc("SOL/USDT", 2.0, 100.3, "tb-en-1-a")
    assert mock_exchange.create_order.await_count == 1  # tidak ada order ganda
    mock_exchange.fetch_order.assert_awaited_with("", "SOL/USDT", {"clientOrderId": "tb-en-1-a"})
    assert result.filled == 2.0


async def test_timeout_dan_order_tidak_ada_berarti_gagal(live, mock_exchange):
    mock_exchange.create_order.side_effect = ccxt.RequestTimeout("binance timeout")
    mock_exchange.fetch_order.side_effect = ccxt.OrderNotFound("binance Order does not exist")
    with pytest.raises(OrderError, match="tidak terkirim"):
        await live.buy_limit_ioc("SOL/USDT", 2.0, 100.3, "tb-en-1-a")


async def test_stop_loss_market_dan_cadangan_stop_limit(live, mock_exchange):
    mock_exchange.create_order.return_value = ccxt_order("tb-sb-1-a", status="open", filled=0.0)
    await live.place_stop_loss("SOL/USDT", 1.0, 97.999, "tb-sb-1-a")
    args = mock_exchange.create_order.await_args.args
    assert args[:5] == ("SOL/USDT", "market", "sell", 1.0, None)
    assert args[5] == {"stopLossPrice": 97.99, "newClientOrderId": "tb-sb-1-a"}

    # Pair tanpa STOP_LOSS market: dipakai STOP_LOSS_LIMIT.
    mock_exchange.markets = {"SOL/USDT": binance_market(order_types=["LIMIT", "LIMIT_MAKER", "MARKET", "STOP_LOSS_LIMIT"])}
    live._rules.clear()
    await live.place_stop_loss("SOL/USDT", 1.0, 98.0, "tb-sb-1-b")
    args = mock_exchange.create_order.await_args.args
    assert args[:5] == ("SOL/USDT", "limit", "sell", 1.0, 97.02)  # batas 1% di bawah stop
    assert args[5]["stopLossPrice"] == 98.0 and args[5]["timeInForce"] == "GTC"


async def test_oco_take_profit_limit_maker_dan_stop_market(live, mock_exchange):
    mock_exchange.private_post_orderlist_oco.return_value = {
        "orderListId": 77,
        "orderReports": [
            {"orderId": 1, "clientOrderId": "tb-tp-1-a", "type": "LIMIT_MAKER", "origQty": "0.999", "price": "104.00"},
            {"orderId": 2, "clientOrderId": "tb-sa-1-a", "type": "STOP_LOSS", "origQty": "0.999", "price": "0"},
        ],
    }
    result = await live.place_oco_exit("SOL/USDT", 0.9994, 103.991, 98.009, "tb-oc-1-a", "tb-tp-1-a", "tb-sa-1-a")
    request = mock_exchange.private_post_orderlist_oco.await_args.args[0]
    assert request == {
        "symbol": "SOLUSDT", "side": "SELL", "quantity": "0.999", "listClientOrderId": "tb-oc-1-a",
        "aboveType": "LIMIT_MAKER", "abovePrice": "104", "aboveClientOrderId": "tb-tp-1-a",
        "belowStopPrice": "98", "belowClientOrderId": "tb-sa-1-a", "newOrderRespType": "FULL", "belowType": "STOP_LOSS",
    }
    assert result.take_profit.client_id == "tb-tp-1-a" and result.stop.client_id == "tb-sa-1-a"
    assert result.list_id == "77"


async def test_fetch_dan_cancel_order_yang_sudah_selesai(live, mock_exchange):
    mock_exchange.fetch_order.side_effect = ccxt.OrderNotFound("binance Order does not exist")
    assert (await live.fetch_order("SOL/USDT", "tb-x")).status == "not_found"
    mock_exchange.fetch_order.side_effect = None
    mock_exchange.fetch_order.return_value = ccxt_order("tb-sb-1-a", status="closed", filled=1.0)
    mock_exchange.cancel_order.side_effect = ccxt.OperationRejected('binance {"code":-2011,"msg":"Unknown order sent."}')
    result = await live.cancel_order("SOL/USDT", "tb-sb-1-a")
    assert result.status == "closed" and result.filled == 1.0  # ternyata sudah terisi


async def test_saldo_live(live, mock_exchange):
    mock_exchange.fetch_balance.return_value = {"free": {"USDT": 90.0, "SOL": 0.0}, "total": {"USDT": 90.0, "SOL": 1.0}}
    assert await live.balances() == {"USDT": (90.0, 90.0), "SOL": (0.0, 1.0)}
