"""Eksekusi order spot Binance: presisi, min notional, client order id, dan executor.

Keputusan desain:

* Aturan tiap pair (step size, tick size, min notional) dibaca langsung dari
  filter Binance dan dibulatkan dengan Decimal agar tidak ada galat float
  (misal 0.1 + 0.2). Jumlah coin selalu dibulatkan KE BAWAH ke step size.
* Setiap order punya client order id unik yang dicatat di database SEBELUM
  order dikirim. Jika request timeout, status order dicek lewat id tersebut,
  bukan dikirim ulang, sehingga tidak terjadi order ganda.
* Entry memakai LIMIT IOC dengan batas slippage: terisi segera (penuh atau
  sebagian) atau batal, tidak pernah membeli di harga liar.
* Exit memakai OCO Binance: take profit berupa LIMIT_MAKER dan stop loss berupa
  STOP_LOSS (market) sehingga stop pasti tereksekusi saat harga anjlok. Jika
  pair tidak mengizinkan STOP_LOSS market, dipakai STOP_LOSS_LIMIT dengan
  batas harga lebih rendah sebagai cadangan.
* Executor live dan PaperExecutor (core/paper_exchange.py) punya antarmuka yang
  sama, sehingga logika posisi identik di mode paper, backtest, dan live.
"""

from __future__ import annotations

import itertools
import logging
import math
import secrets
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from typing import Any, Protocol

import ccxt

from core.exchange import ExchangeClient

log = logging.getLogger(__name__)

CLIENT_ID_PREFIX = "tb"  # penanda order milik bot ini
DEFAULT_ORDER_TYPES = frozenset({"LIMIT", "LIMIT_MAKER", "MARKET", "STOP_LOSS", "STOP_LOSS_LIMIT"})
DONE_STATUSES = frozenset({"closed", "canceled", "expired", "rejected", "not_found"})


class OrderError(RuntimeError):
    """Order tidak bisa dibuat atau statusnya tidak bisa dipastikan."""


# ----------------------------------------------------------------------
# Pembulatan presisi
# ----------------------------------------------------------------------
def _dec(value: float | str) -> Decimal:
    return Decimal(str(value))


def _round_to(value: float, step: float, rounding: str) -> float:
    if step <= 0:
        return float(value)
    units = (_dec(value) / _dec(step)).to_integral_value(rounding=rounding)
    return float(units * _dec(step))


def floor_to_step(value: float, step: float) -> float:
    return _round_to(value, step, ROUND_FLOOR)


def ceil_to_step(value: float, step: float) -> float:
    return _round_to(value, step, ROUND_CEILING)


def round_to_step(value: float, step: float) -> float:
    return _round_to(value, step, ROUND_HALF_UP)


def _filter_float(filters: Mapping[str, Mapping[str, Any]], name: str, key: str) -> float | None:
    value = filters.get(name, {}).get(key)
    try:
        number = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return number if number else None  # "0.00000000" berarti tidak dibatasi


@dataclass(frozen=True)
class SymbolRules:
    """Aturan order satu pair dari exchangeInfo Binance."""

    symbol: str
    base: str
    quote: str
    step_size: float
    tick_size: float
    min_qty: float = 0.0
    max_qty: float = math.inf
    min_notional: float = 0.0
    order_types: frozenset[str] = DEFAULT_ORDER_TYPES
    oco_allowed: bool = True

    @classmethod
    def from_market(cls, market: Mapping[str, Any]) -> SymbolRules:
        info = market.get("info") or {}
        filters = {f.get("filterType"): f for f in info.get("filters") or [] if isinstance(f, Mapping)}
        precision = market.get("precision") or {}
        limits = market.get("limits") or {}

        def limit(group: str, key: str) -> float | None:
            value = (limits.get(group) or {}).get(key)
            return float(value) if value else None

        step = _filter_float(filters, "LOT_SIZE", "stepSize") or float(precision.get("amount") or 0)
        tick = _filter_float(filters, "PRICE_FILTER", "tickSize") or float(precision.get("price") or 0)
        min_notional = (
            _filter_float(filters, "NOTIONAL", "minNotional")
            or _filter_float(filters, "MIN_NOTIONAL", "minNotional")
            or limit("cost", "min")
            or 0.0
        )
        order_types = info.get("orderTypes")
        return cls(
            symbol=str(market["symbol"]),
            base=str(market.get("base", "")),
            quote=str(market.get("quote", "")),
            step_size=step,
            tick_size=tick,
            min_qty=_filter_float(filters, "LOT_SIZE", "minQty") or limit("amount", "min") or 0.0,
            max_qty=_filter_float(filters, "LOT_SIZE", "maxQty") or limit("amount", "max") or math.inf,
            min_notional=min_notional,
            order_types=frozenset(order_types) if order_types else DEFAULT_ORDER_TYPES,
            oco_allowed=bool(info.get("ocoAllowed", True)),
        )

    def qty_down(self, qty: float) -> float:
        return max(0.0, floor_to_step(qty, self.step_size))

    def price_down(self, price: float) -> float:
        return floor_to_step(price, self.tick_size)

    def price_up(self, price: float) -> float:
        return ceil_to_step(price, self.tick_size)

    def price_round(self, price: float) -> float:
        return round_to_step(price, self.tick_size)

    def check(self, qty: float, price: float) -> str | None:
        """Alasan order tidak valid menurut filter Binance, atau None jika valid."""
        if qty <= 0:
            return "jumlah 0"
        if qty < self.min_qty:
            return f"jumlah {qty:g} di bawah minimum {self.min_qty:g}"
        if qty > self.max_qty:
            return f"jumlah {qty:g} di atas maksimum {self.max_qty:g}"
        if qty * price < self.min_notional:
            return f"nilai order {qty * price:.4g} di bawah min notional {self.min_notional:g}"
        return None

    def is_sellable(self, qty: float, price: float, buffer: float = 1.0) -> bool:
        """True jika `qty` bisa dijual di harga `price` (dengan cadangan `buffer` untuk min notional)."""
        return qty >= self.min_qty and qty > 0 and qty * price >= self.min_notional * buffer

    @property
    def supports_stop_market(self) -> bool:
        return "STOP_LOSS" in self.order_types


# ----------------------------------------------------------------------
# Order yang dinormalisasi
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class OrderResult:
    client_id: str
    symbol: str
    side: str
    type: str
    status: str                      # open, closed, canceled, expired, rejected, not_found
    amount: float = 0.0
    filled: float = 0.0
    average: float | None = None     # harga rata rata terisi
    price: float | None = None
    stop_price: float | None = None
    fee_base: float = 0.0            # fee yang dipotong dalam coin (base)
    fee_quote: float = 0.0           # fee yang dipotong dalam USDT (quote)
    fee_other: Mapping[str, float] = field(default_factory=dict)  # misal BNB
    id: str | None = None
    timestamp: int | None = None

    @property
    def is_done(self) -> bool:
        return self.status in DONE_STATUSES

    @property
    def remaining(self) -> float:
        return max(self.amount - self.filled, 0.0)

    @property
    def cost(self) -> float:
        return self.filled * (self.average or 0.0)

    @classmethod
    def not_found(cls, symbol: str, client_id: str) -> OrderResult:
        return cls(client_id=client_id, symbol=symbol, side="", type="", status="not_found")


@dataclass(frozen=True)
class OcoResult:
    list_client_id: str
    take_profit: OrderResult
    stop: OrderResult
    list_id: str | None = None


def normalize_order(order: Mapping[str, Any], base: str, quote: str) -> OrderResult:
    """Ubah struktur order ccxt menjadi OrderResult (termasuk fee per mata uang)."""
    fees: list[Mapping[str, Any]] = [f for f in order.get("fees") or [] if f]
    if not fees and order.get("fee"):
        fees = [order["fee"]]
    fee_base = fee_quote = 0.0
    fee_other: dict[str, float] = {}
    for fee in fees:
        cost = float(fee.get("cost") or 0.0)
        currency = fee.get("currency")
        if currency == base:
            fee_base += cost
        elif currency == quote:
            fee_quote += cost
        elif currency:
            fee_other[currency] = fee_other.get(currency, 0.0) + cost
    info = order.get("info") or {}
    stop_price = order.get("stopPrice") or order.get("triggerPrice") or info.get("stopPrice")
    return OrderResult(
        client_id=str(order.get("clientOrderId") or info.get("clientOrderId") or ""),
        symbol=str(order.get("symbol") or ""),
        side=str(order.get("side") or ""),
        type=str(order.get("type") or info.get("type") or ""),
        status=str(order.get("status") or "open"),
        amount=float(order.get("amount") or 0.0),
        filled=float(order.get("filled") or 0.0),
        average=float(order["average"]) if order.get("average") else None,
        price=float(order["price"]) if order.get("price") else None,
        stop_price=float(stop_price) if stop_price and float(stop_price) > 0 else None,
        fee_base=fee_base,
        fee_quote=fee_quote,
        fee_other=fee_other,
        id=str(order["id"]) if order.get("id") is not None else None,
        timestamp=order.get("timestamp"),
    )


_CLIENT_ID_COUNTER = itertools.count()


def make_client_id(role: str, position_id: int | None = None) -> str:
    """Client order id unik, maksimal 36 karakter sesuai aturan Binance.

    Contoh: tb-en-12-lq3k9z8a0a1b2c3 (prefix bot, peran order, id posisi,
    waktu milidetik + penghitung + acak). Penghitung menjamin id unik walau
    dibuat dalam milidetik yang sama; waktu dan bagian acak menjamin unik antar restart.
    """
    stamp = _base36(int(time.time() * 1000))
    counter = _base36(next(_CLIENT_ID_COUNTER) % 1296).rjust(2, "0")
    pid = "x" if position_id is None else str(position_id)[-10:]
    return f"{CLIENT_ID_PREFIX}-{role}-{pid}-{stamp}{counter}{secrets.token_hex(3)}"[:36]


def _base36(number: int) -> str:
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    text = ""
    while number:
        number, rem = divmod(number, 36)
        text = digits[rem] + text
    return text or "0"


def is_bot_order(client_id: str | None) -> bool:
    return bool(client_id) and str(client_id).startswith(f"{CLIENT_ID_PREFIX}-")


# ----------------------------------------------------------------------
# Antarmuka executor
# ----------------------------------------------------------------------
class Executor(Protocol):
    """Operasi bursa yang dibutuhkan position manager (live dan paper)."""

    mode: str

    async def rules(self, symbol: str) -> SymbolRules: ...

    async def best_prices(self, symbol: str) -> tuple[float, float]: ...

    async def buy_limit_ioc(self, symbol: str, qty: float, limit_price: float, client_id: str) -> OrderResult: ...

    async def sell_market(self, symbol: str, qty: float, client_id: str) -> OrderResult: ...

    async def place_stop_loss(self, symbol: str, qty: float, stop_price: float, client_id: str) -> OrderResult: ...

    async def place_oco_exit(
        self, symbol: str, qty: float, take_profit: float, stop_price: float, list_client_id: str, tp_client_id: str, stop_client_id: str
    ) -> OcoResult: ...

    async def fetch_order(self, symbol: str, client_id: str) -> OrderResult: ...

    async def cancel_order(self, symbol: str, client_id: str) -> OrderResult: ...

    async def cancel_oco(self, symbol: str, list_client_id: str) -> None: ...

    async def balances(self) -> dict[str, tuple[float, float]]: ...

    async def open_orders(self, symbol: str | None = None) -> list[OrderResult]: ...


# ----------------------------------------------------------------------
# Executor live (Binance via ccxt)
# ----------------------------------------------------------------------
class LiveExecutor:
    """Order sungguhan ke Binance. Hanya bisa dibuat setelah mode live dikonfirmasi."""

    mode = "live"

    def __init__(self, client: ExchangeClient, *, confirmed: bool, stop_limit_buffer: float = 0.01, lookup_attempts: int = 3) -> None:
        if not confirmed:
            raise OrderError("LiveExecutor menolak dibuat tanpa konfirmasi mode live (--confirm-live)")
        self.client = client
        self.stop_limit_buffer = stop_limit_buffer
        self.lookup_attempts = lookup_attempts
        self._rules: dict[str, SymbolRules] = {}
        # fetch_open_orders tanpa simbol dipakai saat recovery; matikan peringatan ccxt-nya.
        options = getattr(client.exchange, "options", None)
        if isinstance(options, dict):
            options["warnOnFetchOpenOrdersWithoutSymbol"] = False

    async def rules(self, symbol: str) -> SymbolRules:
        if symbol not in self._rules:
            markets = self.client.markets or await self.client.load_markets()
            if symbol not in markets:
                raise OrderError(f"pair {symbol} tidak ada di Binance spot")
            self._rules[symbol] = SymbolRules.from_market(markets[symbol])
        return self._rules[symbol]

    async def best_prices(self, symbol: str) -> tuple[float, float]:
        ticker = await self.client.fetch_ticker(symbol)
        last = float(ticker.get("last") or ticker.get("close") or 0.0)
        bid = float(ticker.get("bid") or last)
        ask = float(ticker.get("ask") or last)
        if bid <= 0 or ask <= 0:
            raise OrderError(f"harga {symbol} tidak tersedia")
        return bid, ask

    async def _normalized(self, symbol: str, order: Mapping[str, Any]) -> OrderResult:
        rules = await self.rules(symbol)
        return normalize_order(order, rules.base, rules.quote)

    async def _create(self, symbol: str, client_id: str, *args: Any, params: dict[str, Any]) -> OrderResult:
        """Kirim order baru. Jika hasilnya tidak pasti (timeout/5xx), cek lewat client id."""
        params = {**params, "newClientOrderId": client_id}
        try:
            order = await self.client.call("create_order", symbol, *args, params, idempotent=False)
        except ccxt.NetworkError as exc:
            log.warning("Status order %s tidak pasti (%s), dicek lewat client id", client_id, exc)
            result = await self._lookup_after_uncertain(symbol, client_id)
            if result.status == "not_found":
                raise OrderError(f"order {client_id} tidak terkirim: {exc}") from exc
            return result
        return await self._normalized(symbol, order)

    async def _lookup_after_uncertain(self, symbol: str, client_id: str) -> OrderResult:
        for attempt in range(self.lookup_attempts):
            try:
                return await self.fetch_order(symbol, client_id)
            except ccxt.NetworkError:
                if attempt == self.lookup_attempts - 1:
                    raise
        return OrderResult.not_found(symbol, client_id)

    async def buy_limit_ioc(self, symbol: str, qty: float, limit_price: float, client_id: str) -> OrderResult:
        return await self._create(symbol, client_id, "limit", "buy", qty, limit_price, params={"timeInForce": "IOC"})

    async def sell_market(self, symbol: str, qty: float, client_id: str) -> OrderResult:
        return await self._create(symbol, client_id, "market", "sell", qty, None, params={})

    async def place_stop_loss(self, symbol: str, qty: float, stop_price: float, client_id: str) -> OrderResult:
        rules = await self.rules(symbol)
        stop = rules.price_down(stop_price)
        if rules.supports_stop_market:
            return await self._create(symbol, client_id, "market", "sell", qty, None, params={"stopLossPrice": stop})
        # Cadangan: STOP_LOSS_LIMIT dengan batas harga di bawah stop agar tetap terisi saat harga turun cepat.
        limit = rules.price_down(stop * (1 - self.stop_limit_buffer))
        log.warning("%s tidak mendukung STOP_LOSS market, memakai STOP_LOSS_LIMIT (batas %s)", symbol, limit)
        return await self._create(symbol, client_id, "limit", "sell", qty, limit, params={"stopLossPrice": stop, "timeInForce": "GTC"})

    async def place_oco_exit(
        self, symbol: str, qty: float, take_profit: float, stop_price: float, list_client_id: str, tp_client_id: str, stop_client_id: str
    ) -> OcoResult:
        rules = await self.rules(symbol)
        markets = self.client.markets or await self.client.load_markets()
        tp = rules.price_up(take_profit)
        stop = rules.price_down(stop_price)
        request: dict[str, Any] = {
            "symbol": markets[symbol]["id"],
            "side": "SELL",
            "quantity": _fmt(rules.qty_down(qty)),
            "listClientOrderId": list_client_id,
            "aboveType": "LIMIT_MAKER",
            "abovePrice": _fmt(tp),
            "aboveClientOrderId": tp_client_id,
            "belowStopPrice": _fmt(stop),
            "belowClientOrderId": stop_client_id,
            "newOrderRespType": "FULL",
        }
        if rules.supports_stop_market:
            request["belowType"] = "STOP_LOSS"
        else:
            request.update({
                "belowType": "STOP_LOSS_LIMIT",
                "belowPrice": _fmt(rules.price_down(stop * (1 - self.stop_limit_buffer))),
                "belowTimeInForce": "GTC",
            })
        try:
            response = await self.client.call("private_post_orderlist_oco", request, idempotent=False)
        except ccxt.NetworkError as exc:
            log.warning("Status OCO %s tidak pasti (%s), dicek lewat client id", list_client_id, exc)
            tp_order = await self._lookup_after_uncertain(symbol, tp_client_id)
            stop_order = await self._lookup_after_uncertain(symbol, stop_client_id)
            if tp_order.status == "not_found" and stop_order.status == "not_found":
                raise OrderError(f"OCO {list_client_id} tidak terkirim: {exc}") from exc
            return OcoResult(list_client_id, tp_order, stop_order)
        reports = {str(r.get("clientOrderId")): r for r in response.get("orderReports") or []}
        exchange = self.client.exchange
        market = markets[symbol]

        def leg(client_id: str, fallback_type: str) -> OrderResult:
            raw = reports.get(client_id)
            if raw is None:
                return OrderResult(client_id, symbol, "sell", fallback_type, "open", amount=rules.qty_down(qty))
            return normalize_order(exchange.parse_order(raw, market), rules.base, rules.quote)

        return OcoResult(
            list_client_id=list_client_id,
            take_profit=leg(tp_client_id, "limit"),
            stop=leg(stop_client_id, "stop_loss"),
            list_id=str(response.get("orderListId")) if response.get("orderListId") is not None else None,
        )

    async def fetch_order(self, symbol: str, client_id: str) -> OrderResult:
        try:
            order = await self.client.call("fetch_order", "", symbol, {"clientOrderId": client_id})
        except ccxt.OrderNotFound:
            return OrderResult.not_found(symbol, client_id)
        return await self._normalized(symbol, order)

    async def cancel_order(self, symbol: str, client_id: str) -> OrderResult:
        try:
            await self.client.call("cancel_order", "", symbol, {"clientOrderId": client_id})
        except (ccxt.OrderNotFound, ccxt.OperationRejected) as exc:
            # Order sudah terisi/batal sebelumnya: kembalikan status terakhirnya.
            log.info("Order %s tidak bisa dibatalkan (%s), status diperiksa ulang", client_id, exc)
        return await self.fetch_order(symbol, client_id)

    async def cancel_oco(self, symbol: str, list_client_id: str) -> None:
        markets = self.client.markets or await self.client.load_markets()
        try:
            await self.client.call(
                "private_delete_orderlist", {"symbol": markets[symbol]["id"], "listClientOrderId": list_client_id}
            )
        except (ccxt.OrderNotFound, ccxt.OperationRejected) as exc:
            log.info("OCO %s sudah tidak aktif (%s)", list_client_id, exc)

    async def balances(self) -> dict[str, tuple[float, float]]:
        balance = await self.client.fetch_balance()
        free = balance.get("free") or {}
        total = balance.get("total") or {}
        assets = set(free) | set(total)
        return {asset: (float(free.get(asset) or 0.0), float(total.get(asset) or 0.0)) for asset in assets}

    async def open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        orders = await self.client.call("fetch_open_orders", symbol)
        results = []
        for order in orders:
            sym = order.get("symbol") or symbol
            results.append(await self._normalized(sym, order))
        return results


def _fmt(value: float) -> str:
    """Format angka untuk API Binance tanpa notasi ilmiah."""
    text = format(_dec(value).normalize(), "f")
    return text


def summarize_fills(orders: Iterable[OrderResult]) -> tuple[float, float, float]:
    """(jumlah terisi, harga rata rata tertimbang, total fee quote) dari beberapa order."""
    filled = sum(o.filled for o in orders)
    if filled <= 0:
        return 0.0, 0.0, 0.0
    average = sum(o.filled * (o.average or 0.0) for o in orders) / filled
    return filled, average, sum(o.fee_quote for o in orders)
