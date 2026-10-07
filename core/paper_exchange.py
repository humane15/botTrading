"""Simulasi bursa untuk paper trading dan backtest.

PaperExecutor punya antarmuka yang sama dengan LiveExecutor, jadi position
manager menjalankan kode yang sama persis di semua mode. Aturan simulasi:

* Saldo virtual dalam USDT. Fee 0.1% (default) seperti Binance: saat beli fee
  dipotong dari coin yang diterima, saat jual fee dipotong dari USDT.
* Entry LIMIT IOC terisi di harga ask + slippage jika masih di bawah batas
  harga; bisa disimulasikan terisi sebagian (fill_ratio).
* Stop loss (market) terpicu jika low candle <= harga stop. Jika candle dibuka
  di bawah stop (gap), terisi di harga open, bukan di harga stop. Slippage
  selalu merugikan.
* Take profit (LIMIT_MAKER) terisi di harga TP jika high candle melewatinya.
* Jika dalam satu candle stop DAN take profit sama sama tersentuh, stop
  dianggap terjadi lebih dulu (asumsi pesimis, tidak melebih lebihkan hasil).
* Order yang langsung terpicu (stop >= harga sekarang) ditolak, sama seperti
  Binance.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from core.orders import OcoResult, OrderError, OrderResult, SymbolRules

EPSILON = 1e-12


@dataclass
class _SimOrder:
    client_id: str
    symbol: str
    side: str
    type: str                # limit, market, stop_loss, limit_maker
    status: str              # open, closed, canceled, expired, rejected
    amount: float
    price: float | None = None
    stop_price: float | None = None
    filled: float = 0.0
    average: float | None = None
    fee_base: float = 0.0
    fee_quote: float = 0.0
    list_client_id: str | None = None
    timestamp: int = 0

    def result(self) -> OrderResult:
        return OrderResult(
            client_id=self.client_id, symbol=self.symbol, side=self.side, type=self.type, status=self.status,
            amount=self.amount, filled=self.filled, average=self.average, price=self.price,
            stop_price=self.stop_price, fee_base=self.fee_base, fee_quote=self.fee_quote,
            id=self.client_id, timestamp=self.timestamp,
        )


@dataclass
class _OcoList:
    list_client_id: str
    symbol: str
    qty: float
    legs: tuple[str, str]  # (take profit, stop)
    active: bool = True


@dataclass
class Fill:
    """Catatan eksekusi simulasi (untuk laporan dan test)."""

    client_id: str
    symbol: str
    side: str
    qty: float
    price: float
    fee_quote: float
    fee_base: float
    reason: str
    timestamp: int


class PaperExecutor:
    mode = "paper"

    def __init__(
        self,
        rules: Mapping[str, SymbolRules] | None = None,
        *,
        quote: str = "USDT",
        starting_balance: float = 100.0,
        fee_rate: float = 0.001,
        slippage: float = 0.0005,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self._rules: dict[str, SymbolRules] = dict(rules or {})
        self.quote = quote
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.free: dict[str, float] = {quote: float(starting_balance)}
        self.locked: dict[str, float] = {}
        self.orders: dict[str, _SimOrder] = {}
        self.oco_lists: dict[str, _OcoList] = {}
        self.prices: dict[str, tuple[float, float]] = {}
        self.fill_ratio: dict[str, float] = {}
        self.fills: list[Fill] = []
        self._clock = clock or (lambda: int(time.time() * 1000))

    # ------------------------------------------------------------------
    # Data pasar simulasi
    # ------------------------------------------------------------------
    def add_rules(self, rules: SymbolRules) -> None:
        self._rules[rules.symbol] = rules

    def set_price(self, symbol: str, price: float, spread: float = 0.0) -> None:
        """Harga pasar sekarang; spread adalah selisih relatif bid/ask (misal 0.0002)."""
        self.prices[symbol] = (price * (1 - spread / 2), price * (1 + spread / 2))

    async def rules(self, symbol: str) -> SymbolRules:
        if symbol not in self._rules:
            raise OrderError(f"aturan pair {symbol} belum dimuat")
        return self._rules[symbol]

    async def best_prices(self, symbol: str) -> tuple[float, float]:
        if symbol not in self.prices:
            raise OrderError(f"harga {symbol} belum tersedia")
        return self.prices[symbol]

    # ------------------------------------------------------------------
    # Saldo
    # ------------------------------------------------------------------
    def _add(self, book: dict[str, float], asset: str, amount: float) -> None:
        book[asset] = book.get(asset, 0.0) + amount
        if abs(book[asset]) < EPSILON:
            book[asset] = 0.0

    def _lock(self, asset: str, qty: float) -> None:
        if self.free.get(asset, 0.0) + EPSILON < qty:
            raise OrderError(f"saldo {asset} tidak cukup: butuh {qty:g}, tersedia {self.free.get(asset, 0.0):g}")
        self._add(self.free, asset, -qty)
        self._add(self.locked, asset, qty)

    def _unlock(self, asset: str, qty: float) -> None:
        self._add(self.locked, asset, -qty)
        self._add(self.free, asset, qty)

    async def balances(self) -> dict[str, tuple[float, float]]:
        assets = set(self.free) | set(self.locked)
        return {a: (self.free.get(a, 0.0), self.free.get(a, 0.0) + self.locked.get(a, 0.0)) for a in assets}

    def equity(self) -> float:
        """Nilai akun dalam quote: USDT + semua coin dihargai di harga bid."""
        total = self.free.get(self.quote, 0.0) + self.locked.get(self.quote, 0.0)
        for symbol, (bid, _) in self.prices.items():
            base = symbol.split("/")[0]
            total += (self.free.get(base, 0.0) + self.locked.get(base, 0.0)) * bid
        return total

    # ------------------------------------------------------------------
    # Order
    # ------------------------------------------------------------------
    def _new(self, **kwargs) -> _SimOrder:  # noqa: ANN003
        if kwargs["client_id"] in self.orders:
            raise OrderError(f"client order id {kwargs['client_id']} sudah dipakai")
        order = _SimOrder(timestamp=self._clock(), **kwargs)
        self.orders[order.client_id] = order
        return order

    def _execute_sell(self, order: _SimOrder, qty: float, price: float, from_locked: bool, reason: str) -> None:
        rules = self._rules[order.symbol]
        book = self.locked if from_locked else self.free
        self._add(book, rules.base, -qty)
        proceeds = qty * price
        fee = proceeds * self.fee_rate
        self._add(self.free, rules.quote, proceeds - fee)
        order.filled += qty
        order.average = price
        order.fee_quote += fee
        order.status = "closed"
        self.fills.append(Fill(order.client_id, order.symbol, "sell", qty, price, fee, 0.0, reason, self._clock()))

    async def buy_limit_ioc(self, symbol: str, qty: float, limit_price: float, client_id: str) -> OrderResult:
        rules = await self.rules(symbol)
        _, ask = await self.best_prices(symbol)
        qty = rules.qty_down(qty)
        order = self._new(client_id=client_id, symbol=symbol, side="buy", type="limit", status="open", amount=qty, price=limit_price)
        if rules.check(qty, limit_price) or qty * limit_price > self.free.get(rules.quote, 0.0) + EPSILON:
            order.status = "rejected"  # filter Binance atau saldo USDT kurang
            return order.result()
        fill_price = ask * (1 + self.slippage)
        if fill_price > limit_price:
            order.status = "expired"  # IOC: tidak ada likuiditas di bawah batas harga
            return order.result()
        filled = rules.qty_down(qty * self.fill_ratio.get(symbol, 1.0))
        if filled > 0:
            cost = filled * fill_price
            fee = filled * self.fee_rate
            self._add(self.free, rules.quote, -cost)
            self._add(self.free, rules.base, filled - fee)
            order.filled, order.average, order.fee_base = filled, fill_price, fee
            self.fills.append(Fill(client_id, symbol, "buy", filled, fill_price, 0.0, fee, "entry", self._clock()))
        order.status = "closed" if filled >= qty - EPSILON else "expired"
        return order.result()

    async def sell_market(self, symbol: str, qty: float, client_id: str) -> OrderResult:
        rules = await self.rules(symbol)
        bid, _ = await self.best_prices(symbol)
        qty = rules.qty_down(qty)
        order = self._new(client_id=client_id, symbol=symbol, side="sell", type="market", status="open", amount=qty)
        if rules.check(qty, bid) or self.free.get(rules.base, 0.0) + EPSILON < qty:
            order.status = "rejected"
            return order.result()
        self._execute_sell(order, qty, bid * (1 - self.slippage), from_locked=False, reason="market")
        return order.result()

    async def place_stop_loss(self, symbol: str, qty: float, stop_price: float, client_id: str) -> OrderResult:
        rules = await self.rules(symbol)
        bid, _ = await self.best_prices(symbol)
        qty = rules.qty_down(qty)
        stop = rules.price_down(stop_price)
        if stop >= bid:
            raise OrderError(f"stop {stop:g} di atas harga sekarang {bid:g}, order akan langsung terpicu")
        if reason := rules.check(qty, stop):
            raise OrderError(f"stop loss {symbol} ditolak: {reason}")
        self._lock(rules.base, qty)
        order = self._new(client_id=client_id, symbol=symbol, side="sell", type="stop_loss", status="open", amount=qty, stop_price=stop)
        return order.result()

    async def place_oco_exit(
        self, symbol: str, qty: float, take_profit: float, stop_price: float, list_client_id: str, tp_client_id: str, stop_client_id: str
    ) -> OcoResult:
        rules = await self.rules(symbol)
        bid, ask = await self.best_prices(symbol)
        qty = rules.qty_down(qty)
        tp, stop = rules.price_up(take_profit), rules.price_down(stop_price)
        if tp <= ask:
            raise OrderError(f"take profit {tp:g} tidak di atas harga ask {ask:g} (LIMIT_MAKER akan langsung tereksekusi)")
        if stop >= bid:
            raise OrderError(f"stop {stop:g} di atas harga sekarang {bid:g}")
        if reason := rules.check(qty, stop):
            raise OrderError(f"OCO {symbol} ditolak: {reason}")
        self._lock(rules.base, qty)
        tp_order = self._new(client_id=tp_client_id, symbol=symbol, side="sell", type="limit_maker", status="open",
                             amount=qty, price=tp, list_client_id=list_client_id)
        stop_order = self._new(client_id=stop_client_id, symbol=symbol, side="sell", type="stop_loss", status="open",
                               amount=qty, stop_price=stop, list_client_id=list_client_id)
        self.oco_lists[list_client_id] = _OcoList(list_client_id, symbol, qty, (tp_client_id, stop_client_id))
        return OcoResult(list_client_id, tp_order.result(), stop_order.result(), list_id=list_client_id)

    async def fetch_order(self, symbol: str, client_id: str) -> OrderResult:
        order = self.orders.get(client_id)
        if order is None or order.symbol != symbol:
            return OrderResult.not_found(symbol, client_id)
        return order.result()

    def _cancel_list(self, oco: _OcoList, unlock: bool) -> None:
        if not oco.active:
            return
        oco.active = False
        for leg in oco.legs:
            order = self.orders[leg]
            if order.status == "open":
                order.status = "canceled"
        if unlock:
            self._unlock(self._rules[oco.symbol].base, oco.qty)

    async def cancel_order(self, symbol: str, client_id: str) -> OrderResult:
        order = self.orders.get(client_id)
        if order is None or order.symbol != symbol:
            return OrderResult.not_found(symbol, client_id)
        if order.status == "open":
            if order.list_client_id:
                self._cancel_list(self.oco_lists[order.list_client_id], unlock=True)
            else:
                order.status = "canceled"
                if order.side == "sell":
                    self._unlock(self._rules[symbol].base, order.amount - order.filled)
        return order.result()

    async def cancel_oco(self, symbol: str, list_client_id: str) -> None:
        oco = self.oco_lists.get(list_client_id)
        if oco is not None and oco.symbol == symbol:
            self._cancel_list(oco, unlock=True)

    async def open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        return [o.result() for o in self.orders.values() if o.status == "open" and (symbol is None or o.symbol == symbol)]

    # ------------------------------------------------------------------
    # Simulasi pergerakan harga
    # ------------------------------------------------------------------
    def process_candle(self, symbol: str, open_: float, high: float, low: float, close: float) -> list[OrderResult]:
        """Eksekusi order exit yang tersentuh candle ini (stop lebih dulu), lalu set harga = close."""
        executed: list[OrderResult] = []
        active = [o for o in self.orders.values() if o.symbol == symbol and o.status == "open" and o.side == "sell"]
        for order in sorted(active, key=lambda o: o.type != "stop_loss"):  # stop diproses lebih dulu (pesimis)
            if order.status != "open":
                continue  # sudah batal karena leg OCO lain tereksekusi
            if order.type == "stop_loss" and order.stop_price is not None and low <= order.stop_price:
                price = min(open_, order.stop_price) * (1 - self.slippage)
                self._execute_sell(order, order.amount - order.filled, price, from_locked=True, reason="stop_loss")
            elif order.type == "limit_maker" and order.price is not None and high > order.price:
                self._execute_sell(order, order.amount - order.filled, order.price, from_locked=True, reason="take_profit")
            else:
                continue
            if order.list_client_id:
                self._cancel_list(self.oco_lists[order.list_client_id], unlock=False)  # qty OCO sudah terjual
            executed.append(order.result())
        self.set_price(symbol, close)
        return executed

    def process_price(self, symbol: str, price: float) -> list[OrderResult]:
        """Versi satu harga (paper trading live): sama dengan candle datar."""
        return self.process_candle(symbol, price, price, price, price)

    def snapshot(self) -> dict[str, object]:
        return {"free": dict(self.free), "locked": dict(self.locked), "equity": self.equity()}
