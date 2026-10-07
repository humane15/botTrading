"""Pengelolaan siklus posisi: entry, partial fill, proteksi, TP1, breakeven, trailing, exit.

Kode yang sama dipakai di mode live (LiveExecutor) maupun paper/backtest
(PaperExecutor), karena keduanya punya antarmuka yang sama.

Alur sebuah posisi:
1. Posisi dicatat ke database SEBELUM order dikirim (write-ahead), lengkap
   dengan client order id, sehingga bisa dipulihkan jika bot mati di tengah jalan.
2. Entry memakai LIMIT IOC. Jika hanya terisi sebagian, ukuran posisi = jumlah
   yang benar benar terisi (dikurangi fee yang dipotong dalam coin), dan semua
   order stop loss / take profit memakai ukuran itu.
3. Proteksi: posisi dibagi dua.
   * Bagian A (TP1_FRACTION, default 50%): OCO = take profit LIMIT_MAKER di
     target 1 + stop loss market.
   * Bagian B (runner): stop loss market.
   Jika posisi terlalu kecil untuk dibagi (min notional), seluruhnya masuk OCO.
4. Saat target 1 terisi (sebagian atau penuh): sisa order TP dibatalkan dan
   seluruh sisa posisi dilindungi satu stop di harga breakeven (menutup fee).
5. Setelah TP1, stop naik mengikuti harga tertinggi - 2 x ATR (trailing).
   Stop tidak pernah diturunkan.
6. sync() mencocokkan status order di bursa ke database. Fungsi yang sama
   dipakai saat recovery setelah restart.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import ccxt

from analysis.confluence import SignalResult
from config.settings import Settings
from core.orders import (
    Executor,
    OrderError,
    OrderResult,
    SymbolRules,
    is_bot_order,
    make_client_id,
)
from risk.position_sizing import calculate_position_size
from risk.positions import (
    CLOSED,
    DUST,
    FAILED,
    OPEN,
    PENDING,
    TP1_HIT,
    Position,
    PositionStore,
    utc_now_iso,
)
from risk.risk_manager import RiskManager

log = logging.getLogger(__name__)

ORDER_ERRORS = (OrderError, ccxt.BaseError)


@dataclass(frozen=True)
class ManagerConfig:
    fee_rate: float = 0.001
    slippage: float = 0.0005
    tp1_fraction: float = 0.5
    trailing_atr_mult: float = 2.0
    trailing_min_step_atr: float = 0.25   # stop baru dipasang jika naik >= 0.25 ATR (hemat request)
    entry_max_slippage: float = 0.003
    min_notional_buffer: float = 1.1
    min_reward_risk: float = 1.5

    @classmethod
    def from_settings(cls, settings: Settings) -> ManagerConfig:
        return cls(
            fee_rate=settings.fee_rate,
            slippage=settings.slippage_rate,
            tp1_fraction=settings.tp1_fraction,
            trailing_atr_mult=settings.trailing_atr_mult,
            entry_max_slippage=settings.entry_max_slippage,
            min_notional_buffer=settings.min_notional_buffer,
            min_reward_risk=settings.min_reward_risk,
        )


@dataclass(frozen=True)
class EntryAttempt:
    position: Position | None
    reason: str = ""

    @property
    def opened(self) -> bool:
        return self.position is not None and self.position.is_active


class PositionManager:
    def __init__(self, store: PositionStore, executor: Executor, config: ManagerConfig | None = None) -> None:
        self.store = store
        self.executor = executor
        self.config = config or ManagerConfig()
        self.events: list[str] = []
        self._recorded: dict[str, tuple[str, float]] = {}  # status order terakhir yang sudah ditulis ke database

    @property
    def mode(self) -> str:
        return self.executor.mode

    def _event(self, text: str, level: int = logging.INFO) -> None:
        log.log(level, text)
        self.events.append(text)

    # ------------------------------------------------------------------
    # Saldo dan ekuitas
    # ------------------------------------------------------------------
    async def equity(self, quote: str) -> float:
        """USDT (bebas + terkunci) + nilai posisi yang dikelola bot di harga bid."""
        balances = await self.executor.balances()
        total = balances.get(quote, (0.0, 0.0))[1]
        for position in self.store.active(self.mode):
            if position.qty > 0:
                bid, _ = await self.executor.best_prices(position.symbol)
                total += position.qty * bid
        return total

    async def free_quote(self, quote: str) -> float:
        return (await self.executor.balances()).get(quote, (0.0, 0.0))[0]

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------
    async def try_open(
        self, signal: SignalResult, risk: RiskManager, settings: Settings, *, btc_corr: float | None = None
    ) -> EntryAttempt:
        """Cek risk manager, hitung ukuran posisi, lalu buka posisi jika semua aman."""
        if not signal.is_entry or signal.plan is None:
            return EntryAttempt(None, "sinyal bukan entry")
        symbol = signal.symbol
        rules = await self.executor.rules(symbol)
        equity = await self.equity(settings.quote_asset)
        decision = risk.check_entry(
            symbol, equity=equity, open_positions=self.store.active(self.mode), btc_corr=btc_corr, min_notional=rules.min_notional
        )
        if not decision.allowed:
            return EntryAttempt(None, decision.text())

        _, ask = await self.executor.best_prices(symbol)
        plan = signal.plan
        if plan.resistance is not None:
            # Harga bisa sudah bergerak sejak candle tutup: cek ulang ruang ke resistance.
            room = plan.resistance.low - ask
            if ask <= plan.stop or room / (ask - plan.stop) < self.config.min_reward_risk:
                return EntryAttempt(None, "harga sudah bergerak, ruang ke resistance tidak cukup lagi")
        sizing = calculate_position_size(
            equity=equity,
            free_quote=await self.free_quote(settings.quote_asset),
            entry=ask,
            stop=plan.stop,
            rules=rules,
            risk_per_trade=settings.risk_per_trade,
            max_positions=settings.max_open_positions,
            size_multiplier=signal.size_multiplier,
            fee_rate=self.config.fee_rate,
            slippage=self.config.slippage,
            entry_buffer=self.config.entry_max_slippage,
            min_notional_buffer=self.config.min_notional_buffer,
            split_exit=self.config.tp1_fraction < 1,
        )
        if not sizing.ok:
            return EntryAttempt(None, sizing.reason)
        position = await self.open_position(signal, sizing.qty, btc_corr=btc_corr)
        return EntryAttempt(position, "" if position.is_active else position.exit_reason)

    async def open_position(self, signal: SignalResult, qty: float, *, btc_corr: float | None = None) -> Position:
        if signal.plan is None:
            raise ValueError("sinyal tanpa rencana trade")
        plan = signal.plan
        rules = await self.executor.rules(signal.symbol)
        position = Position(
            symbol=signal.symbol,
            mode=self.mode,
            planned_qty=rules.qty_down(qty),
            initial_stop=plan.stop,
            stop_price=plan.stop,
            tp1_price=plan.tp1,
            atr=plan.atr,
            size_multiplier=signal.size_multiplier,
            btc_corr=btc_corr,
            score=signal.score,
            reasons=signal.summary,
            pattern=signal.pattern,
            regime=signal.regime.trend,
            conditions=",".join(signal.conditions),
            features={k: v for k, v in signal.features.items() if not math.isnan(v)},
        )
        self.store.insert(position)
        # Write-ahead: client id dicatat dulu, baru order dikirim.
        position.entry_client_id = make_client_id("en", position.id)
        self.store.save(position)

        _, ask = await self.executor.best_prices(position.symbol)
        limit = rules.price_up(ask * (1 + self.config.entry_max_slippage))
        if ask <= position.stop_price:
            return self._fail(position, "harga sudah di bawah stop loss")
        try:
            order = await self.executor.buy_limit_ioc(position.symbol, position.planned_qty, limit, position.entry_client_id)
        except ORDER_ERRORS as exc:
            return self._fail(position, f"order entry gagal: {exc}")
        self.store.record_order(position.id, "entry", order)
        return await self._finalize_entry(position, order, rules)

    def _fail(self, position: Position, reason: str) -> Position:
        position.status = FAILED
        position.exit_reason = reason
        position.closed_at = utc_now_iso()
        self.store.save(position)
        self._event(f"[ENTRY GAGAL] {position.symbol}: {reason}", logging.WARNING)
        return position

    async def _finalize_entry(self, position: Position, order: OrderResult, rules: SymbolRules) -> Position:
        if order.filled <= 0:
            return self._fail(position, "order entry tidak terisi")
        received = max(order.filled - order.fee_base, 0.0)
        net_qty = rules.qty_down(received)
        position.entry_price = order.average or order.price or 0.0
        position.initial_qty = position.qty = net_qty
        # Sisa di bawah step size tidak bisa dijual (dust). Dicatat terpisah agar PnL dan R
        # trade hanya menghitung coin yang benar benar diperdagangkan.
        position.dust_qty = max(received - net_qty, 0.0)
        position.cost_quote = order.filled * position.entry_price + order.fee_quote
        # Fee beli yang dipotong dalam coin sudah tercermin di net_qty; di sini dicatat
        # nilai setaranya dalam USDT hanya untuk laporan.
        position.fees_quote = order.fee_quote + order.fee_base * position.entry_price
        position.highest_price = position.lowest_price = position.entry_price
        position.opened_at = utc_now_iso()
        stop_exit = position.stop_price * (1 - self.config.slippage) * (1 - self.config.fee_rate)
        position.risk_amount = max(net_qty * (position.cost_per_unit - stop_exit), 0.0)
        if order.filled + rules.step_size / 2 < position.planned_qty:
            self._event(
                f"[PARTIAL FILL] {position.symbol}: terisi {order.filled:g} dari {position.planned_qty:g}, "
                f"ukuran posisi dan order SL/TP disesuaikan menjadi {net_qty:g}"
            )
        if not rules.is_sellable(net_qty, position.stop_price):
            position.status = DUST
            position.exit_reason = "terisi terlalu sedikit untuk dijual (di bawah min notional), cek manual"
            self.store.save(position)
            self._event(f"[DUST] {position.symbol}: {position.exit_reason}", logging.WARNING)
            return position
        position.status = OPEN
        self._event(
            f"[BUY] {position.symbol} {net_qty:g} @ {position.entry_price:.8g} | SL {position.stop_price:.8g} | "
            f"TP1 {position.tp1_price:.8g} | risiko ${position.risk_amount:.2f}"
        )
        await self._protect(position, rules)
        self.store.save(position)
        return position

    async def _protect(self, position: Position, rules: SymbolRules) -> None:
        """Pasang OCO (TP1 + stop) untuk bagian A dan stop untuk runner."""
        tp_qty = rules.qty_down(position.qty * self.config.tp1_fraction)
        runner = rules.qty_down(position.qty - tp_qty)
        buffer = self.config.min_notional_buffer
        if runner <= 0 or not (rules.is_sellable(tp_qty, position.stop_price, buffer) and rules.is_sellable(runner, position.stop_price, buffer)):
            tp_qty, runner = position.qty, 0.0  # terlalu kecil untuk dibagi: satu exit saja
        position.tp1_qty = tp_qty

        placed_oco = False
        if rules.oco_allowed:
            list_id = make_client_id("oc", position.id)
            tp_id, stop_id = make_client_id("tp", position.id), make_client_id("sa", position.id)
            try:
                oco = await self.executor.place_oco_exit(position.symbol, tp_qty, position.tp1_price, position.stop_price, list_id, tp_id, stop_id)
            except ORDER_ERRORS as exc:
                self._event(f"[OCO GAGAL] {position.symbol}: {exc}; TP dipantau bot, stop tetap di bursa", logging.WARNING)
            else:
                position.oco_client_id, position.tp_client_id, position.stop_a_client_id = list_id, tp_id, stop_id
                position.tp_filled = position.stop_a_filled = 0.0
                position.tp_mode = "oco"
                self.store.record_order(position.id, "tp", oco.take_profit)
                self.store.record_order(position.id, "stop_a", oco.stop)
                placed_oco = True
        if not placed_oco:
            position.tp_mode = "software"
            await self._place_stop(position, rules, "stop_a", tp_qty, position.stop_price)
        if runner > 0:
            await self._place_stop(position, rules, "stop_b", runner, position.stop_price)

    # ------------------------------------------------------------------
    # Order stop dan exit
    # ------------------------------------------------------------------
    def _stop_reason(self, position: Position) -> str:
        if not position.breakeven:
            return "stop_loss"
        if position.stop_price > position.breakeven_price(self.config.fee_rate) * (1 + 1e-9):
            return "trailing_stop"
        return "breakeven"

    async def _place_stop(self, position: Position, rules: SymbolRules, role: str, qty: float, stop_price: float) -> bool:
        qty = rules.qty_down(min(qty, position.qty))
        if qty <= 0:
            return False
        stop = rules.price_down(stop_price)
        bid, _ = await self.executor.best_prices(position.symbol)
        if stop >= bid:
            # Harga sudah di bawah stop: order stop akan ditolak bursa, jadi keluar sekarang.
            await self._market_sell(position, rules, qty, self._stop_reason(position))
            return False
        client_id = make_client_id("sa" if role == "stop_a" else "sb", position.id)
        try:
            order = await self.executor.place_stop_loss(position.symbol, qty, stop, client_id)
        except ORDER_ERRORS as exc:
            self._event(f"[STOP GAGAL] {position.symbol}: {exc} (dicoba lagi saat sync berikutnya)", logging.ERROR)
            return False
        setattr(position, f"{role}_client_id", client_id)
        setattr(position, f"{role}_filled", 0.0)
        self.store.record_order(position.id, role, order)
        return True

    async def _market_sell(self, position: Position, rules: SymbolRules, qty: float, reason: str) -> None:
        qty = rules.qty_down(min(qty, position.qty))
        if qty <= 0:
            return
        client_id = make_client_id("ex", position.id)
        try:
            order = await self.executor.sell_market(position.symbol, qty, client_id)
        except ORDER_ERRORS as exc:
            self._event(f"[EXIT GAGAL] {position.symbol}: {exc}", logging.ERROR)
            return
        self.store.record_order(position.id, "exit", order)
        if order.filled > 0:
            price = order.average or 0.0
            pnl = position.record_exit(order.filled, price, order.fee_quote, reason)
            self._event(f"[SELL] {position.symbol} {order.filled:g} @ {price:.8g} ({reason}) | PnL ${pnl:+.2f}")

    def _apply_fill(self, position: Position, role: str, order: OrderResult, reason: str) -> None:
        """Catat bagian order yang baru terisi sejak sync terakhir."""
        state = (order.status, order.filled)
        if order.status != "not_found" and self._recorded.get(order.client_id) != state:
            self.store.record_order(position.id, role, order)  # hanya jika status atau jumlah terisi berubah
            self._recorded[order.client_id] = state
        recorded = getattr(position, f"{role}_filled")
        new_qty = order.filled - recorded
        if new_qty <= 1e-12:
            return
        price = order.average or order.price or order.stop_price or 0.0
        fee = order.fee_quote * (new_qty / order.filled) if order.filled > 0 else 0.0
        pnl = position.record_exit(new_qty, price, fee, reason)
        setattr(position, f"{role}_filled", order.filled)
        label = "TP1" if role == "tp" else "SELL"
        self._event(f"[{label}] {position.symbol} {new_qty:g} @ {price:.8g} ({reason}) | PnL ${pnl:+.2f}")

    async def _cancel_role(self, position: Position, role: str) -> None:
        client_id = getattr(position, f"{role}_client_id")
        if not client_id:
            return
        order = await self.executor.cancel_order(position.symbol, client_id)
        self._apply_fill(position, role, order, "take_profit_1" if role == "tp" else self._stop_reason(position))
        setattr(position, f"{role}_client_id", "")

    async def _cancel_all(self, position: Position) -> None:
        if position.oco_client_id:
            await self.executor.cancel_oco(position.symbol, position.oco_client_id)
            for role in ("tp", "stop_a"):
                client_id = getattr(position, f"{role}_client_id")
                if client_id:
                    order = await self.executor.fetch_order(position.symbol, client_id)
                    self._apply_fill(position, role, order, "take_profit_1" if role == "tp" else self._stop_reason(position))
                    setattr(position, f"{role}_client_id", "")
            position.oco_client_id = ""
        for role in ("stop_a", "stop_b"):
            await self._cancel_role(position, role)

    def _is_flat(self, position: Position, rules: SymbolRules) -> bool:
        return rules.qty_down(position.qty) <= 0

    async def _finish(self, position: Position, reason: str | None = None) -> None:
        await self._cancel_all(position)
        position.status = CLOSED
        position.closed_at = utc_now_iso()
        if reason:
            position.exit_reason = reason
        self._event(
            f"[CLOSED] {position.symbol} ({position.exit_reason}) | PnL ${position.realized_pnl:+.2f} | "
            f"{position.r_multiple:+.2f}R"
        )

    async def _move_stop(self, position: Position, rules: SymbolRules, new_stop: float) -> None:
        """Ganti stop seluruh sisa posisi ke harga baru (breakeven atau trailing)."""
        if not position.oco_client_id:
            await self._cancel_role(position, "stop_a")
        await self._cancel_role(position, "stop_b")
        if self._is_flat(position, rules):
            return
        position.stop_price = new_stop
        await self._place_stop(position, rules, "stop_b", position.qty, new_stop)

    async def _on_tp1(self, position: Position, rules: SymbolRules) -> None:
        """Target 1 terisi: batalkan sisa OCO, lalu lindungi sisa posisi di breakeven."""
        position.status = TP1_HIT
        position.breakeven = True
        if position.oco_client_id:
            tp = await self.executor.fetch_order(position.symbol, position.tp_client_id)
            if not tp.is_done:
                # TP baru terisi sebagian: sisanya dibatalkan dan digabung ke runner.
                await self.executor.cancel_oco(position.symbol, position.oco_client_id)
            await self._cancel_all_oco_legs(position)
        breakeven = position.breakeven_price(self.config.fee_rate)
        self._event(f"[TP1] {position.symbol}: stop sisa posisi dipindah ke breakeven {breakeven:.8g}")
        await self._move_stop(position, rules, max(position.stop_price, rules.price_up(breakeven)))

    async def _cancel_all_oco_legs(self, position: Position) -> None:
        for role in ("tp", "stop_a"):
            client_id = getattr(position, f"{role}_client_id")
            if client_id:
                order = await self.executor.fetch_order(position.symbol, client_id)
                self._apply_fill(position, role, order, "take_profit_1" if role == "tp" else "stop_loss")
                setattr(position, f"{role}_client_id", "")
        position.oco_client_id = ""

    # ------------------------------------------------------------------
    # Sinkronisasi dengan bursa
    # ------------------------------------------------------------------
    async def sync(self, position: Position) -> Position:
        """Cocokkan status order di bursa ke posisi: catat fill, TP1, exit, dan pasang ulang proteksi."""
        if position.status == PENDING:
            return await self._resolve_pending(position)
        if not position.is_active:
            return position
        rules = await self.executor.rules(position.symbol)

        tp_canceled = False
        if position.tp_client_id:
            tp = await self.executor.fetch_order(position.symbol, position.tp_client_id)
            self._apply_fill(position, "tp", tp, "take_profit_1")
            if tp.filled > 0 and position.status == OPEN:
                await self._on_tp1(position, rules)
            elif tp.is_done:
                position.tp_client_id = ""  # batal tanpa terisi: stop leg OCO tereksekusi, atau dibatalkan manual
                tp_canceled = position.status == OPEN

        open_stops: dict[str, OrderResult] = {}
        for role in ("stop_a", "stop_b"):
            client_id = getattr(position, f"{role}_client_id")
            if not client_id:
                continue
            order = await self.executor.fetch_order(position.symbol, client_id)
            self._apply_fill(position, role, order, self._stop_reason(position))
            if order.is_done:
                setattr(position, f"{role}_client_id", "")
            else:
                open_stops[role] = order
        if position.oco_client_id and not position.tp_client_id and not position.stop_a_client_id:
            position.oco_client_id = ""

        if self._is_flat(position, rules):
            await self._finish(position)
        else:
            if tp_canceled and position.stop_a_filled <= 0:
                # OCO batal tanpa ada leg yang terisi (misal dibatalkan manual): TP1 dipantau bot.
                position.tp_mode = "software"
                self._event(f"[TP SOFTWARE] {position.symbol}: order TP1 di bursa tidak aktif lagi, target 1 dipantau bot", logging.WARNING)
            await self._ensure_protected(position, rules, open_stops)
        self.store.save(position)
        return position

    async def _base_balance(self, position: Position) -> tuple[float, float]:
        """Saldo coin posisi: (bebas, total termasuk yang terkunci order)."""
        return (await self.executor.balances()).get(position.base, (0.0, 0.0))

    async def _ensure_protected(self, position: Position, rules: SymbolRules, open_stops: dict[str, OrderResult]) -> None:
        """Pastikan seluruh sisa posisi dilindungi stop di bursa, dan deteksi coin yang hilang dari akun."""
        covered = sum(order.remaining for order in open_stops.values())
        missing = rules.qty_down(position.qty - covered)
        if missing <= 0:
            return
        free_base, total_base = await self._base_balance(position)
        if total_base + 1e-12 < position.qty:
            # Coin tidak ada lagi di akun (bebas maupun terkunci order): dijual/dipindah di luar bot.
            gone = position.qty - total_base
            bid, _ = await self.executor.best_prices(position.symbol)
            # Harga jual sebenarnya tidak diketahui; dipakai harga bid sekarang sebagai perkiraan.
            position.record_exit(gone, bid, 0.0, "ditutup_di_luar_bot")
            self._event(
                f"[DI LUAR BOT] {position.symbol}: {gone:g} coin tidak ada lagi di akun, dicatat terjual di ~{bid:.8g}",
                logging.WARNING,
            )
            if self._is_flat(position, rules):
                await self._finish(position, "ditutup_di_luar_bot")
                return
            missing = rules.qty_down(position.qty - covered)
            if missing <= 0:
                return
        if not rules.is_sellable(missing, position.stop_price):
            if covered <= 0:
                await self._finish(position, "sisa posisi di bawah min notional (dust)")
            return
        if free_base + 1e-12 < missing:
            # Coin masih ada tetapi terkunci order lain, misalnya order bot yang hasil kirimnya
            # tidak pasti saat jaringan putus. Order bot seperti itu dibatalkan agar stop bisa dipasang.
            await self._cancel_stray_orders(position)
        if "stop_b" in open_stops:
            await self._cancel_role(position, "stop_b")  # digabung: satu stop untuk semua bagian yang belum terlindungi
        others = sum(order.remaining for role, order in open_stops.items() if role != "stop_b")
        target = rules.qty_down(position.qty - others)
        free_base, _ = await self._base_balance(position)
        protect = rules.qty_down(min(target, free_base))
        if protect + 1e-12 < target:
            self._event(
                f"[PROTEKSI TERTAHAN] {position.symbol}: {target - protect:g} coin terkunci order di luar bot, "
                "bagian itu belum terlindungi stop (cek order manual di Binance)",
                logging.ERROR,
            )
        if protect > 0 and await self._place_stop(position, rules, "stop_b", protect, position.stop_price):
            self._event(f"[PROTEKSI] {position.symbol}: {missing:g} coin belum terlindungi, stop dipasang ulang", logging.WARNING)
        if self._is_flat(position, rules) and position.is_active:
            await self._finish(position)

    async def _cancel_stray_orders(self, position: Position) -> None:
        """Batalkan order bot di simbol ini yang tidak tercatat di posisi (sisa order yang statusnya tidak pasti)."""
        known = {position.tp_client_id, position.stop_a_client_id, position.stop_b_client_id} - {""}
        for order in await self.executor.open_orders(position.symbol):
            if is_bot_order(order.client_id) and order.client_id not in known:
                canceled = await self.executor.cancel_order(position.symbol, order.client_id)
                self.store.record_order(position.id, "stray", canceled)
                self._event(f"[ORDER TAK TERCATAT] {position.symbol}: order bot {order.client_id} dibatalkan", logging.WARNING)

    async def _resolve_pending(self, position: Position) -> Position:
        """Entry yang statusnya belum tercatat (misal bot mati setelah order dikirim)."""
        if not position.entry_client_id:
            return self._fail(position, "entry belum pernah dikirim")
        order = await self.executor.fetch_order(position.symbol, position.entry_client_id)
        if order.status == "not_found":
            return self._fail(position, "order entry tidak ditemukan di bursa")
        if not order.is_done:
            order = await self.executor.cancel_order(position.symbol, position.entry_client_id)
        self.store.record_order(position.id, "entry", order)
        rules = await self.executor.rules(position.symbol)
        return await self._finalize_entry(position, order, rules)

    # ------------------------------------------------------------------
    # Update per candle (trailing) dan penutupan manual
    # ------------------------------------------------------------------
    async def on_candle(self, position: Position, high: float, low: float, close: float, atr: float | None = None) -> Position:
        """Dipanggil tiap candle 5m tertutup SETELAH sync(): MAE/MFE, TP software, trailing stop."""
        if position.status not in (OPEN, TP1_HIT):
            return position
        rules = await self.executor.rules(position.symbol)
        position.update_extremes(high, low)
        atr_value = atr if atr and atr > 0 else position.atr

        if position.status == OPEN and position.tp_mode == "software" and high > position.tp1_price:
            await self._software_tp1(position, rules)
        elif position.status == TP1_HIT and atr_value > 0:
            candidate = max(
                position.highest_price - self.config.trailing_atr_mult * atr_value,
                position.breakeven_price(self.config.fee_rate),
            )
            candidate = rules.price_down(candidate)
            if candidate > position.stop_price + self.config.trailing_min_step_atr * atr_value:
                self._event(f"[TRAIL] {position.symbol}: stop naik {position.stop_price:.8g} -> {candidate:.8g}")
                await self._move_stop(position, rules, candidate)

        if position.is_active and self._is_flat(position, rules):
            await self._finish(position)
        self.store.save(position)
        return position

    async def _software_tp1(self, position: Position, rules: SymbolRules) -> None:
        """TP1 tanpa OCO: lepas stop agar coin bisa dijual, jual bagian TP1 di market, lalu sisa ke breakeven."""
        await self._cancel_role(position, "stop_a")
        await self._cancel_role(position, "stop_b")
        if self._is_flat(position, rules):
            return
        qty_before = position.qty
        await self._market_sell(position, rules, position.tp1_qty, "take_profit_1")
        if position.qty >= qty_before:
            # Jual gagal: pasang lagi stop di level lama, TP1 dicoba lagi pada candle berikutnya.
            await self._place_stop(position, rules, "stop_b", position.qty, position.stop_price)
            return
        position.status = TP1_HIT
        position.breakeven = True
        breakeven = position.breakeven_price(self.config.fee_rate)
        await self._move_stop(position, rules, max(position.stop_price, rules.price_up(breakeven)))

    async def close_position(self, position: Position, reason: str = "manual") -> Position:
        """Tutup posisi sekarang: batalkan semua order lalu jual sisa di harga market."""
        if position.status == PENDING:
            position = await self._resolve_pending(position)
        if not position.is_active:
            return position
        rules = await self.executor.rules(position.symbol)
        await self._cancel_all(position)
        if not self._is_flat(position, rules):
            await self._market_sell(position, rules, position.qty, reason)
        if self._is_flat(position, rules):
            await self._finish(position, reason)
        else:
            await self._ensure_protected(position, rules, {})
        self.store.save(position)
        return position

    async def abandon(self, position: Position, price: float, reason: str) -> Position:
        """Tutup catatan posisi tanpa menjual (coin sudah tidak ada di akun, misal dijual manual).

        Harga jual sebenarnya tidak diketahui, jadi PnL dihitung dengan `price` sebagai perkiraan.
        """
        await self._cancel_all(position)
        if position.qty > 0:
            position.record_exit(position.qty, price, 0.0, reason)
        position.qty = 0.0
        await self._finish(position, reason)
        self.store.save(position)
        return position

    async def sync_all(self) -> list[Position]:
        return [await self.sync(position) for position in self.store.active(self.mode)]
