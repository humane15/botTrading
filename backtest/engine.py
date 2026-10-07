"""Backtester event driven: kode sinyal, risiko, dan order yang sama persis dengan live.

Alur:
1. Data: candle 5m, 15m, 30m, 1h tiap coin (plus BTC sebagai acuan pasar).
2. Universe point in time: tiap hari dipilih 75 coin dari volume hari sebelumnya.
3. Tahap 1 (backtest/signals.py): sinyal ConfluenceEngine di setiap candle 5m
   tertutup, paralel per coin.
4. Tahap 2 (di sini): simulasi portofolio berurutan waktu, setiap candle 5m:
   a. candle yang baru tertutup diproses bursa simulasi (PaperExecutor) untuk
      coin yang punya posisi: stop loss, TP1, gap, urutan pesimis;
   b. PositionManager.sync() dan on_candle(): TP1, breakeven, trailing stop
      (ATR 15m), persis seperti bot live;
   c. sinyal entry pada candle itu diurutkan dari skor tertinggi lalu dicoba
      lewat PositionManager.try_open(): risk manager (slot, batas rugi harian
      dan mingguan, korelasi BTC), position sizing (fee, slippage, min notional),
      lalu order IOC di harga penutupan candle + slippage;
   d. nilai akun dicatat untuk kurva ekuitas.
5. Posisi yang masih terbuka di akhir periode ditutup di harga terakhir
   (alasan "akhir_backtest") agar semua trade masuk laporan.

Tidak ada data masa depan: keputusan di waktu T hanya memakai candle yang
sudah tertutup pada T, dan order dieksekusi pada candle setelahnya.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from analysis.confluence import EngineParams, SignalResult, prepare_frames, rank_signals
from analysis.indicators import compute_indicators
from analysis.weights import default_weight_values
from backtest.signals import (
    FrameSource,
    SymbolSignals,
    generate_signals,
    market_contexts,
    step_times,
)
from backtest.universe import daily_universe
from config.settings import TIMEFRAME_MS, Settings
from core.data_feed import frame_open_ms
from core.database import connect
from core.orders import SymbolRules
from core.paper_exchange import PaperExecutor
from risk.position_manager import ManagerConfig, PositionManager
from risk.positions import (
    ACTIVE_STATUSES,
    CLOSED,
    Position,
    PositionStore,
    simulated_clock,
)
from risk.risk_manager import RiskManager, btc_correlation

log = logging.getLogger(__name__)

MODE = "backtest"
DAY_MS = 86_400_000
QUIET_LOGGERS = ("risk", "core.paper_exchange", "analysis")


@dataclass(frozen=True)
class BacktestConfig:
    start: pd.Timestamp
    end: pd.Timestamp
    initial_capital: float
    universe_size: int = 75
    min_quote_volume: float = 5_000_000.0
    workers: int = 1
    oos_fraction: float = 1 / 3      # porsi akhir periode yang dilaporkan terpisah (out of sample Fase 5)

    @property
    def oos_start(self) -> pd.Timestamp:
        span = self.end - self.start
        return (self.end - span * self.oos_fraction).floor("D")


@dataclass
class EntryStats:
    signals: int = 0
    opened: int = 0
    skipped: Counter[str] = field(default_factory=Counter)


@dataclass
class BacktestResult:
    config: BacktestConfig
    symbols: list[str]
    weights: dict[str, float]
    settings: dict[str, object]
    universe: dict[pd.Timestamp, tuple[str, ...]]
    equity: pd.Series                 # nilai akun setelah tiap candle 5m (USDT)
    positions_open: pd.Series         # jumlah posisi terbuka tiap candle
    trades: list[Position]            # posisi tertutup, urut waktu buka
    benchmark: pd.Series              # harga BTC tiap candle (pembanding buy and hold)
    entry_stats: EntryStats
    dust_value: float                 # nilai sisa coin di bawah step size di akhir (tetap milik akun)
    evaluations: int
    rejections: Counter[str]
    events: list[str]
    timings: dict[str, float]
    database: sqlite3.Connection      # isi PositionStore hasil backtest (jurnal untuk Fase 5)


class SymbolSeries:
    """Data satu coin untuk simulasi: candle 5m per langkah, ATR 15m, close 1h.

    Hanya baris dan kolom yang dibutuhkan yang disalin (bukan frame utuh), supaya
    puluhan coin x 6 bulan candle 5m tetap hemat memori.
    """

    def __init__(self, raw: Mapping[str, pd.DataFrame], steps_ms: np.ndarray, params: EngineParams) -> None:
        trigger_tf, setup_tf, trend_tf = (params.timeframe_of(r) for r in ("trigger", "setup", "trend"))
        frame_5m = raw[trigger_tf]
        opens = frame_open_ms(frame_5m)
        target = steps_ms - TIMEFRAME_MS[trigger_tf]
        k = np.searchsorted(opens, target, side="right")
        exists = (k > 0) & (opens[np.maximum(k - 1, 0)] == target) if len(opens) else np.zeros(len(steps_ms), bool)
        rows = np.where(exists, k - 1, -1)
        first = int(rows[exists].min()) if exists.any() else 0
        self.row = np.where(exists, rows - first, -1).astype("int32")
        last = int(rows.max()) + 1 if exists.any() else 0
        ohlc = frame_5m[["open", "high", "low", "close"]].to_numpy(dtype="float64")[first:last]
        self.open, self.high, self.low, self.close = (ohlc[:, j].copy() for j in range(4))

        setup = raw[setup_tf]
        atr = compute_indicators(setup, params.indicators)["atr"].to_numpy(dtype="float64") if len(setup) else np.empty(0)
        k_setup = np.searchsorted(frame_open_ms(setup), steps_ms - TIMEFRAME_MS[setup_tf], side="right")
        self.atr = np.where(k_setup > 0, atr[np.maximum(k_setup - 1, 0)] if len(atr) else np.nan, np.nan)

        trend = raw[trend_tf]
        self.closes_1h = trend["close"].copy()
        self.count_1h = np.searchsorted(frame_open_ms(trend), steps_ms - TIMEFRAME_MS[trend_tf], side="right").astype("int32")

    def bar(self, i: int) -> tuple[float, float, float, float] | None:
        r = int(self.row[i])
        if r < 0:
            return None
        return float(self.open[r]), float(self.high[r]), float(self.low[r]), float(self.close[r])

    def last_close(self, i: int) -> float | None:
        r = int(self.row[i])
        return float(self.close[r]) if r >= 0 else None

    def atr_at(self, i: int) -> float | None:
        value = float(self.atr[i])
        return value if value == value and value > 0 else None

    def hourly_closes(self, i: int, count: int = 200) -> pd.Series:
        k = int(self.count_1h[i])
        return self.closes_1h.iloc[max(0, k - count):k]


@contextmanager
def _quiet_logs() -> Iterator[None]:
    """Log per trade tetap dicatat di events, tetapi tidak membanjiri console selama backtest."""
    loggers = [logging.getLogger(name) for name in QUIET_LOGGERS]
    levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for logger, level in zip(loggers, levels, strict=True):
            logger.setLevel(level)


class Backtester:
    def __init__(
        self,
        settings: Settings,
        config: BacktestConfig,
        *,
        source: FrameSource,
        rules: Mapping[str, SymbolRules],
        symbols: Sequence[str],
        weights: Mapping[str, float] | None = None,
        params: EngineParams | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.settings = settings
        self.config = config
        self.source = source
        self.rules = dict(rules)
        self.symbols = [s for s in dict.fromkeys(symbols) if s in self.rules]
        self.weights = default_weight_values() | dict(weights or {})
        self.params = params or EngineParams.from_settings(settings)
        self.progress = progress or (lambda text: None)
        self._now_ms = 0

    # ------------------------------------------------------------------
    def _now(self) -> datetime:
        return datetime.fromtimestamp(self._now_ms / 1000, tz=timezone.utc)

    def _steps(self) -> np.ndarray:
        return step_times(self.config.start, self.config.end, self.params.timeframe_of("trigger"))

    async def run(self) -> BacktestResult:
        timings: dict[str, float] = {}
        clock = time.perf_counter()
        steps = self._steps()
        if len(steps) == 0:
            raise ValueError("periode backtest kosong")
        btc = self.settings.btc_symbol
        trend_tf = self.params.timeframe_of("trend")

        self.progress(f"Memuat data {len(self.symbols)} coin ...")
        series: dict[str, SymbolSeries] = {}
        hourly: dict[str, pd.DataFrame] = {}
        for symbol in self.symbols:
            raw = self.source.load(symbol)
            if any(raw.get(tf) is None or raw[tf].empty for tf in self.params.timeframes):
                continue
            series[symbol] = SymbolSeries(raw, steps, self.params)
            hourly[symbol] = raw[trend_tf]
        btc_raw = self.source.load(btc)
        if any(btc_raw.get(tf) is None or btc_raw[tf].empty for tf in self.params.timeframes):
            raise ValueError(f"data {btc} tidak lengkap: regime BTC dan circuit breaker tidak bisa dihitung")
        btc_series = series.get(btc) or SymbolSeries(btc_raw, steps, self.params)
        timings["muat_data"] = time.perf_counter() - clock

        clock = time.perf_counter()
        days = pd.date_range(self.config.start.floor("D"), self.config.end, freq="D", tz="UTC")
        universe = daily_universe(hourly, days, self.config.universe_size, self.config.min_quote_volume)
        del hourly
        step_days = pd.DatetimeIndex(pd.to_datetime(steps, unit="ms", utc=True)).floor("D")
        active = {symbol: np.zeros(len(steps), dtype=bool) for symbol in series}
        for day, members in universe.items():
            mask = step_days == day
            for symbol in members:
                if symbol in active:
                    active[symbol] |= mask

        self.progress("Menghitung regime BTC dan circuit breaker ...")
        btc_frames = prepare_frames({tf: btc_raw[tf] for tf in self.params.timeframes}, self.params.indicators)
        with _quiet_logs():
            contexts = market_contexts(btc_frames, steps, self.params, self.settings)
        del btc_frames
        timings["konteks_pasar"] = time.perf_counter() - clock

        clock = time.perf_counter()
        self.progress(f"Menghitung sinyal ({self.config.workers} proses) ...")
        with _quiet_logs():
            per_symbol = generate_signals(
                list(series), self.source, steps, active, contexts, self.weights, self.params,
                window=self.settings.ohlcv_limit, workers=self.config.workers, progress=self.progress,
            )
        del contexts
        timings["sinyal"] = time.perf_counter() - clock

        clock = time.perf_counter()
        self.progress("Simulasi portofolio ...")
        with _quiet_logs():
            result = await self._simulate(steps, series, btc_series, per_symbol)
        timings["simulasi"] = time.perf_counter() - clock

        result.universe = universe
        result.timings = timings
        return result

    # ------------------------------------------------------------------
    async def _simulate(
        self, steps: np.ndarray, series: Mapping[str, SymbolSeries], btc: SymbolSeries, per_symbol: Mapping[str, SymbolSignals]
    ) -> BacktestResult:
        """Semua waktu posisi dan order dicatat dengan jam simulasi (waktu candle), bukan jam komputer."""
        with simulated_clock(self._now):
            return await self._simulate_steps(steps, series, btc, per_symbol)

    async def _simulate_steps(
        self, steps: np.ndarray, series: Mapping[str, SymbolSeries], btc: SymbolSeries, per_symbol: Mapping[str, SymbolSignals]
    ) -> BacktestResult:
        settings = self.settings
        executor = PaperExecutor(
            self.rules, quote=settings.quote_asset, starting_balance=self.config.initial_capital,
            fee_rate=settings.fee_rate, slippage=settings.slippage_rate, clock=lambda: self._now_ms, mode=MODE,
        )
        store = PositionStore(connect(":memory:"))
        manager = PositionManager(store, executor, ManagerConfig.from_settings(settings))
        risk = RiskManager(settings, store, MODE, clock=self._now, kill_switch=False)

        by_step: dict[int, list[SignalResult]] = {}
        for item in per_symbol.values():
            for signal in item.entries:
                index = int(np.searchsorted(steps, int(signal.timestamp.value // 1_000_000)))
                by_step.setdefault(index, []).append(signal)
        stats = EntryStats(signals=sum(len(item.entries) for item in per_symbol.values()))

        equity = np.empty(len(steps))
        open_count = np.zeros(len(steps), dtype="int64")
        open_positions: dict[int, Position] = {}
        current_day: int | None = None
        report_every = max(len(steps) // 10, 1)

        for i, step in enumerate(steps):
            self._now_ms = int(step)
            # a + b. Candle yang baru tertutup: eksekusi order di bursa simulasi, lalu sync dan trailing.
            for pid, position in list(open_positions.items()):
                data = series[position.symbol]
                bar = data.bar(i)
                if bar is None:
                    continue  # data coin ini bolong pada candle ini
                executor.process_candle(position.symbol, *bar)
                position = await manager.sync(position)
                if position.is_active:
                    position = await manager.on_candle(position, bar[1], bar[2], bar[3], data.atr_at(i))
                if position.is_active:
                    open_positions[pid] = position
                else:
                    del open_positions[pid]

            # Awal hari UTC: catat nilai akun awal hari/minggu untuk batas rugi.
            day = self._now_ms // DAY_MS
            if day != current_day:
                current_day = day
                risk.loss_status(executor.equity())

            # c. Entry: sinyal candle ini, skor tertinggi dulu.
            for signal in rank_signals(by_step.get(i, [])):
                price = series[signal.symbol].last_close(i)
                if price is None:
                    stats.skipped["tanpa_harga"] += 1
                    continue
                executor.set_price(signal.symbol, price)
                corr = btc_correlation(series[signal.symbol].hourly_closes(i), btc.hourly_closes(i))
                attempt = await manager.try_open(signal, risk, settings, btc_corr=corr)
                if attempt.opened and attempt.position is not None:
                    stats.opened += 1
                    open_positions[int(attempt.position.id)] = attempt.position
                else:
                    stats.skipped[attempt.reason or "tidak terbuka"] += 1

            # d. Kurva ekuitas.
            equity[i] = executor.equity()
            open_count[i] = len(open_positions)
            if i % report_every == 0 and i:
                self.progress(f"  simulasi {i / len(steps):.0%} | ekuitas ${equity[i]:,.2f} | posisi {len(open_positions)}")

        for position in list(open_positions.values()):
            await manager.close_position(position, "akhir_backtest")
        final_equity = executor.equity()
        if len(steps):
            equity[-1] = final_equity
        quote = settings.quote_asset
        dust_value = final_equity - executor.free.get(quote, 0.0) - executor.locked.get(quote, 0.0)

        index = pd.DatetimeIndex(pd.to_datetime(steps, unit="ms", utc=True), name="timestamp")
        trades = sorted(store.closed(MODE), key=lambda p: (p.opened_at or "", p.id or 0))
        leftovers = [p for p in store.with_status(ACTIVE_STATUSES, MODE) if p.status != CLOSED]
        if leftovers:
            log.warning("%d posisi tidak bisa ditutup di akhir backtest", len(leftovers))
        benchmark = pd.Series([btc.last_close(i) for i in range(len(steps))], index=index, dtype="float64").ffill()
        rejections: Counter[str] = Counter()
        for item in per_symbol.values():
            rejections.update(item.rejections)
        return BacktestResult(
            config=self.config,
            symbols=list(series),
            weights=dict(self.weights),
            settings=_settings_snapshot(settings),
            universe={},
            equity=pd.Series(equity, index=index, name="equity"),
            positions_open=pd.Series(open_count, index=index, name="positions"),
            trades=trades,
            benchmark=benchmark,
            entry_stats=stats,
            dust_value=dust_value,
            evaluations=sum(item.evaluations for item in per_symbol.values()),
            rejections=rejections,
            events=list(manager.events),
            timings={},
            database=store.conn,
        )


def _settings_snapshot(settings: Settings) -> dict[str, object]:
    """Parameter yang memengaruhi hasil (tanpa API key)."""
    keys = (
        "risk_per_trade", "max_open_positions", "daily_loss_limit", "weekly_loss_limit", "fee_rate", "slippage_rate",
        "tp1_fraction", "trailing_atr_mult", "entry_max_slippage", "min_notional_buffer", "btc_correlation_threshold",
        "max_btc_correlated_positions", "min_signal_score", "min_aligned_timeframes", "min_reward_risk",
        "circuit_breaker_drop", "circuit_breaker_hours", "timeframes", "ohlcv_limit", "universe_size",
        "min_quote_volume_usd", "quote_asset",
    )
    data = settings.model_dump(include=set(keys))
    return {key: (list(value) if isinstance(value, tuple) else value) for key, value in data.items()}


def position_rows(trades: Sequence[Position]) -> list[dict[str, object]]:
    """Baris trade untuk CSV/laporan (tanpa kolom client id)."""
    rows = []
    for p in trades:
        row = asdict(p)
        for key in [k for k in row if k.endswith("_client_id") or k.endswith("_filled") or k == "features"]:
            row.pop(key)
        row["r_multiple"] = p.r_multiple
        row["mfe_pct"] = p.mfe_pct
        row["mae_pct"] = p.mae_pct
        rows.append(row)
    return rows


async def run_backtest(backtester: Backtester) -> BacktestResult:
    return await backtester.run()


def run_backtest_sync(backtester: Backtester) -> BacktestResult:
    return asyncio.run(backtester.run())
