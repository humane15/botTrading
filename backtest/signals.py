"""Tahap 1 backtest: sinyal per coin dengan ConfluenceEngine yang sama persis dengan live.

Cara kerja tiap langkah (setiap candle 5m tertutup pada waktu T):
* setiap timeframe hanya berisi candle yang SUDAH TERTUTUP pada T, maksimal 1000
  candle terakhir (jendela yang sama dengan data feed live);
* indikator dihitung sekali di seluruh riwayat lalu diiris per langkah. Ini aman
  karena semua indikator kausal (diuji di Fase 2), sedangkan pivot, zona S&R, dan
  Fibonacci tetap dihitung ulang dari jendela yang diiris, persis seperti live;
* konteks pasar (regime BTC dan circuit breaker) dihitung berurutan dari data BTC
  dengan kode yang sama dengan live.

Sinyal tiap coin tidak bergantung pada saldo atau posisi, jadi bisa dihitung
paralel per coin di beberapa proses. Simulasi portofolio (tahap 2,
backtest/engine.py) lalu memproses sinyal ini secara berurutan waktu.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import Any, Protocol

import numpy as np
import pandas as pd

from analysis.confluence import (
    ConfluenceEngine,
    EngineParams,
    SignalResult,
    prepare_frames,
)
from analysis.regime import CircuitBreaker, MarketContext, analyze_market
from backtest.data import HistoryStore
from config.settings import TIMEFRAME_MS, Settings
from core.data_feed import frame_open_ms

log = logging.getLogger(__name__)


def step_times(start: pd.Timestamp, end: pd.Timestamp, timeframe: str = "5m") -> np.ndarray:
    """Waktu tutup (ms) setiap candle `timeframe` yang dibuka di [start, end)."""
    tf_ms = TIMEFRAME_MS[timeframe]
    first = -(-int(start.value // 1_000_000) // tf_ms) * tf_ms
    last = int(end.value // 1_000_000)
    opens = np.arange(first, last, tf_ms, dtype="int64")
    return opens + tf_ms


class FrameSource(Protocol):
    def load(self, symbol: str) -> dict[str, pd.DataFrame]: ...


@dataclass(frozen=True)
class StoreSource:
    """Data dari HistoryStore (bisa dikirim ke proses lain: hanya berisi path)."""

    root: str
    timeframes: tuple[str, ...]

    def load(self, symbol: str) -> dict[str, pd.DataFrame]:
        return HistoryStore(self.root).load_symbol(symbol, self.timeframes)


@dataclass(frozen=True)
class MemorySource:
    """Data yang sudah ada di memori (test dan demo)."""

    frames: Mapping[str, Mapping[str, pd.DataFrame]]

    def load(self, symbol: str) -> dict[str, pd.DataFrame]:
        return dict(self.frames[symbol])


class WindowSlicer:
    """Jendela candle tertutup per timeframe untuk setiap langkah backtest.

    counts[tf][i] = jumlah candle tf yang sudah tertutup pada waktu langkah i.
    Irisan timeframe besar dipakai ulang selama candle barunya belum tertutup,
    sehingga cache analisis engine tetap kena (kunci cache sama).
    """

    def __init__(self, frames: Mapping[str, pd.DataFrame], steps_ms: np.ndarray, window: int = 1000) -> None:
        self.frames = dict(frames)
        self.window = window
        self.steps_ms = steps_ms
        self.opens = {tf: frame_open_ms(frame) for tf, frame in self.frames.items()}
        self.counts = {
            tf: np.searchsorted(opens, steps_ms - TIMEFRAME_MS[tf], side="right") for tf, opens in self.opens.items()
        }
        self._cache: dict[str, tuple[int, pd.DataFrame]] = {}

    def count(self, timeframe: str, i: int) -> int:
        return int(self.counts[timeframe][i])

    def closes_at(self, timeframe: str, i: int) -> bool:
        """True jika ada candle `timeframe` yang tepat tertutup pada langkah i (data tidak bolong)."""
        k = self.count(timeframe, i)
        return k > 0 and int(self.opens[timeframe][k - 1]) == int(self.steps_ms[i]) - TIMEFRAME_MS[timeframe]

    def frame(self, timeframe: str, i: int) -> pd.DataFrame:
        k = self.count(timeframe, i)
        cached = self._cache.get(timeframe)
        if cached is not None and cached[0] == k:
            return cached[1]
        view = self.frames[timeframe].iloc[max(0, k - self.window):k]
        self._cache[timeframe] = (k, view)
        return view

    def windows(self, i: int) -> dict[str, pd.DataFrame]:
        return {tf: self.frame(tf, i) for tf in self.frames}


def market_contexts(
    btc_frames: Mapping[str, pd.DataFrame], steps_ms: np.ndarray, params: EngineParams, settings: Settings
) -> list[MarketContext]:
    """Konteks pasar (regime BTC + circuit breaker) di setiap langkah, dihitung berurutan."""
    trend_tf, trigger_tf = params.timeframe_of("trend"), params.timeframe_of("trigger")
    breaker = CircuitBreaker.from_settings(settings)
    slicer = WindowSlicer({tf: btc_frames[tf] for tf in (trend_tf, trigger_tf)}, steps_ms, settings.ohlcv_limit)
    contexts: list[MarketContext] = []
    for i in range(len(steps_ms)):
        contexts.append(
            analyze_market(
                slicer.windows(i), breaker, trend_timeframe=trend_tf, breaker_timeframe=trigger_tf, params=params.regime
            )
        )
    return contexts


@dataclass
class SymbolSignals:
    symbol: str
    entries: list[SignalResult] = field(default_factory=list)
    evaluations: int = 0
    rejections: Counter[str] = field(default_factory=Counter)


def symbol_signals(
    symbol: str,
    raw_frames: Mapping[str, pd.DataFrame],
    steps_ms: np.ndarray,
    active: np.ndarray,
    contexts: Sequence[MarketContext],
    weights: Mapping[str, float],
    params: EngineParams,
    window: int = 1000,
) -> SymbolSignals:
    """Evaluasi satu coin di setiap langkah aktif (coin masuk universe hari itu)."""
    result = SymbolSignals(symbol)
    if any(raw_frames.get(tf) is None or raw_frames[tf].empty for tf in params.timeframes):
        return result
    frames = prepare_frames({tf: raw_frames[tf] for tf in params.timeframes}, params.indicators)
    engine = ConfluenceEngine(weights, params)
    slicer = WindowSlicer(frames, steps_ms, window)
    trigger_tf = params.timeframe_of("trigger")
    for i in np.flatnonzero(active):
        if not slicer.closes_at(trigger_tf, int(i)):
            continue  # candle 5m langkah ini tidak ada (belum listing atau data bolong)
        signal = engine.evaluate(symbol, slicer.windows(int(i)), contexts[int(i)], fast_reject=True)
        result.evaluations += 1
        if signal.is_entry:
            result.entries.append(signal)
        else:
            result.rejections.update(signal.rejections)
    return result


# ----------------------------------------------------------------------
# Eksekusi paralel per coin
# ----------------------------------------------------------------------
_WORKER: dict[str, Any] = {}


def _init_worker(
    source: FrameSource, steps_ms: np.ndarray, contexts: list[MarketContext], weights: dict[str, float], params: EngineParams, window: int
) -> None:
    logging.getLogger("analysis").setLevel(logging.ERROR)
    _WORKER.update(source=source, steps_ms=steps_ms, contexts=contexts, weights=weights, params=params, window=window)


def _worker_task(symbol: str, active: np.ndarray) -> SymbolSignals:
    w = _WORKER
    return symbol_signals(
        symbol, w["source"].load(symbol), w["steps_ms"], active, w["contexts"], w["weights"], w["params"], w["window"]
    )


def generate_signals(
    symbols: Sequence[str],
    source: FrameSource,
    steps_ms: np.ndarray,
    active: Mapping[str, np.ndarray],
    contexts: list[MarketContext],
    weights: Mapping[str, float],
    params: EngineParams,
    *,
    window: int = 1000,
    workers: int = 1,
    progress: Callable[[str], None] | None = None,
) -> dict[str, SymbolSignals]:
    """Sinyal semua coin. workers > 1 memakai beberapa proses (hasil identik dengan 1 proses)."""
    results: dict[str, SymbolSignals] = {}
    todo = [s for s in symbols if active[s].any()]

    def report(done: int) -> None:
        if progress is not None and (done % 10 == 0 or done == len(todo)):
            progress(f"  sinyal {done}/{len(todo)} coin")

    if workers <= 1 or len(todo) <= 1:
        for done, symbol in enumerate(todo, start=1):
            results[symbol] = symbol_signals(
                symbol, source.load(symbol), steps_ms, active[symbol], contexts, weights, params, window
            )
            report(done)
    else:
        initargs = (source, steps_ms, contexts, dict(weights), params, window)
        # "spawn" agar perilaku sama di Linux, macOS, dan Windows.
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=_init_worker, initargs=initargs) as pool:
            futures = {pool.submit(_worker_task, symbol, active[symbol]): symbol for symbol in todo}
            for done, future in enumerate(as_completed(futures), start=1):
                results[futures[future]] = future.result()
                report(done)
    return {symbol: results[symbol] for symbol in todo}
