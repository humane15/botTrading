"""Pembuat data pasar sintetis untuk test modul analisis (tanpa jaringan)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from analysis.indicators import compute_indicators

START = "2025-01-01"


def make_ohlcv(
    closes: Sequence[float],
    *,
    opens: Sequence[float] | None = None,
    highs: Sequence[float] | None = None,
    lows: Sequence[float] | None = None,
    volumes: Sequence[float] | float = 1000.0,
    wick: float = 0.002,
    freq: str = "1h",
    start: str = START,
) -> pd.DataFrame:
    """Frame OHLCV dari deret close. Open default = close sebelumnya, wick default 0.2%."""
    close = np.asarray(closes, dtype="float64")
    open_ = np.asarray(opens, dtype="float64") if opens is not None else np.r_[close[0], close[:-1]]
    high = np.asarray(highs, dtype="float64") if highs is not None else np.maximum(open_, close) * (1 + wick)
    low = np.asarray(lows, dtype="float64") if lows is not None else np.minimum(open_, close) * (1 - wick)
    volume = np.full(close.shape, float(volumes)) if np.isscalar(volumes) else np.asarray(volumes, dtype="float64")
    index = pd.date_range(start, periods=len(close), freq=freq.replace("m", "min") if freq.endswith("m") else freq, tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index)


def random_walk(n: int, drift: float = 0.0, vol: float = 0.01, seed: int = 0, start: float = 100.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(rng.normal(drift, vol, n)))


def random_ohlcv(n: int, drift: float = 0.0, vol: float = 0.01, seed: int = 0, freq: str = "1h") -> pd.DataFrame:
    """OHLCV acak yang realistis (wick dan volume bervariasi)."""
    rng = np.random.default_rng(seed + 1000)
    close = random_walk(n, drift, vol, seed)
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, vol / 2, n)) * close
    return make_ohlcv(
        close,
        opens=open_,
        highs=np.maximum(open_, close) + wick,
        lows=np.minimum(open_, close) - wick,
        volumes=rng.lognormal(6, 0.4, n),
        freq=freq,
    )


def resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    aggregated = frame.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return aggregated.dropna()


def multi_timeframe(frame_5m: pd.DataFrame, window: int | None = 1000) -> dict[str, pd.DataFrame]:
    """Frame 5m, 15m, 30m, 1h yang konsisten (hasil resample dari data 5m yang sama)."""
    frames = {
        "5m": frame_5m,
        "15m": resample_ohlcv(frame_5m, "15min"),
        "30m": resample_ohlcv(frame_5m, "30min"),
        "1h": resample_ohlcv(frame_5m, "1h"),
    }
    return {tf: (f.iloc[-window:] if window else f) for tf, f in frames.items()}


def indicator_frame(rows: dict[str, Sequence[float]], freq: str = "1h") -> pd.DataFrame:
    """Frame indikator buatan tangan (kolom bebas) untuk menguji fungsi skor secara terisolasi."""
    length = len(next(iter(rows.values())))
    index = pd.date_range(START, periods=length, freq=freq, tz="UTC")
    return pd.DataFrame({k: np.asarray(v, dtype="float64") for k, v in rows.items()}, index=index)


def with_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    return compute_indicators(frame)
