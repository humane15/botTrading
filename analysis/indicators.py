"""Indikator teknikal versi pandas/numpy murni (satu sumber perhitungan).

Keputusan desain:

* Semua indikator ditulis sendiri dengan pandas/numpy supaya hasil backtest
  dan live dijamin identik dan tidak bergantung pada library yang rawan gagal
  di-install. Rumus mengikuti konvensi TA-Lib (EMA dan smoothing Wilder diawali
  SMA, standar deviasi populasi untuk Bollinger) dan dicocokkan dengan TA-Lib
  di unit test.
* Semua fungsi KAUSAL: nilai di baris t hanya memakai data sampai baris t.
  Tidak ada rolling center, bfill, atau shift negatif, sehingga aman dipakai
  di backtest tanpa look ahead.
* EMA punya memori panjang, jadi nilainya sedikit bergantung pada titik awal
  data. Karena itu data feed menyimpan sampai 1000 candle: selisih EMA 200
  antara jendela 1000 candle dan riwayat penuh sekitar 0.0004%.
* Perhitungan inti memakai array numpy agar satu frame 1000 candle selesai
  dalam beberapa milidetik (75 coin x 4 timeframe tiap scan).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IndicatorParams:
    """Parameter indikator sesuai spesifikasi strategi."""

    ema_periods: tuple[int, int, int] = (20, 50, 200)
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    bb_rank_window: int = 100
    atr_period: int = 14
    atr_rank_window: int = 500
    adx_period: int = 14
    volume_window: int = 20
    obv_ema_period: int = 20


DEFAULT_PARAMS = IndicatorParams()


# ----------------------------------------------------------------------
# Inti numpy
# ----------------------------------------------------------------------
def _ewm(values: np.ndarray, alpha: float, period: int, seed: float | None = None, seed_count: int | None = None) -> np.ndarray:
    """EMA rekursif y_t = y_(t-1) + alpha * (x_t - y_(t-1)) yang diawali seed.

    Default seed adalah SMA dari `period` nilai valid pertama (konvensi TA-Lib),
    sehingga output pertama ada di nilai valid ke-`period`. Nilai setelah
    nilai valid pertama diasumsikan tidak ada yang NaN.
    """
    out = np.full(values.shape, np.nan)
    valid = np.flatnonzero(~np.isnan(values))
    count = period if seed_count is None else seed_count
    if valid.size < max(count, 1) or valid.size < period:
        return out
    start = int(valid[0])
    seed_pos = start + count - 1
    tail = values[seed_pos:].copy()
    tail[0] = values[start : seed_pos + 1].mean() if seed is None else seed
    out[seed_pos:] = pd.Series(tail, copy=False).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    return out


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values, copy=False).rolling(window, min_periods=window).mean().to_numpy()


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.r_[np.nan, close[:-1]]
    with np.errstate(invalid="ignore"):
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    if tr.size:
        tr[0] = np.nan  # seperti TA-Lib: TR butuh close sebelumnya
    return tr


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    change = np.r_[np.nan, np.diff(close)]
    avg_gain = _ewm(np.where(np.isnan(change), np.nan, np.maximum(change, 0.0)), 1.0 / period, period)
    avg_loss = _ewm(np.where(np.isnan(change), np.nan, np.maximum(-change, 0.0)), 1.0 / period, period)
    total = avg_gain + avg_loss
    with np.errstate(divide="ignore", invalid="ignore"):
        value = 100.0 * avg_gain / total
    # Pasar datar total (tanpa naik turun) diberi nilai netral 50.
    return np.where(total == 0, 50.0, value)


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    up = np.r_[np.nan, np.diff(high)]
    down = np.r_[np.nan, -np.diff(low)]
    with np.errstate(invalid="ignore"):
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm[np.isnan(up)] = np.nan
    minus_dm[np.isnan(down)] = np.nan
    tr = _true_range(high, low, close)

    def wilder_sum_smooth(values: np.ndarray) -> np.ndarray:
        # TA-Lib: jumlahkan n-1 nilai pertama, lalu S_t = S_(t-1) - S_(t-1)/n + x_t.
        # Dibagi n, rumus ini sama dengan EMA alpha 1/n yang diawali jumlah/n.
        valid = values[~np.isnan(values)]
        if valid.size < period:
            return np.full(values.shape, np.nan)
        seed = valid[: period - 1].sum() / period
        return _ewm(values, 1.0 / period, period, seed=seed, seed_count=period - 1)

    smooth_tr = wilder_sum_smooth(tr)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * wilder_sum_smooth(plus_dm) / smooth_tr
        minus_di = 100.0 * wilder_sum_smooth(minus_dm) / smooth_tr
        # Seperti TA-Lib, DI pertama keluar setelah satu langkah smoothing (baris ke-n).
        plus_di[:period] = np.nan
        minus_di[:period] = np.nan
        di_sum = plus_di + minus_di
        dx = np.where(di_sum == 0, 0.0, 100.0 * np.abs(plus_di - minus_di) / di_sum)
    dx[np.isnan(plus_di)] = np.nan
    return _ewm(dx, 1.0 / period, period), plus_di, minus_di


def _obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    if close.size == 0:
        return close.copy()
    flow = np.sign(np.r_[0.0, np.diff(close)]) * volume
    flow[0] = volume[0]  # seperti TA-Lib: OBV awal = volume candle pertama
    return np.cumsum(flow)


# ----------------------------------------------------------------------
# API publik berbasis pandas
# ----------------------------------------------------------------------
def _series(values: np.ndarray, like: pd.Series, name: str | None = None) -> pd.Series:
    return pd.Series(values, index=like.index, name=name)


def _arr(series: pd.Series) -> np.ndarray:
    return series.to_numpy(dtype="float64")


def sma(series: pd.Series, period: int) -> pd.Series:
    return _series(_rolling_mean(_arr(series), period), series)


def ema(series: pd.Series, period: int) -> pd.Series:
    """EMA dengan alpha 2/(n+1), diawali SMA n nilai pertama."""
    return _series(_ewm(_arr(series), 2.0 / (period + 1), period), series)


def rma(series: pd.Series, period: int) -> pd.Series:
    """Smoothing Wilder (RMA) dengan alpha 1/n, diawali SMA n nilai pertama."""
    return _series(_ewm(_arr(series), 1.0 / period, period), series)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI Wilder. Pasar datar total (tanpa naik turun) diberi nilai 50."""
    return _series(_rsi(_arr(close), period), close)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    values = _arr(close)
    line = _ewm(values, 2.0 / (fast + 1), fast) - _ewm(values, 2.0 / (slow + 1), slow)
    signal_line = _ewm(line, 2.0 / (signal + 1), signal)
    return pd.DataFrame({"macd": line, "macd_signal": signal_line, "macd_hist": line - signal_line}, index=close.index)


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands dengan standar deviasi populasi (ddof=0) seperti TA-Lib."""
    rolling = close.astype("float64").rolling(period, min_periods=period)
    mid = rolling.mean().to_numpy()
    std = rolling.std(ddof=0).to_numpy()
    upper = mid + num_std * std
    lower = mid - num_std * std
    band = upper - lower
    with np.errstate(divide="ignore", invalid="ignore"):
        width = band / mid
        pct_b = np.where(band == 0, 0.5, (_arr(close) - lower) / band)
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_width": width, "bb_pctb": pct_b},
        index=close.index,
    )


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range. Baris pertama NaN karena butuh close sebelumnya (seperti TA-Lib)."""
    return _series(_true_range(_arr(high), _arr(low), _arr(close)), close)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = _true_range(_arr(high), _arr(low), _arr(close))
    return _series(_ewm(tr, 1.0 / period, period), close)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """ADX, +DI, dan -DI (Wilder) dengan inisialisasi yang sama seperti TA-Lib."""
    adx_values, plus_di, minus_di = _adx(_arr(high), _arr(low), _arr(close), period)
    return pd.DataFrame({"adx": adx_values, "plus_di": plus_di, "minus_di": minus_di}, index=close.index)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On Balance Volume, nilai awal = volume candle pertama (seperti TA-Lib).

    Level OBV bergantung pada titik awal data, jadi yang dipakai untuk analisis
    hanya bentuk relatifnya (OBV dibanding EMA-nya), yang tidak terpengaruh.
    """
    return _series(_obv(_arr(close), _arr(volume)), close)


def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume candle ini dibanding rata rata `period` candle SEBELUMNYA.

    Candle sendiri tidak ikut dirata rata supaya lonjakan volume tidak
    mengecilkan rasionya sendiri.
    """
    values = _arr(volume)
    baseline = np.r_[np.nan, _rolling_mean(values, period)[:-1]] if values.size else values
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(baseline > 0, values / baseline, np.nan)
    return _series(ratio, volume)


def rolling_rank_pct(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Persentil nilai terakhir dalam jendela `window` candle (0..1)."""
    return series.rolling(window, min_periods=min_periods or window).rank(pct=True)


def compute_indicators(ohlcv: pd.DataFrame, params: IndicatorParams = DEFAULT_PARAMS) -> pd.DataFrame:
    """Hitung semua indikator untuk satu frame OHLCV (satu simbol, satu timeframe).

    Kolom hasil: open, high, low, close, volume, ema20, ema50, ema200, rsi,
    macd, macd_signal, macd_hist, bb_mid, bb_upper, bb_lower, bb_width,
    bb_pctb, bb_width_rank, atr, atr_pct, atr_pct_rank, adx, plus_di,
    minus_di, obv, obv_ema, volume_ratio.
    """
    index = ohlcv.index
    open_ = _arr(ohlcv["open"])
    high = _arr(ohlcv["high"])
    low = _arr(ohlcv["low"])
    close = _arr(ohlcv["close"])
    volume = _arr(ohlcv["volume"])

    columns: dict[str, np.ndarray] = {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    for period in params.ema_periods:
        columns[f"ema{period}"] = _ewm(close, 2.0 / (period + 1), period)
    columns["rsi"] = _rsi(close, params.rsi_period)

    fast = _ewm(close, 2.0 / (params.macd_fast + 1), params.macd_fast)
    slow = _ewm(close, 2.0 / (params.macd_slow + 1), params.macd_slow)
    columns["macd"] = fast - slow
    columns["macd_signal"] = _ewm(columns["macd"], 2.0 / (params.macd_signal + 1), params.macd_signal)
    columns["macd_hist"] = columns["macd"] - columns["macd_signal"]

    bands = bollinger(ohlcv["close"], params.bb_period, params.bb_std)
    for name in bands.columns:
        columns[name] = bands[name].to_numpy()

    tr = _true_range(high, low, close)
    columns["atr"] = _ewm(tr, 1.0 / params.atr_period, params.atr_period)
    with np.errstate(divide="ignore", invalid="ignore"):
        columns["atr_pct"] = columns["atr"] / close
    columns["adx"], columns["plus_di"], columns["minus_di"] = _adx(high, low, close, params.adx_period)
    columns["obv"] = _obv(close, volume)
    columns["obv_ema"] = _ewm(columns["obv"], 2.0 / (params.obv_ema_period + 1), params.obv_ema_period)
    columns["volume_ratio"] = volume_ratio(ohlcv["volume"], params.volume_window).to_numpy()

    frame = pd.DataFrame(columns, index=index)
    frame["bb_width_rank"] = rolling_rank_pct(frame["bb_width"], params.bb_rank_window)
    frame["atr_pct_rank"] = rolling_rank_pct(frame["atr_pct"], params.atr_rank_window, min_periods=100)
    return frame
