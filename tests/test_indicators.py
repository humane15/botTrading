"""Test indikator inti: kebenaran rumus, kecocokan dengan TA-Lib, dan anti look ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from synthetic import make_ohlcv, random_ohlcv

from analysis import indicators as ind
from analysis.indicators import compute_indicators

EXPECTED_COLUMNS = {
    "open", "high", "low", "close", "volume", "ema20", "ema50", "ema200", "rsi", "macd", "macd_signal",
    "macd_hist", "bb_mid", "bb_upper", "bb_lower", "bb_width", "bb_pctb", "bb_width_rank", "atr", "atr_pct",
    "atr_pct_rank", "adx", "plus_di", "minus_di", "obv", "obv_ema", "volume_ratio",
}


@pytest.fixture(scope="module")
def market() -> pd.DataFrame:
    return random_ohlcv(3000, drift=0.0001, vol=0.01, seed=7)


def test_ema_diawali_sma():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = ind.ema(series, 3).tolist()
    assert np.isnan(result[0]) and np.isnan(result[1])
    assert result[2:] == pytest.approx([2.0, 3.0, 4.0, 5.0])  # seed = rata rata 1,2,3; alpha 0.5


def test_rsi_nilai_ekstrem():
    naik = pd.Series(np.arange(1.0, 40.0))
    turun = pd.Series(np.arange(40.0, 1.0, -1))
    datar = pd.Series(np.full(40, 5.0))
    assert ind.rsi(naik).iloc[-1] == pytest.approx(100.0)
    assert ind.rsi(turun).iloc[-1] == pytest.approx(0.0)
    assert ind.rsi(datar).iloc[-1] == pytest.approx(50.0)
    assert ind.rsi(naik).iloc[:14].isna().all()


def test_bollinger_harga_datar():
    bands = ind.bollinger(pd.Series(np.full(30, 10.0)))
    assert bands["bb_width"].iloc[-1] == 0.0
    assert bands["bb_pctb"].iloc[-1] == 0.5
    assert bands["bb_upper"].iloc[-1] == bands["bb_lower"].iloc[-1] == 10.0


def test_volume_ratio_tidak_memasukkan_candle_sendiri():
    volume = pd.Series([100.0] * 20 + [300.0])
    assert ind.volume_ratio(volume).iloc[-1] == pytest.approx(3.0)
    assert np.isnan(ind.volume_ratio(volume).iloc[19])  # belum ada 20 candle sebelumnya


def test_obv_mengikuti_arah_harga():
    close = pd.Series([10.0, 11.0, 10.5, 10.5, 12.0])
    volume = pd.Series([100.0, 200.0, 50.0, 70.0, 30.0])
    assert ind.obv(close, volume).tolist() == [100.0, 300.0, 250.0, 250.0, 280.0]


def test_atr_dan_true_range():
    frame = make_ohlcv([10, 11, 12], highs=[10.5, 11.5, 12.8], lows=[9.5, 10.2, 11.0])
    tr = ind.true_range(frame["high"], frame["low"], frame["close"])
    assert np.isnan(tr.iloc[0])
    assert tr.iloc[1] == pytest.approx(1.5)   # max(1.3, |11.5-10|, |10.2-10|)
    assert tr.iloc[2] == pytest.approx(1.8)


def test_compute_indicators_kolom_lengkap(market):
    frame = compute_indicators(market.iloc[-1000:])
    assert set(frame.columns) == EXPECTED_COLUMNS
    last = frame.iloc[-1]
    assert last.notna().all()
    assert 0 <= last["rsi"] <= 100 and 0 <= last["adx"] <= 100
    assert 0 < last["bb_width_rank"] <= 1 and 0 < last["atr_pct_rank"] <= 1


@pytest.mark.parametrize("cut", [150, 777, 1999, 2999])
def test_tanpa_look_ahead(market, cut):
    """Nilai indikator di baris t tidak berubah walau data setelah t ditambahkan."""
    full = compute_indicators(market)
    partial = compute_indicators(market.iloc[: cut + 1])
    pd.testing.assert_frame_equal(full.iloc[: cut + 1], partial, check_exact=False, rtol=1e-9, atol=1e-9)


def test_jendela_1000_candle_hampir_sama_dengan_riwayat_penuh(market):
    """Live memakai 1000 candle terakhir, backtest memakai riwayat penuh: hasil harus setara."""
    full = compute_indicators(market).iloc[-1]
    window = compute_indicators(market.iloc[-1000:]).iloc[-1]
    for column in EXPECTED_COLUMNS - {"obv", "obv_ema"}:
        assert window[column] == pytest.approx(full[column], rel=1e-4, abs=1e-9), column
    # Level OBV bergantung titik awal, tetapi selisih OBV dengan EMA-nya tidak.
    assert window["obv"] - window["obv_ema"] == pytest.approx(full["obv"] - full["obv_ema"], rel=1e-6)


# ----------------------------------------------------------------------
# Pembanding TA-Lib (dilewati jika TA-Lib tidak terpasang)
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def talib():
    return pytest.importorskip("talib")


def _assert_close(mine: pd.Series, reference: np.ndarray, skip: int = 0, rtol: float = 1e-9) -> None:
    mine_values = mine.to_numpy()
    mask = ~(np.isnan(mine_values) | np.isnan(reference))
    mask[:skip] = False
    assert mask.sum() > 100
    np.testing.assert_allclose(mine_values[mask], reference[mask], rtol=rtol, atol=1e-9)


def test_cocok_dengan_talib(market, talib):
    c, h, l, v = (market[k].to_numpy() for k in ("close", "high", "low", "volume"))
    close, high, low, volume = market["close"], market["high"], market["low"], market["volume"]
    _assert_close(ind.ema(close, 20), talib.EMA(c, 20))
    _assert_close(ind.ema(close, 200), talib.EMA(c, 200))
    _assert_close(ind.rsi(close, 14), talib.RSI(c, 14))
    _assert_close(ind.atr(high, low, close, 14), talib.ATR(h, l, c, 14))
    adx = ind.adx(high, low, close, 14)
    _assert_close(adx["adx"], talib.ADX(h, l, c, 14))
    _assert_close(adx["plus_di"], talib.PLUS_DI(h, l, c, 14))
    _assert_close(adx["minus_di"], talib.MINUS_DI(h, l, c, 14))
    upper, middle, lower = talib.BBANDS(c, 20, 2, 2, 0)
    bands = ind.bollinger(close)
    _assert_close(bands["bb_upper"], upper)
    _assert_close(bands["bb_mid"], middle)
    _assert_close(bands["bb_lower"], lower)
    _assert_close(ind.obv(close, volume), talib.OBV(c, v))
    # TA-Lib memulai EMA cepat MACD sedikit berbeda; setelah pemanasan hasilnya sama.
    macd_line, signal, hist = talib.MACD(c, 12, 26, 9)
    mine = ind.macd(close)
    _assert_close(mine["macd"], macd_line, skip=300, rtol=1e-6)
    _assert_close(mine["macd_signal"], signal, skip=300, rtol=1e-6)
    _assert_close(mine["macd_hist"], hist, skip=300, rtol=1e-5)
