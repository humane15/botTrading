"""Test pivot, Moving Average, RSI, dan MACD."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import indicator_frame, make_ohlcv, random_ohlcv, with_indicators

from analysis.macd import find_recent_cross, score_macd
from analysis.moving_average import (
    TREND_DOWN,
    TREND_SIDEWAYS,
    TREND_UP,
    score_moving_average,
)
from analysis.pivots import find_pivots
from analysis.rsi import detect_divergence, score_rsi


# ----------------------------------------------------------------------
# Pivot
# ----------------------------------------------------------------------
def zigzag_frame():
    #        0  1  2  3   4  5  6  7  8  9 10  11 12 13 14
    highs = [5, 6, 7, 10, 7, 6, 5, 4, 3, 4, 5, 9, 6, 5, 4]
    lows = [h - 1 for h in highs]
    return make_ohlcv(np.array(highs) - 0.5, highs=highs, lows=lows)


def test_pivot_high_dan_low_terdeteksi():
    swing_highs, swing_lows = find_pivots(zigzag_frame(), left=2, right=2)
    assert [(p.position, p.price) for p in swing_highs] == [(3, 10.0), (11, 9.0)]
    assert [(p.position, p.price) for p in swing_lows] == [(8, 2.0)]
    assert all(p.kind == "high" for p in swing_highs)


def test_pivot_baru_terkonfirmasi_setelah_candle_kanan_tutup():
    frame = zigzag_frame()
    # Hanya 1 candle setelah puncak di posisi 11: belum terkonfirmasi (butuh 2).
    swing_highs, _ = find_pivots(frame.iloc[:13], left=2, right=2)
    assert [p.position for p in swing_highs] == [3]


def test_puncak_datar_dihitung_sekali_dan_lookback():
    frame = make_ohlcv([1, 2, 3, 3, 2, 1, 0.5], highs=[1, 2, 3, 3, 2, 1, 0.5], lows=[0, 1, 2, 2, 1, 0, 0])
    swing_highs, _ = find_pivots(frame, left=2, right=2)
    assert [p.position for p in swing_highs] == [2]
    swing_highs, _ = find_pivots(zigzag_frame(), left=2, right=2, lookback=8)
    assert [p.position for p in swing_highs] == [11]
    with pytest.raises(ValueError):
        find_pivots(frame, left=0, right=2)


# ----------------------------------------------------------------------
# Moving Average
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def uptrend():
    return with_indicators(random_ohlcv(600, drift=0.004, vol=0.006, seed=1))


@pytest.fixture(scope="module")
def downtrend():
    return with_indicators(random_ohlcv(600, drift=-0.004, vol=0.006, seed=2))


def test_ma_uptrend(uptrend):
    signal = score_moving_average(uptrend)
    assert signal.details["trend"] == TREND_UP
    assert signal.score > 0.6
    assert "ema_uptrend" in signal.tags
    assert "EMA20 > EMA50 > EMA200" in signal.reason


def test_ma_downtrend(downtrend):
    signal = score_moving_average(downtrend)
    assert signal.details["trend"] == TREND_DOWN
    assert signal.score < -0.6
    assert "ema_downtrend" in signal.tags


def ma_frame(close, e20, e50, e200, low=None, atr=1.0, n=30, e50_slope=0.05):
    ema50 = [e50 - e50_slope * (n - 1 - i) for i in range(n)]
    return indicator_frame({
        "close": [close] * n, "low": [low if low is not None else close - 0.5] * n,
        "ema20": [e20] * n, "ema50": ema50, "ema200": [e200] * n, "atr": [atr] * n,
    })


def test_ma_pullback_ke_ema20():
    signal = score_moving_average(ma_frame(close=105.2, e20=105.0, e50=103.0, e200=100.0, low=104.9))
    assert signal.details["trend"] == TREND_UP
    assert "pullback_ema20" in signal.tags
    assert signal.label == "Pullback EMA 20"


def test_ma_golden_cross():
    n = 30
    ema50 = np.r_[np.full(20, 99.0), np.full(n - 20, 101.0)]  # memotong EMA200 (100) 10 candle lalu
    frame = indicator_frame({
        "close": np.full(n, 103.0), "low": np.full(n, 102.0), "ema20": np.full(n, 102.0),
        "ema50": ema50, "ema200": np.full(n, 100.0), "atr": np.ones(n),
    })
    signal = score_moving_average(frame)
    assert signal.details["golden_cross"] is True
    assert "golden_cross" in signal.tags


def test_ma_tanpa_ema200_dan_data_kurang():
    frame = ma_frame(close=105, e20=104, e50=103, e200=float("nan"))
    signal = score_moving_average(frame)
    assert signal.details["trend"] == TREND_UP
    assert np.isnan(signal.details["dist_ema200_atr"])
    assert score_moving_average(frame.iloc[:1]).score == 0.0
    sideways = score_moving_average(ma_frame(close=101, e20=102, e50=103, e200=100, e50_slope=-0.01))
    assert sideways.details["trend"] == TREND_SIDEWAYS


# ----------------------------------------------------------------------
# RSI
# ----------------------------------------------------------------------
def rsi_frame(value, n=30):
    closes = np.linspace(100, 101, n)
    return indicator_frame({"close": closes, "high": closes + 0.5, "low": closes - 0.5, "rsi": np.full(n, value)})


@pytest.mark.parametrize(
    ("value", "trend", "expected"),
    [
        (25, "sideways", 0.6), (75, "sideways", -0.6), (50, "sideways", 0.0),
        (45, "up", 0.5), (75, "up", 0.0), (85, "up", -0.5),
        (25, "down", 0.0), (65, "down", -0.4),
    ],
)
def test_rsi_dibaca_sesuai_konteks_tren(value, trend, expected):
    assert score_rsi(rsi_frame(value), trend).score == pytest.approx(expected)


def divergence_frame(bullish=True, second_rsi=35.0, tail=4):
    """Dua pivot low (atau high) dengan RSI yang berlawanan arah harga."""
    n = 26 + tail
    base = np.full(n, 100.0)
    rsi = np.full(n, 50.0)
    sign = -1 if bullish else 1
    for pos, depth, value in ((10, 5.0, 25.0 if bullish else 75.0), (22, 7.0, second_rsi)):
        base[pos - 3 : pos + 4] += sign * np.array([1, 2, 3.5, depth, 3.5, 2, 1])
        rsi[pos] = value
    return indicator_frame({"close": base, "high": base + 0.2, "low": base - 0.2, "rsi": rsi})


def test_rsi_bullish_divergence():
    frame = divergence_frame(bullish=True)
    assert detect_divergence(frame) == "bullish"
    signal = score_rsi(frame, "sideways")
    assert "rsi_bull_div" in signal.tags
    assert signal.label == "RSI bullish divergence"
    assert signal.score == pytest.approx(0.4)  # RSI 50 netral + bonus divergence


def test_rsi_bearish_divergence():
    frame = divergence_frame(bullish=False, second_rsi=65.0)
    assert detect_divergence(frame) == "bearish"
    assert "rsi_bear_div" in score_rsi(frame, "sideways").tags


def test_rsi_tanpa_divergence_jika_rsi_searah_harga_atau_sudah_lama():
    assert detect_divergence(divergence_frame(bullish=True, second_rsi=20.0)) is None
    assert detect_divergence(divergence_frame(bullish=True, tail=20)) is None


# ----------------------------------------------------------------------
# MACD
# ----------------------------------------------------------------------
def macd_frame(macd, signal):
    macd = np.asarray(macd, dtype="float64")
    signal = np.asarray(signal, dtype="float64")
    return indicator_frame({"macd": macd, "macd_signal": signal, "macd_hist": macd - signal})


def test_macd_cross_naik_di_bawah_nol_saat_uptrend_lebih_tinggi():
    frame = macd_frame([-0.5, -0.4, -0.3, -0.1], [-0.2, -0.2, -0.2, -0.2])
    assert find_recent_cross(frame, 3) == (1, 3)  # selisih MACD-signal berganti tanda di candle terakhir
    in_uptrend = score_macd(frame, "up")
    in_sideways = score_macd(frame, "sideways")
    assert in_uptrend.score > in_sideways.score
    assert in_uptrend.score == pytest.approx(0.5 + 0.3 - 0.15 + 0.15)
    assert {"macd_cross_up", "macd_cross_below_zero"} <= set(in_uptrend.tags)
    assert in_uptrend.label == "MACD cross di bawah nol"


def test_macd_cross_turun():
    frame = macd_frame([0.5, 0.4, 0.2, 0.1], [0.3, 0.3, 0.3, 0.3])
    signal = score_macd(frame, "up")
    assert "macd_cross_down" in signal.tags
    assert signal.score == pytest.approx(-0.5 + 0.15 - 0.15)


def test_macd_tanpa_cross_dan_cross_lama():
    frame = macd_frame([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.0, 0.1, 0.15, 0.2, 0.25, 0.3])
    signal = score_macd(frame)
    assert signal.score == pytest.approx(0.3)  # di atas nol dan histogram menguat
    old_cross = macd_frame([-0.3, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5], [0, 0, 0, 0, 0, 0, 0])
    assert find_recent_cross(old_cross, 3) is None


def test_macd_data_kurang():
    assert score_macd(macd_frame([np.nan, np.nan], [np.nan, np.nan])).score == 0.0
