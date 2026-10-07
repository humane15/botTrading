"""Test Support & Resistance dan Fibonacci retracement."""

from __future__ import annotations

import math

import numpy as np
import pytest
from synthetic import make_ohlcv

from analysis.fibonacci import find_fib_leg, score_fibonacci
from analysis.support_resistance import (
    Zone,
    find_zones,
    nearest_levels,
    reward_risk,
    score_support_resistance,
    suggest_stop_loss,
)


# ----------------------------------------------------------------------
# Support & Resistance
# ----------------------------------------------------------------------
def range_frame(tail, steps=8):
    """Pasar ranging 100..110 dengan tiga sentuhan di tiap sisi, lalu deret `tail`."""
    points = [105, 110, 100, 110.2, 100.3, 109.9, 99.8, 105]
    path = []
    for start, end in zip(points[:-1], points[1:], strict=True):
        path.extend(np.linspace(start, end, steps, endpoint=False))
    path.extend(tail)
    path = np.asarray(path, dtype="float64")
    frame = make_ohlcv(path, highs=np.r_[path[0], np.maximum(path[:-1], path[1:])] + 0.1, lows=np.r_[path[0], np.minimum(path[:-1], path[1:])] - 0.1)
    frame["atr"] = 1.0
    frame["volume_ratio"] = 1.0
    return frame


def test_pivot_dikelompokkan_menjadi_zona():
    zones = find_zones(range_frame([105, 104]), "1h")
    assert len(zones) == 2
    support, resistance = zones
    assert support.touches == 3 and resistance.touches == 3
    assert 99.6 <= support.low <= support.high <= 100.4
    assert 109.9 <= resistance.low <= resistance.high <= 110.5
    assert 0 < support.strength <= 1 and support.timeframe == "1h"


def test_pivot_yang_berjauhan_tidak_digabung():
    frame = range_frame([105])
    frame["atr"] = 0.2  # toleransi 0.1: sentuhan 110.1, 110.3, 110.0 tidak lagi satu zona
    assert len(find_zones(frame)) > 2


def test_support_dan_resistance_terdekat():
    zones = find_zones(range_frame([105, 104]))
    support, resistance = nearest_levels(105, zones)
    assert support.center < 105 < resistance.center
    assert nearest_levels(98, zones)[0] is None
    assert nearest_levels(112, zones)[1] is None


def test_pantulan_dari_support_kuat_bernilai_positif():
    signal = score_support_resistance(range_frame([104, 102.5, 101, 100.4, 100.1, 100.9]), find_zones(range_frame([105])))
    assert "pantulan_support" in signal.tags
    assert signal.label == "Pantulan support"
    assert signal.score > 0.5


def test_mendekati_resistance_bernilai_negatif():
    frame = range_frame([106, 107.5, 108.5, 109.2, 109.6])
    signal = score_support_resistance(frame, find_zones(frame))
    assert "dekat_resistance" in signal.tags
    assert signal.score < 0
    assert signal.details["dist_resistance_atr"] == pytest.approx(0.4, abs=0.15)


def test_breakout_resistance_bukan_pantulan():
    frame = range_frame([106, 107.5, 108.8, 109.6, 109.8, 111.4])
    signal = score_support_resistance(frame, find_zones(frame))
    assert "breakout_resistance" in signal.tags
    assert "pantulan_support" not in signal.tags
    assert signal.score > 0


@pytest.mark.parametrize(
    ("support", "expected_stop"),
    [
        (Zone(104.5, 104.8, 2, 0.5, 0), 103.5),   # support terlalu dekat: minimal 1.5 ATR
        (Zone(100.0, 100.5, 2, 0.5, 0), 102.5),   # support terlalu jauh: maksimal 2.5 ATR
        (Zone(103.0, 103.3, 2, 0.5, 0), 102.8),   # di bawah support (buffer 0.2 ATR)
        (None, 103.0),                            # tanpa support: 2 ATR
    ],
)
def test_stop_loss_di_bawah_support_dibatasi_atr(support, expected_stop):
    assert suggest_stop_loss(105.0, 1.0, support) == pytest.approx(expected_stop)


def test_rasio_ruang_ke_resistance_terhadap_risiko():
    resistance = Zone(111.0, 111.5, 3, 0.7, 0)
    assert reward_risk(105.0, 103.0, resistance) == pytest.approx(3.0)
    assert math.isinf(reward_risk(105.0, 103.0, None))
    assert reward_risk(105.0, 103.0, Zone(104.0, 106.0, 2, 0.5, 0)) == 0.0
    assert reward_risk(105.0, 106.0, resistance) == 0.0


def test_sr_tanpa_zona_atau_data():
    frame = range_frame([105])
    assert score_support_resistance(frame, []).score == 0.0
    assert find_zones(frame.drop(columns=["atr"])) == []


# ----------------------------------------------------------------------
# Fibonacci
# ----------------------------------------------------------------------
def fib_frame(pullback_low, last_open, last_close, atr=1.0):
    """Leg naik tepat 100 -> 120, turun ke `pullback_low`, lalu candle terakhir."""
    pre = np.full(5, 101.0)
    up = np.linspace(100.5, 119.9, 31)
    down = np.linspace(119.0, max(pullback_low + 0.5, 99.5), 7)
    closes = np.r_[pre, up, down, last_close]
    opens = np.r_[closes[0], closes[:-1]]
    opens[-1] = last_open
    highs = np.maximum(opens, closes) + 0.05
    lows = np.minimum(opens, closes) - 0.05
    lows[5] = 100.0             # titik awal leg
    highs[5 + 30] = 120.0       # puncak leg
    lows[-1] = pullback_low     # titik terdalam koreksi
    frame = make_ohlcv(closes, opens=opens, highs=highs, lows=lows)
    frame["atr"] = atr
    return frame


def test_leg_fibonacci_dan_extension():
    leg = find_fib_leg(fib_frame(107.6, 107.8, 108.4), "1h")
    assert (leg.low, leg.high, leg.timeframe) == (100.0, 120.0, "1h")
    assert leg.level(0.618) == pytest.approx(107.64)
    assert leg.extensions() == pytest.approx((125.44, 132.36))
    assert leg.retracement(110.0) == pytest.approx(0.5)


def test_pantulan_golden_zone_618():
    frame = fib_frame(107.6, 107.8, 108.4)
    signal = score_fibonacci(frame, find_fib_leg(frame, "1h"))
    assert "fib_618" in signal.tags
    assert signal.score == pytest.approx(0.6)
    assert signal.label == "Pantulan Fib 61.8% 1h"
    assert signal.details["bounce"] is True


def test_skor_tertinggi_saat_confluence_dengan_support_atau_ema():
    frame = fib_frame(107.6, 107.8, 108.4)
    leg = find_fib_leg(frame, "1h")
    with_support = score_fibonacci(frame, leg, support_zones=[Zone(107.4, 107.9, 3, 0.7, 0, timeframe="1h")])
    assert with_support.score == pytest.approx(0.9)
    assert "fib_confluence" in with_support.tags
    assert with_support.label == "Pantulan Fib 61.8% + support 1h"
    with_ema = score_fibonacci(frame, leg, ema_levels=[107.8])
    assert with_ema.score == pytest.approx(0.9)
    assert with_ema.label == "Pantulan Fib 61.8% + EMA 1h"


def test_tanpa_pantulan_skor_dipotong_setengah():
    frame = fib_frame(107.6, 108.4, 107.8)  # candle terakhir masih turun
    signal = score_fibonacci(frame, find_fib_leg(frame))
    assert signal.score == pytest.approx(0.3)
    assert signal.label == "Fib 61.8%"


@pytest.mark.parametrize(
    ("low", "open_", "close", "tag", "score"),
    [
        (112.3, 112.5, 113.0, "fib_382", 0.4),
        (110.1, 110.3, 110.9, "fib_50", 0.6),
        (104.4, 104.6, 105.2, "fib_786", 0.3),
        (103.0, 103.2, 103.8, "fib_terlalu_dalam", -0.2),
        (98.5, 99.5, 99.0, "fib_patah", -0.5),
    ],
)
def test_level_fibonacci_lain(low, open_, close, tag, score):
    frame = fib_frame(low, open_, close)
    signal = score_fibonacci(frame, find_fib_leg(frame))
    assert tag in signal.tags
    assert signal.score == pytest.approx(score)


def test_leg_tidak_valid():
    assert find_fib_leg(fib_frame(107.6, 107.8, 108.4, atr=10.0)) is None  # swing < 3 ATR
    rising = make_ohlcv(np.linspace(100, 130, 60))
    rising["atr"] = 1.0
    assert find_fib_leg(rising) is None  # puncak masih terbentuk (belum ada koreksi)
    assert score_fibonacci(rising, None).score == 0.0
