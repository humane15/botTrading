"""Test Bollinger Bands, volume/OBV, trigger candle, dan struktur pasar."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import indicator_frame, make_ohlcv, random_walk, with_indicators

from analysis.bollinger import score_bollinger
from analysis.candles import score_trigger_candle
from analysis.structure import score_structure
from analysis.volume import score_volume


# ----------------------------------------------------------------------
# Bollinger Bands
# ----------------------------------------------------------------------
def bb_frame(close, *, upper=110.0, lower=90.0, rank=0.5, ranks_before=None, volume_ratio=1.0, n=15):
    ranks = list(ranks_before) if ranks_before is not None else [rank] * (n - 1)
    ranks = ranks[-(n - 1) :] + [rank]
    pct_b = (close - lower) / (upper - lower)
    return indicator_frame({
        "close": [100.0] * (n - 1) + [close],
        "bb_upper": [upper] * n, "bb_lower": [lower] * n, "bb_pctb": [0.5] * (n - 1) + [pct_b],
        "bb_width_rank": ranks, "volume_ratio": [1.0] * (n - 1) + [volume_ratio],
    })


def test_breakout_setelah_squeeze_dengan_volume_sinyal_kuat():
    signal = score_bollinger(bb_frame(112, rank=0.6, ranks_before=[0.1] * 14, volume_ratio=2.3))
    assert signal.score == pytest.approx(0.8)
    assert {"bb_breakout", "bb_squeeze_breakout"} <= set(signal.tags)
    assert signal.label == "BB squeeze breakout"


def test_breakout_tanpa_volume_lemah_dan_tanpa_squeeze_sedang():
    weak = score_bollinger(bb_frame(112, ranks_before=[0.1] * 14, volume_ratio=0.8))
    assert weak.score == pytest.approx(0.1)
    assert "bb_breakout_lemah" in weak.tags
    normal = score_bollinger(bb_frame(112, rank=0.6, volume_ratio=2.0))
    assert normal.score == pytest.approx(0.4)


def test_squeeze_ditandai_sebagai_potensi_breakout():
    signal = score_bollinger(bb_frame(100, rank=0.1))
    assert "bb_squeeze" in signal.tags
    assert signal.score == 0.0
    assert "squeeze" in signal.reason


def test_mean_reversion_di_pasar_ranging():
    near_lower = bb_frame(91)
    assert score_bollinger(near_lower, ranging=True).score == pytest.approx(0.3)
    assert score_bollinger(near_lower, ranging=False).score == 0.0
    assert score_bollinger(bb_frame(109), ranging=True).score == pytest.approx(-0.3)
    breakdown = score_bollinger(bb_frame(88, volume_ratio=2.0))
    assert breakdown.score < -0.4 and "bb_breakdown" in breakdown.tags


def test_squeeze_breakout_dari_data_sintetis():
    rng = np.random.default_rng(5)
    calm = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
    squeeze = calm[-1] * np.exp(np.cumsum(rng.normal(0, 0.0005, 60)))
    closes = np.r_[calm, squeeze, squeeze[-1] * 1.03]
    volumes = np.r_[np.full(260, 1000.0), 5000.0]
    frame = with_indicators(make_ohlcv(closes, volumes=volumes))
    signal = score_bollinger(frame)
    assert "bb_squeeze_breakout" in signal.tags
    assert signal.score == pytest.approx(0.8)


# ----------------------------------------------------------------------
# Volume dan OBV
# ----------------------------------------------------------------------
def volume_frame(ratio, bullish=True, obv_above=True):
    close, open_ = (101.0, 100.0) if bullish else (99.0, 100.0)
    return indicator_frame({
        "open": [100.0, open_], "close": [100.0, close], "volume_ratio": [1.0, ratio],
        "obv": [1.0, 2.0 if obv_above else 0.0], "obv_ema": [1.0, 1.0],
    })


def test_lonjakan_volume_beli_dan_jual():
    buy = score_volume(volume_frame(2.5))
    assert buy.score == pytest.approx(0.8)
    assert {"volume_spike", "obv_naik"} <= set(buy.tags)
    assert buy.label == "Volume 2.5x"
    sell = score_volume(volume_frame(2.5, bullish=False, obv_above=False))
    assert sell.score == pytest.approx(-0.8)


def test_volume_lemah_ditandai():
    signal = score_volume(volume_frame(0.5))
    assert "volume_lemah" in signal.tags
    assert signal.score == pytest.approx(0.3)  # hanya kontribusi OBV


# ----------------------------------------------------------------------
# Trigger candle
# ----------------------------------------------------------------------
def candle_frame(prev, curr):
    return indicator_frame({k: [prev[i], curr[i]] for i, k in enumerate(("open", "high", "low", "close"))})


@pytest.mark.parametrize(
    ("prev", "curr", "tag", "score"),
    [
        ((10.0, 10.1, 9.4, 9.5), (9.4, 10.3, 9.3, 10.2), "bullish_engulfing", 0.7),
        ((9.5, 10.1, 9.4, 10.0), (10.1, 10.2, 9.3, 9.4), "bearish_engulfing", -0.7),
        ((10.0, 10.1, 9.8, 9.9), (9.9, 10.0, 9.0, 9.98), "hammer", 0.5),
        ((10.0, 10.1, 9.8, 9.9), (9.9, 11.0, 9.85, 9.88), "shooting_star", -0.5),
        ((10.0, 10.2, 9.9, 10.1), (10.1, 10.9, 10.05, 10.85), "breakout_candle", 0.5),
    ],
)
def test_pola_candle(prev, curr, tag, score):
    signal = score_trigger_candle(candle_frame(prev, curr))
    assert tag in signal.tags
    assert signal.score == pytest.approx(score)


def test_candle_tanpa_pola_dan_tanpa_gerak():
    assert score_trigger_candle(candle_frame((10, 10.5, 9.5, 10.1), (10.1, 10.6, 9.9, 10.3))).score == pytest.approx(0.1)
    assert score_trigger_candle(candle_frame((10, 10, 10, 10), (10, 10, 10, 10))).score == 0.0


# ----------------------------------------------------------------------
# Struktur pasar
# ----------------------------------------------------------------------
def swing_frame(points, tail_end, steps=6, tail_steps=4):
    """Harga zigzag melalui titik swing, lalu bergerak ke `tail_end` tanpa membentuk swing baru."""
    path = []
    for start, end in zip(points[:-1], points[1:], strict=True):
        path.extend(np.linspace(start, end, steps, endpoint=False))
    path.extend(np.linspace(points[-1], tail_end, tail_steps + 1))
    path = np.asarray(path)
    return make_ohlcv(path, highs=path + 0.1, lows=path - 0.1)


def test_struktur_higher_high_higher_low():
    signal = score_structure(swing_frame([100, 110, 104, 114, 108], tail_end=111))
    assert "hh_hl" in signal.tags
    assert signal.score == pytest.approx(0.6)
    assert signal.label == "HH + HL"


def test_struktur_lower_high_lower_low_dan_break_of_structure():
    down = score_structure(swing_frame([120, 110, 116, 106, 112, 108], tail_end=107))
    assert "lh_ll" in down.tags and down.score == pytest.approx(-0.6)
    breakout = score_structure(swing_frame([100, 110, 104, 114, 108], tail_end=115))
    assert {"hh_hl", "bos_up"} <= set(breakout.tags)
    assert breakout.score == pytest.approx(1.0)


def test_struktur_swing_kurang():
    assert score_structure(make_ohlcv(random_walk(5))).score == 0.0
