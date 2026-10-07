"""Test confluence engine: skor 0..100, gerbang entry, bobot, cache, dan paritas backtest/live."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from helpers import make_settings
from synthetic import multi_timeframe, random_ohlcv

from analysis.confluence import (
    SETUP_TAGS,
    ConfluenceEngine,
    EngineParams,
    describe_signal,
    format_signal,
    prepare_frames,
    rank_signals,
)
from analysis.indicators import compute_indicators
from analysis.regime import TRENDING_DOWN, MarketContext, Regime
from analysis.weights import default_weight_values
from config.settings import TIMEFRAME_MS

TIMEFRAMES = ("5m", "15m", "30m", "1h")


def build_market(drift: float, seed: int, n: int = 30_000) -> dict[str, pd.DataFrame]:
    """Indikator untuk riwayat penuh tiap timeframe (seperti backtest)."""
    raw = multi_timeframe(random_ohlcv(n, drift=drift, vol=0.0025, seed=seed, freq="5m"), window=None)
    return {tf: compute_indicators(frame) for tf, frame in raw.items()}


def views_at(frames: dict[str, pd.DataFrame], now: pd.Timestamp, window: int | None = None) -> dict[str, pd.DataFrame]:
    """Candle yang SUDAH tertutup pada waktu `now` (tanpa look ahead)."""
    views = {}
    for tf, frame in frames.items():
        closed = frame[frame.index + pd.Timedelta(milliseconds=TIMEFRAME_MS[tf]) <= now]
        views[tf] = closed.iloc[-window:] if window else closed
    return views


def eval_times(frames, start=12_000, step=48, count=120):
    index = frames["5m"].index
    return [index[i] + pd.Timedelta(minutes=5) for i in range(start, min(start + step * count, len(index)), step)]


@pytest.fixture(scope="module")
def uptrend():
    return build_market(drift=0.00015, seed=3)


@pytest.fixture(scope="module")
def downtrend():
    return build_market(drift=-0.00015, seed=4)


@pytest.fixture(scope="module")
def uptrend_results(uptrend):
    engine = ConfluenceEngine()
    return [engine.evaluate("UP/USDT", views_at(uptrend, now)) for now in eval_times(uptrend)]


def test_uptrend_menghasilkan_entry_yang_konsisten(uptrend_results):
    entries = [r for r in uptrend_results if r.is_entry]
    assert entries, "uptrend sintetis seharusnya menghasilkan minimal satu entry"
    for result in entries:
        assert result.score >= 65
        assert result.aligned >= 3
        assert result.regime.allows_entry
        plan = result.plan
        assert plan.stop < plan.entry < plan.tp1
        assert 1.5 * plan.atr - 1e-9 <= plan.entry - plan.stop <= 2.5 * plan.atr + 1e-9
        assert plan.reward_risk >= 1.5
        assert result.reasons
        assert result.summary == " + ".join(result.reasons)
        assert result.pattern
        assert all("@" in tag for tag in result.tags)


def test_downtrend_ditolak_tren_1h_bearish(downtrend):
    engine = ConfluenceEngine()
    results = [engine.evaluate("DOWN/USDT", views_at(downtrend, now)) for now in eval_times(downtrend, count=60)]
    bearish = [r for r in results if {"tren_1h_bearish", "regime_turun"} & set(r.rejections)]
    assert len(bearish) >= 0.8 * len(results)
    assert sum(r.is_entry for r in results) <= 0.05 * len(results)


def test_skor_dan_alasan_berbentuk_benar(uptrend_results):
    for result in uptrend_results:
        assert 0 <= result.score <= 100
        assert result.score == pytest.approx(min(max(50 * (1 + result.raw_score - result.penalty), 0), 100), abs=0.01)
        assert set(result.timeframes) == set(TIMEFRAMES)
        assert len(result.reasons) <= 4
        assert result.aligned == sum(ts.aligned for ts in result.timeframes.values())


def test_snapshot_fitur_lengkap_dan_bebas_skala(uptrend_results):
    features = uptrend_results[-1].features
    for tf in TIMEFRAMES:
        for key in ("score", "rsi", "adx", "volume_ratio", "bb_width_rank"):
            assert f"{tf}.{key}" in features
    for key in ("dist_support_atr", "dist_resistance_atr", "fib_retracement", "reward_risk", "btc_change_1h", "hour", "weekday"):
        assert key in features
    assert features["reward_risk"] <= 10
    assert all(isinstance(value, float) for value in features.values())


def test_data_kurang_ditolak():
    engine = ConfluenceEngine()
    short = {tf: frame.iloc[:30] for tf, frame in prepare_frames(multi_timeframe(random_ohlcv(400, freq="5m"))).items()}
    result = engine.evaluate("X/USDT", short)
    assert result.rejections == ("data_kurang",)
    assert result.score == 0.0 and result.plan is None
    missing = engine.evaluate("X/USDT", {"5m": short["5m"]})
    assert missing.rejections == ("data_kurang",)


def test_circuit_breaker_dan_regime_btc(uptrend):
    engine = ConfluenceEngine()
    views = views_at(uptrend, eval_times(uptrend)[-1])
    base = engine.evaluate("X/USDT", views)
    breaker = engine.evaluate("X/USDT", views, MarketContext(circuit_breaker_active=True))
    assert "circuit_breaker" in breaker.rejections
    btc_down = Regime(TRENDING_DOWN, False, 35.0, 0.5, "uji")
    bearish_market = engine.evaluate("X/USDT", views, MarketContext(btc_regime=btc_down))
    # Penalti default BTC Trending Down = 0.2 pada skala -1..1 = 10 poin.
    assert bearish_market.penalty == pytest.approx(base.penalty + 0.2)
    assert bearish_market.score == pytest.approx(max(base.score - 10, 0), abs=0.01)


def test_bobot_dari_database_mempengaruhi_skor(uptrend):
    views = views_at(uptrend, eval_times(uptrend)[-1])
    only_trend = default_weight_values() | {"tf.structure": 0.0, "tf.setup": 0.0, "tf.trigger": 0.0}
    result = ConfluenceEngine(only_trend).evaluate("X/USDT", views)
    assert result.raw_score == pytest.approx(result.timeframes["1h"].score, abs=1e-6)

    penalised = default_weight_values() | {f"penalty.{name}": 0.1 for name in ("melawan_tren_1h", "dekat_resistance", "volume_lemah", "entry_terlambat", "btc_dump")}
    base = ConfluenceEngine().evaluate("X/USDT", views)
    stricter = ConfluenceEngine(penalised).evaluate("X/USDT", views)
    assert stricter.penalty == pytest.approx(base.penalty + 0.1 * len(base.conditions))


def test_cache_tidak_mengubah_hasil(uptrend):
    cached, uncached = ConfluenceEngine(cache=True), ConfluenceEngine(cache=False)
    for now in eval_times(uptrend, step=1, count=40):  # tiap 5 menit: 1h/30m/15m sering diambil dari cache
        views = views_at(uptrend, now)
        a, b = cached.evaluate("X/USDT", views), uncached.evaluate("X/USDT", views)
        assert (a.score, a.rejections, a.reasons, a.aligned) == (b.score, b.rejections, b.reasons, b.aligned)
        assert a.plan == b.plan


def test_paritas_backtest_dan_live(uptrend):
    """Indikator riwayat penuh yang dipotong (backtest) = indikator dari data s.d. t (live)."""
    raw = multi_timeframe(random_ohlcv(30_000, drift=0.00015, vol=0.0025, seed=3, freq="5m"), window=None)
    engine = ConfluenceEngine(cache=False)
    for now in eval_times(uptrend, step=400, count=6):
        backtest = engine.evaluate("X/USDT", views_at(uptrend, now))
        live_full = engine.evaluate("X/USDT", prepare_frames(views_at(raw, now)))
        live_window = engine.evaluate("X/USDT", prepare_frames(views_at(raw, now, window=1000)))
        assert live_full.score == pytest.approx(backtest.score, abs=1e-6)
        assert live_full.rejections == backtest.rejections
        # Jendela 1000 candle (live): EMA sedikit berbeda, skor tetap hampir sama.
        assert live_window.score == pytest.approx(backtest.score, abs=1.0)


def test_engine_params_dari_settings(tmp_path):
    settings = make_settings(tmp_path, min_signal_score=70, min_aligned_timeframes=4, min_reward_risk=2.0)
    params = EngineParams.from_settings(settings)
    assert params.roles == {"5m": "trigger", "15m": "setup", "30m": "structure", "1h": "trend"}
    assert (params.min_score, params.min_aligned, params.min_reward_risk) == (70, 4, 2.0)
    with pytest.raises(ValueError, match="4 timeframe"):
        EngineParams.from_settings(make_settings(tmp_path, timeframes="5m,1h"))


def test_format_dan_peringkat_sinyal(uptrend_results):
    ranked = rank_signals(uptrend_results)
    entries = [r for r in ranked if r.is_entry]
    assert ranked[: len(entries)] == entries  # entry valid selalu di atas
    assert [r.score for r in entries] == sorted((r.score for r in entries), reverse=True)
    line = format_signal(entries[0])
    assert "UP/USDT" in line and "ENTRY" in line and f"{entries[0].score:5.1f}" in line
    report = describe_signal(entries[0])
    for text in ("skor", "Alasan utama", "Timeframe searah", "Rencana: entry", "1h (trend)", "5m (trigger)"):
        assert text in report
    rejected = next(r for r in ranked if not r.is_entry)
    assert "tolak:" in format_signal(rejected)
    assert math.isfinite(rejected.score)


def test_tag_setup_wajib_sesuai_regime(uptrend):
    engine = ConfluenceEngine()
    for now in eval_times(uptrend, count=60):
        result = engine.evaluate("X/USDT", views_at(uptrend, now))
        if result.is_entry:
            tags = {tag.split("@")[0] for tag in result.tags if not tag.endswith("@5m")}
            assert tags & SETUP_TAGS[result.regime.trend]


def test_nilai_numerik_aman(uptrend_results):
    for result in uptrend_results:
        assert np.isfinite(result.raw_score) and np.isfinite(result.penalty)
