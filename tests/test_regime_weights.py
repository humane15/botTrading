"""Test market regime, circuit breaker BTC, dan penyimpanan bobot di SQLite."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from synthetic import indicator_frame, make_ohlcv, random_ohlcv, with_indicators

from analysis.regime import (
    RANGING,
    TRENDING_DOWN,
    TRENDING_UP,
    CircuitBreaker,
    analyze_market,
    classify_regime,
)
from analysis.weights import DEFAULT_WEIGHTS, WeightStore, default_weight_values
from config.logging_setup import LEARNING_LOGGER
from core.database import connect


# ----------------------------------------------------------------------
# Regime
# ----------------------------------------------------------------------
def regime_frame(adx, *, e20, e50, e200, close=None, plus_di=30.0, minus_di=15.0, atr_rank=0.5):
    close = e20 + 1 if close is None else close
    return indicator_frame({
        "adx": [adx], "plus_di": [plus_di], "minus_di": [minus_di], "close": [close],
        "ema20": [e20], "ema50": [e50], "ema200": [e200], "atr_pct_rank": [atr_rank],
    })


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        (regime_frame(30, e20=105, e50=103, e200=100), TRENDING_UP),
        (regime_frame(30, e20=95, e50=97, e200=100, close=94, plus_di=12, minus_di=30), TRENDING_DOWN),
        (regime_frame(15, e20=105, e50=103, e200=100), RANGING),
        (regime_frame(22, e20=95, e50=97, e200=100, close=94), TRENDING_DOWN),   # transisi tapi tersusun turun
        (regime_frame(22, e20=105, e50=103, e200=100), RANGING),                 # transisi, uptrend belum terkonfirmasi
        (regime_frame(30, e20=105, e50=103, e200=104, close=102, plus_di=15, minus_di=30), TRENDING_DOWN),
    ],
)
def test_klasifikasi_regime(frame, expected):
    assert classify_regime(frame).trend == expected


def test_regime_trending_down_melarang_entry_dan_high_vol_memotong_posisi():
    down = classify_regime(regime_frame(30, e20=95, e50=97, e200=100, close=94, plus_di=12, minus_di=30))
    assert not down.allows_entry and down.strategy == "tidak_entry" and down.code() == -1
    volatile = classify_regime(regime_frame(30, e20=105, e50=103, e200=100, atr_rank=0.95))
    assert volatile.high_volatility
    assert volatile.size_multiplier == 0.5
    assert volatile.label == "Trending Up + High Volatility"
    assert volatile.strategy == "pullback"
    calm = classify_regime(regime_frame(15, e20=105, e50=103, e200=100))
    assert calm.size_multiplier == 1.0 and calm.strategy == "range"


def test_regime_dari_data_sintetis():
    assert classify_regime(with_indicators(random_ohlcv(600, drift=0.004, vol=0.006, seed=1))).trend == TRENDING_UP
    assert classify_regime(with_indicators(random_ohlcv(600, drift=-0.004, vol=0.006, seed=2))).trend == TRENDING_DOWN
    assert classify_regime(pd.DataFrame()).trend == RANGING


# ----------------------------------------------------------------------
# Circuit breaker
# ----------------------------------------------------------------------
def btc_5m(closes, start="2025-01-01 00:00"):
    return make_ohlcv(closes, freq="5m", start=start)


def test_btc_turun_lebih_dari_3_persen_menghentikan_entry_2_jam():
    closes = [100.0] * 20 + list(np.linspace(100, 96.5, 12))
    btc = btc_5m(closes)
    breaker = CircuitBreaker()
    assert breaker.update(btc) is True
    close_time = btc.index[-1] + pd.Timedelta(minutes=5)
    assert breaker.active_until == close_time + pd.Timedelta(hours=2)
    assert breaker.is_active(close_time + pd.Timedelta(hours=1, minutes=59))
    assert not breaker.is_active(close_time + pd.Timedelta(hours=2, minutes=1))
    assert breaker.last_drop == pytest.approx(0.035)


def test_penurunan_kecil_tidak_memicu_dan_crash_di_tengah_jam_terdeteksi():
    calm = CircuitBreaker()
    assert calm.update(btc_5m([100.0] * 20 + [98.0] * 12)) is False
    # Crash 3.5% lalu memantul: dipicu saat candle crash, tetap aktif setelah pantulan.
    closes = [100.0] * 20 + [99.0, 97.5, 96.5, 97.5, 98.0]
    breaker = CircuitBreaker()
    states = [breaker.update(btc_5m(closes[: i + 1])) for i in range(len(closes))]
    assert states[-1] is True
    assert states.index(True) == 22


def test_analyze_market_menggabungkan_regime_btc_dan_breaker():
    btc_1h = with_indicators(random_ohlcv(600, drift=0.004, vol=0.006, seed=1))
    crash = btc_5m([100.0] * 20 + list(np.linspace(100, 96.0, 12)))
    market = analyze_market({"1h": btc_1h, "5m": crash}, CircuitBreaker())
    assert market.btc_regime.trend == TRENDING_UP
    assert market.circuit_breaker_active
    assert market.btc_drop_1h > 0.03 and market.btc_change_1h < -0.03
    assert market.label == "Trending Up"


# ----------------------------------------------------------------------
# Bobot di database
# ----------------------------------------------------------------------
@pytest.fixture
def store(tmp_path):
    store = WeightStore.open(tmp_path / "bot.db")
    yield store
    store.conn.close()


def test_bobot_default_terisi_di_database(store):
    weights = store.all()
    assert weights == default_weight_values()
    assert {"tf.trend", "tf.trigger", "w.setup.fib", "penalty.dekat_resistance"} <= set(weights)
    assert weights["penalty.dekat_resistance"] == 0.0


def test_perubahan_bobot_maksimal_10_persen_per_siklus(store):
    applied = store.update({"w.setup.fib": 5.0, "tf.trend": 0.0}, reason="uji")
    assert applied["w.setup.fib"] == (1.0, pytest.approx(1.1))
    assert applied["tf.trend"] == (0.3, pytest.approx(0.27))
    # Penalti bernilai 0 naik maksimal 10% dari rentangnya (0..0.5) = 0.05.
    applied = store.update({"penalty.dekat_resistance": 0.4}, reason="12 dari 30 trade rugi")
    assert applied["penalty.dekat_resistance"] == (0.0, pytest.approx(0.05))
    history = store.history("penalty.dekat_resistance")
    assert history[0]["new_value"] == pytest.approx(0.05)
    assert history[0]["reason"] == "12 dari 30 trade rugi"


def test_bobot_dijaga_dalam_batas(store):
    for _ in range(60):
        store.update({"tf.setup": 0.0}, reason="turunkan")
    assert store.get("tf.setup") == pytest.approx(DEFAULT_WEIGHTS["tf.setup"].min)
    assert store.update({"tf.setup": 0.0}, reason="sudah di batas") == {}


def test_perubahan_bobot_dicatat_ke_learning_log(store, caplog):
    with caplog.at_level(logging.INFO, logger=LEARNING_LOGGER):
        store.update({"penalty.volume_lemah": 0.1}, reason="VOLUME_LEMAH 35% kerugian")
    assert "penalty.volume_lemah dinaikkan 0.0000 -> 0.0500" in caplog.text


def test_bobot_tersimpan_permanen_dan_reset(tmp_path):
    path = tmp_path / "bot.db"
    first = WeightStore.open(path)
    first.update({"w.trigger.macd": 2.0}, reason="uji")
    first.conn.close()
    reopened = WeightStore.open(path)
    assert reopened.get("w.trigger.macd") == pytest.approx(1.1)  # nilai lama tidak ditimpa default
    reopened.reset_defaults()
    assert reopened.get("w.trigger.macd") == 1.0
    reopened.conn.close()


def test_bobot_tidak_dikenal_dan_snapshot_tidak_bisa_diubah(store):
    with pytest.raises(KeyError):
        store.update({"tidak.ada": 1.0}, reason="uji")
    with pytest.raises(KeyError):
        store.get("tidak.ada")
    snapshot = store.snapshot()
    with pytest.raises(TypeError):
        snapshot["tf.trend"] = 1.0  # type: ignore[index]


def test_database_memory_dan_wal(tmp_path):
    memory = connect(":memory:")
    assert WeightStore(memory).get("tf.trend") == 0.3
    file_conn = connect(tmp_path / "x" / "bot.db")
    assert file_conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    file_conn.close()
