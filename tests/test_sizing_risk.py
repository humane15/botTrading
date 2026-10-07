"""Test position sizing (risiko, slot, saldo, min notional, anti martingale) dan risk manager."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from helpers import make_settings, sol_rules

from core.database import connect
from risk.position_sizing import (
    calculate_position_size,
    effective_max_positions,
    loss_per_unit,
)
from risk.positions import OPEN, Position, PositionStore
from risk.risk_manager import RiskManager, btc_correlation, period_keys


def size(**overrides):
    values = dict(
        equity=10_000.0, free_quote=10_000.0, entry=100.0, stop=90.0, rules=sol_rules(), risk_per_trade=0.01,
        max_positions=5, fee_rate=0.0, slippage=0.0, entry_buffer=0.0,
    )
    values.update(overrides)
    return calculate_position_size(**values)


# ----------------------------------------------------------------------
# Position sizing
# ----------------------------------------------------------------------
def test_ukuran_berdasarkan_risiko_dan_jarak_stop():
    result = size()  # risiko $100, rugi per coin $10
    assert result.ok and result.limited_by == "risiko"
    assert result.qty == pytest.approx(10.0)
    assert result.risk_amount == pytest.approx(100.0) and result.risk_pct == pytest.approx(0.01)


def test_fee_dan_slippage_masuk_hitungan_risiko():
    result = size(fee_rate=0.001, slippage=0.0005)
    per_unit = loss_per_unit(100, 90, 0.001, 0.0005)
    assert per_unit > 10
    assert result.qty == pytest.approx(sol_rules().qty_down(100 / per_unit))
    assert result.risk_amount <= 100.0  # rugi nyata saat stop tidak melebihi 1% modal


def test_stop_ketat_dibatasi_porsi_slot():
    result = size(stop=99.0)  # risiko ingin 100 coin ($10.000), slot hanya $2.000
    assert result.limited_by == "porsi_slot"
    assert result.notional == pytest.approx(2_000.0)
    assert result.risk_pct < 0.01


def test_dibatasi_saldo_usdt_tersedia():
    result = size(free_quote=300.0)
    assert result.limited_by == "saldo"
    assert result.notional <= 300.0


def test_slot_posisi_berkurang_saat_modal_kecil():
    assert effective_max_positions(10_000, 5, 11) == 5
    assert effective_max_positions(30, 5, 11) == 2
    assert effective_max_positions(10, 5, 11) == 0
    small = size(equity=30.0, free_quote=30.0, stop=99.0)
    assert small.max_positions == 2 and small.notional <= 15.0 + 1e-9
    tiny = size(equity=10.0, free_quote=10.0)
    assert not tiny.ok and "terlalu kecil" in tiny.reason


def test_ditolak_jika_bagian_exit_di_bawah_min_notional():
    result = size(equity=100.0, free_quote=100.0, stop=60.0)  # risiko $1 / $40 per coin = 0.025 coin
    assert not result.ok
    assert "terlalu kecil" in result.reason and result.qty == 0


def test_volatilitas_tinggi_memotong_setengah_dan_anti_martingale():
    normal = size()
    half = size(size_multiplier=0.5)
    assert half.qty == pytest.approx(normal.qty / 2)
    assert size(size_multiplier=2.0).qty == normal.qty  # pengali > 1 diabaikan (tidak memperbesar)
    after_loss = size(equity=9_000.0, free_quote=9_000.0)
    assert after_loss.qty < normal.qty  # modal turun setelah rugi -> posisi mengecil, bukan membesar


def test_stop_tidak_valid_dan_pembulatan_step():
    assert size(stop=101.0).reason == "stop loss harus di bawah harga entry"
    result = size(entry=33.0, stop=30.0)
    assert result.qty == sol_rules().qty_down(result.qty)


# ----------------------------------------------------------------------
# Risk manager
# ----------------------------------------------------------------------
class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def store():
    return PositionStore(connect(":memory:"))


def manager(tmp_path, store, clock=None, **settings_overrides):
    settings = make_settings(tmp_path, **settings_overrides)
    return RiskManager(settings, store, "paper", kill_file=tmp_path / "STOP", clock=clock or Clock(datetime(2026, 10, 5, 9, tzinfo=timezone.utc)))


def open_position(symbol, btc_corr=None):
    return Position(symbol=symbol, mode="paper", status=OPEN, qty=1.0, btc_corr=btc_corr)


def test_kill_switch_file_stop(tmp_path, store):
    risk = manager(tmp_path, store)
    assert risk.check_entry("SOL/USDT", equity=1000, open_positions=[]).allowed
    (tmp_path / "STOP").write_text("")
    decision = risk.check_entry("SOL/USDT", equity=1000, open_positions=[])
    assert not decision.allowed and decision.reasons == ("kill_switch",)
    assert "STOP" in decision.text()


def test_batas_rugi_harian_dan_reset_hari_berikutnya(tmp_path, store):
    clock = Clock(datetime(2026, 10, 5, 1, tzinfo=timezone.utc))
    risk = manager(tmp_path, store, clock)
    assert risk.check_entry("SOL/USDT", equity=1000, open_positions=[]).allowed  # awal hari: 1000
    blocked = risk.check_entry("SOL/USDT", equity=949, open_positions=[])
    assert blocked.reasons == ("batas_rugi_harian",)
    clock.now += timedelta(days=1)
    assert risk.check_entry("SOL/USDT", equity=949, open_positions=[]).allowed  # hari baru


def test_batas_rugi_mingguan(tmp_path, store):
    clock = Clock(datetime(2026, 10, 5, 1, tzinfo=timezone.utc))  # senin
    risk = manager(tmp_path, store, clock)
    risk.check_entry("SOL/USDT", equity=1000, open_positions=[])
    clock.now += timedelta(days=1)
    risk.check_entry("SOL/USDT", equity=930, open_positions=[])  # awal selasa
    decision = risk.check_entry("SOL/USDT", equity=899, open_positions=[])
    assert decision.reasons == ("batas_rugi_mingguan",)  # harian -3.3% masih aman, mingguan -10.1%
    clock.now += timedelta(days=6)  # senin berikutnya
    assert risk.check_entry("SOL/USDT", equity=899, open_positions=[]).allowed


def test_slot_penuh_coin_sama_dan_diblokir(tmp_path, store):
    risk = manager(tmp_path, store)
    five = [open_position(f"C{i}/USDT") for i in range(5)]
    assert "posisi_penuh" in risk.check_entry("SOL/USDT", equity=10_000, open_positions=five).reasons
    small_account = manager(tmp_path, PositionStore(connect(":memory:")))  # modal $25: hanya 2 slot
    one_held = small_account.check_entry("SOL/USDT", equity=25, open_positions=[open_position("C0/USDT")])
    assert one_held.allowed and one_held.max_positions == 2
    two_held = [open_position("C0/USDT"), open_position("C1/USDT")]
    assert small_account.check_entry("SOL/USDT", equity=25, open_positions=two_held).reasons == ("posisi_penuh",)
    duplicate = risk.check_entry("SOL/USDT", equity=10_000, open_positions=[open_position("SOL/USDT")])
    assert duplicate.reasons == ("sudah_ada_posisi",)
    risk.block_symbols(["XRP/USDT"])
    assert risk.check_entry("XRP/USDT", equity=10_000, open_positions=[]).reasons == ("simbol_diblokir",)


def test_maksimal_dua_posisi_berkorelasi_tinggi_dengan_btc(tmp_path, store):
    risk = manager(tmp_path, store)
    held = [open_position("ETH/USDT", 0.92), open_position("SOL/USDT", 0.85), open_position("DOGE/USDT", 0.4)]
    assert risk.check_entry("AVAX/USDT", equity=10_000, open_positions=held, btc_corr=0.88).reasons == ("korelasi_btc_penuh",)
    assert risk.check_entry("TRX/USDT", equity=10_000, open_positions=held, btc_corr=0.5).allowed
    assert risk.check_entry("NEW/USDT", equity=10_000, open_positions=held, btc_corr=float("nan")).allowed


def test_korelasi_return_terhadap_btc():
    rng = np.random.default_rng(1)
    index = pd.date_range("2025-01-01", periods=300, freq="1h", tz="UTC")
    btc = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=index)
    follower = btc * np.exp(rng.normal(0, 0.001, 300))
    independent = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=index)
    assert btc_correlation(follower, btc) > 0.9
    assert abs(btc_correlation(independent, btc)) < 0.3
    assert np.isnan(btc_correlation(btc.iloc[:20], btc.iloc[:20]))


def test_kunci_periode_utc():
    assert period_keys(datetime(2026, 10, 7, 23, tzinfo=timezone.utc)) == ("day:2026-10-07", "week:2026-W41")
