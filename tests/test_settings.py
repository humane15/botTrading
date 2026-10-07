"""Test konfigurasi: pembacaan .env, validasi, dan gerbang mode live."""

from __future__ import annotations

import logging

import pytest
from dotenv import dotenv_values
from helpers import make_settings
from pydantic import ValidationError

from config.settings import (
    PROJECT_ROOT,
    LiveTradingNotAllowed,
    Settings,
    ensure_live_allowed,
    load_settings,
)

ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
FAKE_KEY = "kunci-palsu-untuk-test-123456"
FAKE_SECRET = "rahasia-palsu-untuk-test-654321"


def write_env(tmp_path, text: str):
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


def test_default_mode_paper_dan_nilai_bawaan(settings):
    assert settings.trading_mode == "paper"
    assert settings.universe_size == 75
    assert settings.min_quote_volume_usd == 5_000_000
    assert settings.timeframes == ("5m", "15m", "30m", "1h")
    assert settings.max_concurrent_requests == 5
    assert settings.has_api_credentials is False


def test_membaca_file_env(tmp_path):
    env = write_env(
        tmp_path,
        "TRADING_MODE=paper\nUNIVERSE_SIZE=50\nMIN_QUOTE_VOLUME_USD=10000000\n"
        "TIMEFRAMES=1h,5m,30m,15m\nEXCLUDED_SYMBOLS=luna, ustc/usdt\nBINANCE_TESTNET=true\n",
    )
    s = load_settings(env, environ={}, data_dir=tmp_path)
    assert s.universe_size == 50
    assert s.min_quote_volume_usd == 10_000_000
    # Timeframe diurutkan dari terkecil ke terbesar.
    assert s.timeframes == ("5m", "15m", "30m", "1h")
    assert s.excluded_symbols == ("LUNA", "USTC/USDT")
    assert s.binance_testnet is True


def test_environment_variable_mengalahkan_file_env(tmp_path):
    env = write_env(tmp_path, "UNIVERSE_SIZE=50\n")
    s = load_settings(env, environ={"UNIVERSE_SIZE": "60"}, data_dir=tmp_path)
    assert s.universe_size == 60


def test_nilai_kosong_memakai_default(tmp_path):
    env = write_env(tmp_path, "BINANCE_API_KEY=\nBINANCE_API_SECRET=\nUNIVERSE_SIZE=\n")
    s = load_settings(env, environ={}, data_dir=tmp_path)
    assert s.binance_api_key is None
    assert s.universe_size == 75


def test_kunci_tidak_dikenal_diberi_peringatan(tmp_path, caplog):
    env = write_env(tmp_path, "TRADNG_MODE=live\n")
    with caplog.at_level(logging.WARNING):
        s = load_settings(env, environ={}, data_dir=tmp_path)
    assert s.trading_mode == "paper"
    assert "TRADNG_MODE" in caplog.text


@pytest.mark.parametrize(
    "overrides",
    [
        {"risk_per_trade": 0.03},       # melebihi batas 2%
        {"risk_per_trade": 0},
        {"timeframes": "5m,2m"},        # timeframe tidak dikenal
        {"timeframes": "5m,5m"},        # duplikat
        {"trading_mode": "yolo"},
        {"daily_loss_limit": 0.2, "weekly_loss_limit": 0.1},
        {"retry_base_delay": 10, "retry_max_delay": 5},
        {"max_concurrent_requests": 0},
        {"binance_api_key": FAKE_KEY},  # secret tidak diisi
    ],
)
def test_validasi_menolak_nilai_salah(tmp_path, overrides):
    with pytest.raises(ValidationError):
        make_settings(tmp_path, **overrides)


def test_api_key_tidak_bocor_di_repr_dan_ringkasan(tmp_path):
    s = make_settings(tmp_path, binance_api_key=FAKE_KEY, binance_api_secret=FAKE_SECRET)
    assert s.has_api_credentials
    for text in (repr(s), str(s), s.summary()):
        assert FAKE_KEY not in text
        assert FAKE_SECRET not in text
    assert set(s.secret_values()) == {FAKE_KEY, FAKE_SECRET}


def test_api_key_dengan_spasi_dibersihkan(tmp_path):
    s = make_settings(tmp_path, binance_api_key=f"  {FAKE_KEY} ", binance_api_secret=FAKE_SECRET)
    assert s.binance_api_key.get_secret_value() == FAKE_KEY


def test_path_relatif_dihitung_dari_root_proyek():
    s = load_settings(env_file=None, environ={"DATA_DIR": "data_test"})
    assert s.data_dir == PROJECT_ROOT / "data_test"
    assert s.universe_file == PROJECT_ROOT / "data_test" / "universe.json"


def test_env_example_lengkap_dan_tanpa_rahasia(tmp_path):
    values = dotenv_values(ENV_EXAMPLE)
    assert {key.lower() for key in values} == set(Settings.model_fields)
    assert values["TRADING_MODE"] == "paper"
    for key in ("BINANCE_API_KEY", "BINANCE_API_SECRET", "TELEGRAM_BOT_TOKEN"):
        assert not values[key], f"{key} di .env.example harus kosong"
    # .env.example harus bisa langsung dipakai sebagai konfigurasi yang valid.
    s = load_settings(ENV_EXAMPLE, environ={}, data_dir=tmp_path)
    assert s.trading_mode == "paper"


# ----------------------------------------------------------------------
# Gerbang mode live
# ----------------------------------------------------------------------
def live_settings(tmp_path, **overrides):
    values = {"trading_mode": "live", "binance_api_key": FAKE_KEY, "binance_api_secret": FAKE_SECRET}
    values.update(overrides)
    return make_settings(tmp_path, **values)


def test_live_diizinkan_jika_semua_syarat_terpenuhi(tmp_path):
    ensure_live_allowed(live_settings(tmp_path), confirm_live=True)


def test_live_ditolak_tanpa_flag_confirm_live(tmp_path):
    with pytest.raises(LiveTradingNotAllowed, match="--confirm-live"):
        ensure_live_allowed(live_settings(tmp_path), confirm_live=False)


def test_live_ditolak_jika_trading_mode_paper(tmp_path):
    s = live_settings(tmp_path, trading_mode="paper")
    with pytest.raises(LiveTradingNotAllowed, match="TRADING_MODE"):
        ensure_live_allowed(s, confirm_live=True)


def test_live_ditolak_tanpa_api_key(tmp_path):
    s = make_settings(tmp_path, trading_mode="live")
    with pytest.raises(LiveTradingNotAllowed, match="BINANCE_API_KEY"):
        ensure_live_allowed(s, confirm_live=True)
