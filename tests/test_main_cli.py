"""Test CLI main.py: gerbang mode live, perintah yang belum tersedia, dan perintah Fase 1."""

from __future__ import annotations

from helpers import (
    T0,
    FakeClock,
    FakeKlineServer,
    install_markets,
    make_market,
    make_ticker,
)

import main
from core.exchange import ExchangeClient

FAKE_KEYS = {"BINANCE_API_KEY": "kunci-palsu-123", "BINANCE_API_SECRET": "rahasia-palsu-456"}


def env_for(tmp_path, **values):
    environ = {"DATA_DIR": str(tmp_path / "data"), "LOGS_DIR": str(tmp_path / "logs")}
    environ.update(values)
    return environ


def run_cli(tmp_path, args, **env_values):
    missing_env = str(tmp_path / "tidak_ada.env")
    return main.main(["--env-file", missing_env, *args], environ=env_for(tmp_path, **env_values))


def test_live_ditolak_pada_mode_paper(tmp_path, capsys):
    code = run_cli(tmp_path, ["live", "--confirm-live"], **FAKE_KEYS)
    assert code == 1
    assert "TRADING_MODE" in capsys.readouterr().err


def test_live_ditolak_tanpa_flag_confirm_live(tmp_path, capsys):
    code = run_cli(tmp_path, ["live"], TRADING_MODE="live", **FAKE_KEYS)
    assert code == 1
    assert "--confirm-live" in capsys.readouterr().err


def test_live_lolos_gerbang_tetapi_belum_tersedia(tmp_path, capsys):
    code = run_cli(tmp_path, ["live", "--confirm-live"], TRADING_MODE="live", **FAKE_KEYS)
    assert code == 2
    assert "Fase 7" in capsys.readouterr().out


def test_perintah_fase_berikutnya_belum_tersedia(tmp_path, capsys):
    for command, phase in (("backtest", 4), ("train", 5), ("paper", 6), ("report", 4)):
        assert run_cli(tmp_path, [command]) == 2
        assert f"Fase {phase}" in capsys.readouterr().out


def test_konfigurasi_tidak_valid(tmp_path, capsys):
    env = tmp_path / ".env"
    env.write_text("RISK_PER_TRADE=0.5\n", encoding="utf-8")
    code = main.main(["--env-file", str(env), "paper"], environ=env_for(tmp_path))
    assert code == 1
    assert "tidak valid" in capsys.readouterr().err


async def test_run_universe_mencetak_daftar_coin(settings, mock_exchange, capsys):
    markets = {m["symbol"]: m for m in (make_market(f"C{i:02d}") for i in range(80))}
    install_markets(mock_exchange, markets)
    mock_exchange.fetch_tickers.return_value = {
        symbol: make_ticker(symbol, 1e9 - i * 1e6) for i, symbol in enumerate(sorted(markets))
    }
    mock_exchange.fetch.return_value = {"data": []}
    client = ExchangeClient(settings, mock_exchange, jitter=0)

    assert await main.run_universe(settings, client, refresh=True, top=3) == 0
    out = capsys.readouterr().out
    assert "Universe: 75 coin" in out
    assert "C00/USDT" in out and "C02/USDT" in out and "C03/USDT" not in out


async def test_run_fetch_mencetak_semua_timeframe(settings, mock_exchange, capsys):
    clock = FakeClock(T0 + 5_000)
    server = FakeKlineServer(clock)
    install_markets(mock_exchange, {"SOL/USDT": make_market("SOL")})
    mock_exchange.fetch_ohlcv.side_effect = server.fetch_ohlcv
    mock_exchange.fetch_time.return_value = T0 + 5_000
    client = ExchangeClient(settings, mock_exchange, jitter=0, wall_clock_ms=clock)

    assert await main.run_fetch(settings, client, "sol/usdt") == 0
    out = capsys.readouterr().out
    for timeframe in ("5m", "15m", "30m", "1h"):
        assert f"  {timeframe:>4} |  300 candle" in out


async def test_run_fetch_simbol_tidak_ada(settings, mock_exchange, capsys):
    install_markets(mock_exchange, {"SOL/USDT": make_market("SOL")})
    client = ExchangeClient(settings, mock_exchange, jitter=0)
    assert await main.run_fetch(settings, client, "XYZ/USDT") == 1
    assert "tidak ditemukan" in capsys.readouterr().err
