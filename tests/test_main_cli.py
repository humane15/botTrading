"""Test CLI main.py: gerbang mode live, perintah yang belum tersedia, perintah Fase 1 sampai 3."""

from __future__ import annotations

import re

import ccxt
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
from risk.positions import CLOSED, DUST, OPEN, Position, PositionStore

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
        assert f"  {timeframe:>4} |  999 candle" in out  # 1000 diminta, 1 candle berjalan dibuang


async def test_run_fetch_simbol_tidak_ada(settings, mock_exchange, capsys):
    install_markets(mock_exchange, {"SOL/USDT": make_market("SOL")})
    client = ExchangeClient(settings, mock_exchange, jitter=0)
    assert await main.run_fetch(settings, client, "XYZ/USDT") == 1
    assert "tidak ditemukan" in capsys.readouterr().err


def kline_client(settings, mock_exchange, symbols):
    """Client dengan mock exchange yang melayani candle deterministik untuk `symbols`."""
    clock = FakeClock(T0 + 5_000)
    server = FakeKlineServer(clock)
    install_markets(mock_exchange, {s: make_market(s.split("/")[0]) for s in symbols})
    mock_exchange.fetch_ohlcv.side_effect = server.fetch_ohlcv
    mock_exchange.fetch_time.return_value = clock()
    mock_exchange.fetch.return_value = {"data": []}
    mock_exchange.fetch_tickers.return_value = {
        symbol: make_ticker(symbol, 1e9 - i * 1e7) for i, symbol in enumerate(symbols)
    }
    return ExchangeClient(settings, mock_exchange, jitter=0, wall_clock_ms=clock)


async def test_run_analyze_mencetak_laporan_lengkap(settings, mock_exchange, capsys):
    client = kline_client(settings, mock_exchange, ["SOL/USDT", "BTC/USDT"])
    assert await main.run_analyze(settings, client, "sol/usdt") == 0
    out = capsys.readouterr().out
    assert out.startswith("SOL/USDT | skor")
    for text in ("Regime BTC:", "Alasan utama:", "1h (trend)", "30m (structure)", "15m (setup)", "5m (trigger)", "Rencana: entry"):
        assert text in out
    assert settings.db_path.exists()  # bobot dibaca dari database SQLite


async def test_run_analyze_simbol_tidak_ada(settings, mock_exchange, capsys):
    client = kline_client(settings, mock_exchange, ["SOL/USDT", "BTC/USDT"])
    assert await main.run_analyze(settings, client, "XYZ/USDT") == 1
    assert "tidak ditemukan" in capsys.readouterr().err


async def test_run_scan_menampilkan_sinyal_teratas(settings, mock_exchange, capsys):
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT", "AVAX/USDT"]
    client = kline_client(settings, mock_exchange, symbols)
    assert await main.run_scan(settings, client, top=3) == 0
    out = capsys.readouterr().out
    assert "[SCAN] 5 coin | Regime BTC:" in out
    assert "Sinyal teratas:" in out
    ranked_lines = [line for line in out.splitlines() if re.match(r"^\s+\d+\. ", line)]
    assert len(ranked_lines) == 3
    assert all(" skor " in line for line in ranked_lines)
    assert "dari 5 coin yang dianalisis" in out


async def test_run_scan_tetap_jalan_tanpa_data_btc(settings, mock_exchange, capsys):
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    client = kline_client(settings, mock_exchange, symbols)
    server_fetch = mock_exchange.fetch_ohlcv.side_effect

    async def fail_btc(symbol, *args, **kwargs):
        if symbol == "BTC/USDT":
            raise ccxt.BadSymbol("binance simulasi data BTC gagal")
        return await server_fetch(symbol, *args, **kwargs)

    mock_exchange.fetch_ohlcv.side_effect = fail_btc
    assert await main.run_scan(settings, client, top=5) == 0
    captured = capsys.readouterr()
    assert "regime BTC dan circuit breaker tidak diketahui" in captured.err
    assert "Regime BTC: tidak diketahui" in captured.out
    assert "dari 2 coin yang dianalisis" in captured.out


def test_positions_menampilkan_isi_database(tmp_path, capsys):
    store = PositionStore.open(tmp_path / "data" / "bot.db")
    sol = store.insert(Position(symbol="SOL/USDT", mode="paper", status=OPEN, entry_price=100.05, initial_qty=1.998,
                                qty=1.998, stop_price=98, tp1_price=104))
    store.insert(Position(symbol="ETH/USDT", mode="paper", status=CLOSED, realized_pnl=10.42, risk_amount=4.6,
                          exit_reason="trailing_stop", closed_at="2026-10-07T10:15:00+00:00"))
    store.insert(Position(symbol="XRP/USDT", mode="paper", status=CLOSED, realized_pnl=-4.0, risk_amount=4.0,
                          exit_reason="stop_loss", closed_at="2026-10-07T09:00:00+00:00"))
    store.insert(Position(symbol="PEPE/USDT", mode="paper", status=DUST, qty=12.0, exit_reason="cek manual"))
    store.insert(Position(symbol="BNB/USDT", mode="live", status=OPEN, qty=1.0))  # mode lain tidak ditampilkan
    store.conn.close()

    assert run_cli(tmp_path, ["positions", "--limit", "1"]) == 0
    out = capsys.readouterr().out
    assert f"#{sol.id} SOL/USDT | open | entry 100.05 | sisa 1.998 dari 1.998 | stop loss 98 | TP1 104" in out
    assert "Perlu dicek manual (1)" in out and "PEPE/USDT | sisa 12 PEPE" in out
    assert "ETH/USDT | PnL +10.42 USDT (+2.27R) | trailing_stop | ditutup 2026-10-07 10:15 UTC" in out
    assert "XRP/USDT" not in out  # hanya 1 posisi tertutup terakhir yang ditampilkan
    assert "menang 1, kalah 1, win rate 50.0%, total PnL +6.42 USDT" in out
    assert "BNB/USDT" not in out


def test_positions_database_kosong(tmp_path, capsys):
    assert run_cli(tmp_path, ["positions"]) == 0
    out = capsys.readouterr().out
    assert "Posisi aktif: tidak ada" in out and "Belum ada posisi tertutup" in out
