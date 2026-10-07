"""Test CLI main.py: gerbang mode live, perintah yang belum tersedia, perintah Fase 1 sampai 4."""

from __future__ import annotations

import re

import ccxt
import pandas as pd
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
    for command, phase in (("train", 5), ("paper", 6)):
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


# ----------------------------------------------------------------------
# Fase 4: backtest dan report
# ----------------------------------------------------------------------
def make_history(tmp_path, symbols=("BTC/USDT", "AAA/USDT", "BBB/USDT")):
    from helpers import sol_rules
    from synthetic import synthetic_market

    from backtest.data import HistoryStore

    store = HistoryStore(tmp_path / "data" / "history")
    market = synthetic_market(list(symbols), days=16, start="2025-03-01", seed=5)
    for symbol, frames in market.items():
        for tf, frame in frames.items():
            store.save(symbol, tf, frame)
    store.save_rules({s: sol_rules(symbol=s, base=s.split("/")[0], step_size=1e-6, tick_size=1e-8) for s in symbols})
    store.save_candidates([s for s in symbols if s != "BTC/USDT"], "uji")
    return store


def test_report_tanpa_hasil_backtest(tmp_path, capsys):
    assert run_cli(tmp_path, ["report"]) == 1
    assert "Belum ada hasil backtest" in capsys.readouterr().err


def test_backtest_offline_tanpa_data(tmp_path, capsys):
    assert run_cli(tmp_path, ["backtest", "--offline"]) == 1
    assert "Belum ada data historis" in capsys.readouterr().err


def test_backtest_offline_lalu_report(tmp_path, capsys):
    make_history(tmp_path)
    args = ["backtest", "--offline", "--start", "2025-03-13", "--end", "2025-03-15", "--capital", "1000", "--workers", "1"]
    assert run_cli(tmp_path, args) == 0
    out = capsys.readouterr().out
    assert "HASIL BACKTEST" in out and "2025-03-13 00:00 s/d 2025-03-15 00:00" in out
    runs = list((tmp_path / "data" / "backtests").iterdir())
    assert len(runs) == 1 and (runs[0] / "trades.csv").exists()
    assert run_cli(tmp_path, ["report"]) == 0
    assert "HASIL BACKTEST" in capsys.readouterr().out


def test_backtest_tanggal_tidak_valid(tmp_path, capsys):
    make_history(tmp_path)
    assert run_cli(tmp_path, ["backtest", "--offline", "--start", "2025-03-15", "--end", "2025-03-13"]) == 1
    assert "tanggal mulai harus sebelum tanggal akhir" in capsys.readouterr().err


async def test_backtest_mengunduh_data_lewat_api_mock(tmp_path, settings, mock_exchange, capsys):
    import argparse

    from helpers import T0, FakeClock, FakeKlineServer

    from backtest.data import HistoryStore, history_dir

    markets = {s: make_market(s.split("/")[0]) for s in ("AAA/USDT", "BTC/USDT")}
    install_markets(mock_exchange, markets)
    mock_exchange.fetch_ohlcv.side_effect = FakeKlineServer(FakeClock(T0)).fetch_ohlcv
    client = ExchangeClient(settings, mock_exchange, jitter=0)
    end = pd.Timestamp(T0, unit="ms", tz="UTC").floor("D")
    args = argparse.Namespace(
        start=str((end - pd.Timedelta(days=1)).date()), end=str(end.date()), months=6, capital=500.0,
        symbols="AAA/USDT", candidates=None, workers=1, oos_fraction=1 / 3, offline=False,
    )
    assert await main.run_backtest(settings, args, client=client) == 0
    out = capsys.readouterr().out
    assert "Data historis: 2 coin" in out and "HASIL BACKTEST" in out
    store = HistoryStore(history_dir(settings))
    assert store.has("AAA/USDT", "5m") and store.has("BTC/USDT", "1h") and "AAA/USDT" in store.load_rules()
