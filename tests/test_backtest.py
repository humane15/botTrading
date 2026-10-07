"""Test backtester: tanpa look ahead, kode sinyal sama dengan live, mekanik simulasi, dan laporan."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from helpers import make_settings, make_signal, sol_rules
from synthetic import random_ohlcv, resample_ohlcv, synthetic_market

from analysis.confluence import ConfluenceEngine, EngineParams, prepare_frames
from analysis.regime import MarketContext
from backtest.engine import BacktestConfig, Backtester, SymbolSeries
from backtest.report import (
    breakdown,
    build_report,
    buy_and_hold,
    compute_metrics,
    drawdown_stats,
    latest_run,
    monthly_returns,
    save_report,
)
from backtest.signals import (
    MemorySource,
    SymbolSignals,
    WindowSlicer,
    generate_signals,
    market_contexts,
    step_times,
    symbol_signals,
)
from backtest.universe import daily_universe
from config.settings import TIMEFRAME_MS
from core.data_feed import frame_open_ms
from risk.positions import CLOSED, Position

START = pd.Timestamp("2025-03-01", tz="UTC")


def multi_tf(frame_5m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "5m": frame_5m,
        "15m": resample_ohlcv(frame_5m, "15min"),
        "30m": resample_ohlcv(frame_5m, "30min"),
        "1h": resample_ohlcv(frame_5m, "1h"),
    }


# ----------------------------------------------------------------------
# Waktu langkah, jendela candle tertutup, universe point in time
# ----------------------------------------------------------------------
def test_langkah_adalah_waktu_tutup_candle_5m():
    steps = step_times(START, START + pd.Timedelta(hours=1))
    assert len(steps) == 12
    assert steps[0] == START.value // 1_000_000 + 300_000  # candle 00:00 tutup pukul 00:05


def test_jendela_hanya_berisi_candle_yang_sudah_tertutup():
    frames = multi_tf(random_ohlcv(3000, freq="5m"))
    steps = step_times(frames["5m"].index[1500], frames["5m"].index[1600])
    slicer = WindowSlicer(frames, steps, window=1000)
    for i in (0, 7, 50):
        now = int(steps[i])
        for tf, frame in slicer.windows(i).items():
            opens = frame_open_ms(frame)
            assert opens[-1] + TIMEFRAME_MS[tf] <= now  # candle terakhir sudah tutup
            assert opens[-1] + 2 * TIMEFRAME_MS[tf] > now  # dan tidak ada candle tertutup yang terlewat
            closed = int((frame_open_ms(frames[tf]) + TIMEFRAME_MS[tf] <= now).sum())
            assert len(frame) == min(1000, closed)
    assert slicer.frame("1h", 0) is slicer.frame("1h", 1)  # irisan 1h dipakai ulang selama candle 1h belum berganti
    assert slicer.closes_at("5m", 0)


def test_universe_harian_memakai_volume_hari_sebelumnya():
    index = pd.date_range("2025-03-01", periods=72, freq="1h", tz="UTC")

    def frame(volumes_per_day):
        volume = np.repeat(volumes_per_day, 24) / 24
        return pd.DataFrame({"close": 1.0, "volume": volume}, index=index)

    frames = {"AAA/USDT": frame([9e6, 1e6, 1e6]), "BBB/USDT": frame([6e6, 8e6, 8e6]), "CCC/USDT": frame([1e6, 7e6, 7e6])}
    days = pd.date_range("2025-03-01", periods=4, freq="D", tz="UTC")
    universe = daily_universe(frames, days, size=2, min_quote_volume=5e6)
    assert universe[days[0]] == ()  # belum ada data hari sebelumnya
    assert universe[days[1]] == ("AAA/USDT", "BBB/USDT")  # volume 1 Maret
    assert universe[days[2]] == ("BBB/USDT", "CCC/USDT")  # AAA turun di bawah 5 juta pada 2 Maret
    assert universe[days[3]] == ("BBB/USDT", "CCC/USDT")


# ----------------------------------------------------------------------
# Sinyal: kode yang sama dengan live dan tanpa data masa depan
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def market():
    return synthetic_market(["BTC/USDT", "AAA/USDT", "BBB/USDT"], days=24, start="2025-03-01", seed=11)


def _window_period(market):
    start = market["BTC/USDT"]["5m"].index[0] + pd.Timedelta(days=14)
    return start.floor("D"), start.floor("D") + pd.Timedelta(days=4)


def test_sinyal_backtest_sama_dengan_evaluasi_live(market, tmp_path):
    settings = make_settings(tmp_path)
    params = EngineParams.from_settings(settings)
    start, end = _window_period(market)
    steps = step_times(start, end)
    contexts = market_contexts(prepare_frames(market["BTC/USDT"]), steps, params, settings)
    raw = market["AAA/USDT"]
    result = symbol_signals("AAA/USDT", raw, steps, np.ones(len(steps), bool), contexts, {}, params)
    assert result.evaluations == len(steps) and result.entries

    # Live: indikator dihitung ulang hanya dari data sampai waktu T, jendela 1000 candle, tanpa cache.
    live = ConfluenceEngine(params=params, cache=False)
    by_time = {int(s.timestamp.value // 1_000_000): s for s in result.entries}
    samples = list(by_time)[:3] + [int(steps[i]) for i in (10, 200, 700)]
    for now in samples:
        i = int(np.searchsorted(steps, now))
        truncated = {tf: f[frame_open_ms(f) + TIMEFRAME_MS[tf] <= now] for tf, f in raw.items()}
        frames = {tf: f.iloc[-1000:] for tf, f in prepare_frames(truncated, params.indicators).items()}
        expected = live.evaluate("AAA/USDT", frames, contexts[i])
        assert expected.is_entry == (now in by_time)
        if expected.is_entry:
            got = by_time[now]
            assert (got.score, got.plan, got.reasons, got.pattern) == (expected.score, expected.plan, expected.reasons, expected.pattern)


def test_data_setelah_waktu_t_tidak_mengubah_sinyal(market, tmp_path):
    settings = make_settings(tmp_path)
    params = EngineParams.from_settings(settings)
    start, end = _window_period(market)
    steps = step_times(start, end)
    contexts = market_contexts(prepare_frames(market["BTC/USDT"]), steps, params, settings)
    cut = int(steps[len(steps) // 2])
    raw = market["AAA/USDT"]
    changed = {}
    for tf, frame in raw.items():
        future = frame_open_ms(frame) + TIMEFRAME_MS[tf] > cut
        frame = frame.copy()
        frame.loc[future, ["open", "high", "low", "close"]] *= 3.0  # masa depan diubah drastis
        frame.loc[future, "volume"] *= 10
        changed[tf] = frame
    active = np.ones(len(steps), bool)
    before = symbol_signals("AAA/USDT", raw, steps, active, contexts, {}, params)
    after = symbol_signals("AAA/USDT", changed, steps, active, contexts, {}, params)

    def upto_cut(result):
        return [(s.timestamp, s.score, s.plan) for s in result.entries if s.timestamp.value // 1_000_000 <= cut]

    assert upto_cut(before) and upto_cut(before) == upto_cut(after)


def test_penolakan_cepat_keputusannya_identik_dengan_evaluasi_penuh(tmp_path):
    raw5 = random_ohlcv(9000, drift=0.00003, vol=0.003, seed=21, freq="5m")
    frames = prepare_frames(multi_tf(raw5))
    steps = step_times(raw5.index[7000], raw5.index[8200])
    slicer = WindowSlicer(frames, steps)
    full, fast = ConfluenceEngine(), ConfluenceEngine()
    entries = skipped = 0
    for i in range(len(steps)):
        window = slicer.windows(i)
        a = full.evaluate("X", window, MarketContext())
        b = fast.evaluate("X", window, MarketContext(), fast_reject=True)
        assert a.is_entry == b.is_entry and set(b.rejections) <= set(a.rejections)
        if a.is_entry:
            entries += 1
            assert (a.score, a.plan, a.reasons) == (b.score, b.plan, b.reasons)
        skipped += "fast_reject" in b.features
    assert entries > 0 and skipped > len(steps) // 4  # sebagian besar penolakan memang dilewati cepat


def test_circuit_breaker_bitcoin_di_backtest(tmp_path):
    settings = make_settings(tmp_path)
    params = EngineParams.from_settings(settings)
    close = np.full(20000, 60_000.0)
    close[15000:] = 57_000.0  # turun 5% dalam satu candle 5m
    frame = pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 10.0},
        index=pd.date_range("2025-01-01", periods=len(close), freq="5min", tz="UTC"),
    )
    steps = step_times(frame.index[14990], frame.index[15100])
    contexts = market_contexts(prepare_frames(multi_tf(frame)), steps, params, settings)
    crash_close = int(frame_open_ms(frame)[15000]) + 300_000
    active = [int(s) for s, c in zip(steps, contexts, strict=True) if c.circuit_breaker_active]
    # Penurunan masih terlihat di jendela 1 jam sampai 55 menit setelah crash, lalu ditambah 2 jam.
    last_seen = crash_close + 55 * 60_000
    assert active == list(range(crash_close, last_seen + 2 * 3_600_000, 300_000))


# ----------------------------------------------------------------------
# Simulasi portofolio: mekanik order dan risiko dengan jam simulasi
# ----------------------------------------------------------------------
def flat_frames(days: float, price: float = 100.0, start: pd.Timestamp = START) -> pd.DataFrame:
    index = pd.date_range(start, periods=int(days * 288), freq="5min", tz="UTC")
    return pd.DataFrame({"open": price, "high": price, "low": price, "close": price, "volume": 1000.0}, index=index)


def build_backtester(tmp_path, frames_5m: dict[str, pd.DataFrame], days: float, **settings_overrides):
    settings = make_settings(tmp_path, **settings_overrides)
    rules = {s: sol_rules(symbol=s, base=s.split("/")[0]) for s in frames_5m}
    config = BacktestConfig(start=START, end=START + pd.Timedelta(days=days), initial_capital=1000.0)
    raw = {s: multi_tf(f) for s, f in frames_5m.items()}
    backtester = Backtester(settings, config, source=MemorySource(raw), rules=rules, symbols=list(frames_5m))
    steps = step_times(config.start, config.end)
    series = {s: SymbolSeries(raw[s], steps, backtester.params) for s in raw}
    return backtester, steps, series


def signal_at(steps, index, symbol="SOL/USDT", **plan):
    return dataclasses.replace(make_signal(symbol, **plan), timestamp=pd.Timestamp(int(steps[index]), unit="ms", tz="UTC"))


async def test_siklus_tp1_trailing_dengan_fee_dan_slippage(tmp_path):
    sol = flat_frames(1)
    k = 20
    sol.iloc[k + 1] = [100.0, 104.5, 99.9, 104.2, 1000.0]   # TP1 (limit maker 104) terisi
    sol.iloc[k + 2] = [104.2, 110.0, 104.1, 109.0, 1000.0]  # trailing: 110 - 2 x ATR 1 = 108
    sol.iloc[k + 3] = [109.0, 109.2, 107.0, 107.5, 1000.0]  # stop 108 tersentuh
    sol.iloc[k + 4:] = [107.5, 107.5, 107.5, 107.5, 1000.0]
    bt, steps, series = build_backtester(tmp_path, {"SOL/USDT": sol, "BTC/USDT": flat_frames(1, 60_000)}, 1)
    signals = {"SOL/USDT": SymbolSignals("SOL/USDT", entries=[signal_at(steps, k)])}
    result = await bt._simulate(steps, series, series["BTC/USDT"], signals)

    (trade,) = result.trades
    assert trade.status == CLOSED and trade.exit_reason == "trailing_stop"
    assert trade.entry_price == pytest.approx(100.05)  # penutupan candle sinyal + slippage 0.05%
    assert pd.Timestamp(trade.opened_at) == pd.Timestamp(int(steps[k]), unit="ms", tz="UTC")  # jam simulasi
    assert pd.Timestamp(trade.closed_at) == pd.Timestamp(int(steps[k + 3]), unit="ms", tz="UTC")
    exits = [e for e in result.events if e.startswith(("[TP1] SOL/USDT 0", "[SELL]"))]
    assert "@ 104 " in exits[0] and "@ 107.946 " in exits[1]  # stop 108 x (1 - slippage)
    dust_cost = trade.dust_qty * trade.cost_per_unit  # sisa di bawah step size tetap di akun
    assert result.equity.iloc[-1] - 1000.0 == pytest.approx(trade.realized_pnl + result.dust_value - dust_cost, abs=1e-9)
    assert result.entry_stats.opened == 1 and result.positions_open.iloc[-1] == 0


async def test_batas_rugi_harian_memakai_waktu_simulasi(tmp_path):
    names = ["AAA/USDT", "BBB/USDT", "CCC/USDT", "DDD/USDT"]
    frames = {name: flat_frames(2) for name in names}
    k = 10
    for name in names[:3]:
        frames[name].iloc[k + 1] = [100.0, 100.0, 79.0, 79.5, 1000.0]  # tiga posisi kena stop 80 bersamaan
        frames[name].iloc[k + 2:] = [79.5, 79.5, 79.5, 79.5, 1000.0]
    frames["BTC/USDT"] = flat_frames(2, 60_000)
    bt, steps, series = build_backtester(tmp_path, frames, 2, risk_per_trade=0.02)
    plan = dict(entry=100.0, stop=80.0, tp1=140.0, atr=5.0)
    entries = {name: [signal_at(steps, k, name, **plan)] for name in names[:3]}
    entries["DDD/USDT"] = [signal_at(steps, k + 5, "DDD/USDT", **plan), signal_at(steps, 288 + 5, "DDD/USDT", **plan)]
    signals = {name: SymbolSignals(name, entries=items) for name, items in entries.items()}
    result = await bt._simulate(steps, series, series["BTC/USDT"], signals)

    losses = [t for t in result.trades if t.symbol != "DDD/USDT"]
    assert len(losses) == 3 and sum(t.realized_pnl for t in losses) < -0.05 * 1000
    assert result.entry_stats.skipped["batas rugi harian tercapai"] == 1  # sinyal DDD di hari yang sama ditolak
    ddd = [t for t in result.trades if t.symbol == "DDD/USDT"]
    assert len(ddd) == 1 and pd.Timestamp(ddd[0].opened_at) >= START + pd.Timedelta(days=1)  # hari berikutnya boleh
    assert ddd[0].exit_reason == "akhir_backtest"  # posisi terbuka ditutup di akhir periode


async def test_backtest_penuh_konsisten(market, tmp_path):
    settings = make_settings(tmp_path)
    symbols = ["AAA/USDT", "BBB/USDT", "BTC/USDT"]
    rules = {s: sol_rules(symbol=s, base=s.split("/")[0], step_size=1e-6, tick_size=1e-8) for s in symbols}
    start, end = _window_period(market)
    config = BacktestConfig(start=start, end=end, initial_capital=1000.0, universe_size=2, min_quote_volume=1e6)
    result = await Backtester(settings, config, source=MemorySource(market), rules=rules, symbols=symbols).run()

    assert result.trades and result.positions_open.max() <= settings.max_open_positions
    assert all(t.status == CLOSED for t in result.trades)
    assert all(start <= pd.Timestamp(t.opened_at) <= pd.Timestamp(t.closed_at) <= end for t in result.trades)
    dust_cost = sum(t.dust_qty * t.cost_per_unit for t in result.trades)
    pnl = sum(t.realized_pnl for t in result.trades)
    assert result.equity.iloc[-1] - 1000.0 == pytest.approx(pnl + result.dust_value - dust_cost, abs=1e-6)
    assert all(len(members) <= 2 for members in result.universe.values())
    report = build_report(result)
    assert report.full.trades == len(result.trades)
    assert report.segment_a.trades + report.segment_b.trades == len(result.trades)


# ----------------------------------------------------------------------
# Laporan
# ----------------------------------------------------------------------
def test_metrik_ekuitas_dan_drawdown():
    index = pd.date_range("2025-01-01 00:05", periods=4, freq="1D", tz="UTC")
    equity = pd.Series([100.0, 110.0, 99.0, 120.0], index=index)
    dd, days = drawdown_stats(equity)
    assert dd == pytest.approx(99 / 110 - 1) and days == pytest.approx(2.0)
    metrics = compute_metrics(equity, [], initial=100.0)
    assert metrics.total_return == pytest.approx(0.20) and metrics.max_drawdown == pytest.approx(99 / 110 - 1)
    returns = np.array([0.0, 0.10, 99 / 110 - 1, 120 / 99 - 1])
    assert metrics.sharpe == pytest.approx(returns.mean() / returns.std(ddof=1) * np.sqrt(365))
    assert buy_and_hold(equity) == pytest.approx({"return": 0.2, "max_drawdown": 99 / 110 - 1})
    monthly = monthly_returns(pd.Series([100.0, 105.0], index=pd.to_datetime(["2025-01-31", "2025-02-28"], utc=True)), 100.0)
    assert monthly.to_dict() == pytest.approx({"2025-01": 0.0, "2025-02": 0.05})


def test_metrik_trade():
    def trade(pnl, hours):
        return Position(symbol="SOL/USDT", mode="backtest", status=CLOSED, realized_pnl=pnl, risk_amount=5.0,
                        exit_reason="stop_loss" if pnl < 0 else "trailing_stop",
                        opened_at="2025-01-01T00:00:00+00:00", closed_at=f"2025-01-01T{hours:02d}:00:00+00:00")

    trades = [trade(10, 2), trade(-5, 4), trade(-5, 1), trade(5, 1)]
    index = pd.date_range("2025-01-01", periods=2, freq="1D", tz="UTC")
    m = compute_metrics(pd.Series([100.0, 105.0], index=index), trades, 100.0)
    assert (m.trades, m.wins, m.losses, m.win_rate) == (4, 2, 2, 0.5)
    assert m.profit_factor == pytest.approx(15 / 10)
    assert m.avg_r == pytest.approx((2 - 1 - 1 + 1) / 4) and m.max_consecutive_losses == 2
    assert m.avg_hold_hours == pytest.approx(2.0) and m.expectancy == pytest.approx(1.25)
    rows = breakdown(trades, lambda t: t.exit_reason)
    assert [(r.key, r.trades, r.win_rate) for r in rows] == [("trailing_stop", 2, 1.0), ("stop_loss", 2, 0.0)]


async def test_laporan_disimpan_dan_dibaca_ulang(market, tmp_path):
    settings = make_settings(tmp_path)
    symbols = ["AAA/USDT", "BTC/USDT"]
    rules = {s: sol_rules(symbol=s, base=s.split("/")[0], step_size=1e-6, tick_size=1e-8) for s in symbols}
    start, end = _window_period(market)
    config = BacktestConfig(start=start, end=end, initial_capital=500.0, universe_size=1, min_quote_volume=1e6)
    result = await Backtester(settings, config, source=MemorySource(market), rules=rules, symbols=symbols).run()
    report = build_report(result)
    assert "HASIL BACKTEST" in report.text and "Segmen B (OOS)" in report.text and "buy and hold BTC" in report.text
    assert "\u2014" not in report.text and "\u2013" not in report.text  # tanpa strip panjang
    folder = save_report(result, report, tmp_path / "runs" / "20250101-000000")
    assert {p.name for p in folder.iterdir()} >= {"report.txt", "report.json", "trades.csv", "equity.csv", "events.log", "journal.db"}
    assert latest_run(tmp_path / "runs") == folder


async def test_paralel_hasilnya_sama_dengan_satu_proses(market, tmp_path):
    settings = make_settings(tmp_path)
    params = EngineParams.from_settings(settings)
    start, end = _window_period(market)
    steps = step_times(start, start + pd.Timedelta(days=1))
    contexts = market_contexts(prepare_frames(market["BTC/USDT"]), steps, params, settings)
    symbols = ["AAA/USDT", "BBB/USDT"]
    active = {s: np.ones(len(steps), bool) for s in symbols}
    source = MemorySource({s: market[s] for s in symbols})
    serial = generate_signals(symbols, source, steps, active, contexts, {}, params, workers=1)
    parallel = generate_signals(symbols, source, steps, active, contexts, {}, params, workers=2)
    for symbol in symbols:
        assert [(s.timestamp, s.score) for s in serial[symbol].entries] == [(s.timestamp, s.score) for s in parallel[symbol].entries]
        assert serial[symbol].evaluations == parallel[symbol].evaluations
