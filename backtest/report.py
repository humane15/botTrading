"""Laporan performa backtest: metrik, rincian, pembanding, dan penyimpanan file.

Metrik dihitung dari kurva ekuitas (nilai akun setiap candle 5m, termasuk posisi
terbuka) dan dari daftar trade tertutup:
* total return, CAGR (disetahunkan), max drawdown dan lamanya;
* Sharpe dan Sortino dari return HARIAN, disetahunkan dengan akar 365 (crypto
  diperdagangkan setiap hari), tanpa suku bunga bebas risiko;
* win rate, profit factor, rata rata R, median R, t-stat rata rata R
  (|t| < 2 berarti hasil belum bisa dibedakan dari kebetulan), expectancy;
* exposure (porsi waktu ada posisi terbuka) dan total fee.

Periode juga dibagi dua: segmen A (awal) dan segmen B (akhir, default 1/3
periode). Di Fase 5 modul learning hanya boleh belajar dari segmen A, dan
klaim performa memakai segmen B (out of sample).
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult, position_rows
from risk.positions import Position

ANNUAL_DAYS = 365


@dataclass(frozen=True)
class PerformanceMetrics:
    start: str
    end: str
    days: float
    initial_equity: float
    final_equity: float
    total_return: float
    cagr: float
    max_drawdown: float
    max_drawdown_days: float
    sharpe: float
    sortino: float
    calmar: float
    trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    avg_r: float
    median_r: float
    r_tstat: float
    expectancy: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    max_consecutive_losses: int
    avg_hold_hours: float
    exposure: float
    fees: float


@dataclass(frozen=True)
class BreakdownRow:
    key: str
    trades: int
    win_rate: float
    avg_r: float
    pnl: float


def _hours_between(start: str | None, end: str | None) -> float:
    if not start or not end:
        return float("nan")
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 3600


def drawdown_stats(equity: pd.Series) -> tuple[float, float]:
    """(max drawdown sebagai angka negatif, lama drawdown terpanjang dalam hari).

    Lama drawdown dihitung dari puncak sebelumnya sampai nilai akun kembali ke
    puncak itu (atau sampai akhir periode jika belum pulih).
    """
    if equity.empty:
        return 0.0, 0.0
    values = equity.to_numpy(dtype="float64")
    seconds = equity.index.as_unit("s").asi8
    max_dd = longest = 0.0
    peak, peak_at, underwater = values[0], seconds[0], False
    for value, when in zip(values, seconds, strict=True):
        if value >= peak:
            if underwater:
                longest = max(longest, (when - peak_at) / 86400)
                underwater = False
            peak, peak_at = value, when
        else:
            underwater = True
            max_dd = min(max_dd, value / peak - 1)
    if underwater:
        longest = max(longest, (seconds[-1] - peak_at) / 86400)
    return float(max_dd), float(longest)


def daily_returns(equity: pd.Series, initial: float) -> pd.Series:
    """Return harian dari nilai akun di akhir tiap hari UTC (hari pertama dibanding modal awal)."""
    if equity.empty:
        return pd.Series(dtype="float64")
    closes = equity.resample("1D").last().dropna()
    previous = closes.shift(1)
    previous.iloc[0] = initial
    return closes / previous - 1


def equity_metrics(equity: pd.Series, initial: float) -> dict[str, float]:
    final = float(equity.iloc[-1]) if not equity.empty else initial
    days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400 + 5 / 1440 if len(equity) else 0.0
    total = final / initial - 1 if initial > 0 else 0.0
    cagr = (1 + total) ** (ANNUAL_DAYS / days) - 1 if days > 0 and total > -1 else float("nan")
    if len(equity):
        with_start = pd.concat([pd.Series([initial], index=[equity.index[0] - pd.Timedelta(minutes=5)]), equity])
    else:
        with_start = equity
    max_dd, dd_days = drawdown_stats(with_start)
    returns = daily_returns(equity, initial)
    std = float(returns.std(ddof=1)) if len(returns) > 1 else float("nan")
    mean = float(returns.mean()) if len(returns) else float("nan")
    sharpe = mean / std * math.sqrt(ANNUAL_DAYS) if std and std > 0 else float("nan")
    downside = float(np.sqrt(np.mean(np.minimum(returns.to_numpy(), 0.0) ** 2))) if len(returns) else float("nan")
    sortino = mean / downside * math.sqrt(ANNUAL_DAYS) if downside and downside > 0 else float("nan")
    calmar = cagr / abs(max_dd) if max_dd < 0 and not math.isnan(cagr) else float("nan")
    return {
        "days": days, "initial_equity": initial, "final_equity": final, "total_return": total, "cagr": cagr,
        "max_drawdown": max_dd, "max_drawdown_days": dd_days, "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
    }


def trade_metrics(trades: Sequence[Position]) -> dict[str, float]:
    pnl = np.array([t.realized_pnl for t in trades], dtype="float64")
    r = np.array([t.r_multiple for t in trades], dtype="float64")
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    streak = longest = 0
    for value in pnl:
        streak = streak + 1 if value <= 0 else 0
        longest = max(longest, streak)
    holds = [h for h in (_hours_between(t.opened_at, t.closed_at) for t in trades) if not math.isnan(h)]
    n = len(trades)
    r_std = float(r.std(ddof=1)) if n > 1 else float("nan")
    return {
        "trades": n,
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate": len(wins) / n if n else float("nan"),
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else float("nan")),
        "avg_r": float(r.mean()) if n else float("nan"),
        "median_r": float(np.median(r)) if n else float("nan"),
        "r_tstat": float(r.mean() / (r_std / math.sqrt(n))) if n > 1 and r_std > 0 else float("nan"),
        "expectancy": float(pnl.mean()) if n else float("nan"),
        "avg_win": float(wins.mean()) if len(wins) else float("nan"),
        "avg_loss": float(losses.mean()) if len(losses) else float("nan"),
        "best_trade": float(pnl.max()) if n else float("nan"),
        "worst_trade": float(pnl.min()) if n else float("nan"),
        "max_consecutive_losses": longest,
        "avg_hold_hours": float(np.mean(holds)) if holds else float("nan"),
        "fees": float(sum(t.fees_quote for t in trades)),
    }


def compute_metrics(
    equity: pd.Series, trades: Sequence[Position], initial: float, positions_open: pd.Series | None = None
) -> PerformanceMetrics:
    values = equity_metrics(equity, initial) | trade_metrics(trades)
    exposure = float((positions_open > 0).mean()) if positions_open is not None and len(positions_open) else float("nan")
    start = f"{equity.index[0] - pd.Timedelta(minutes=5):%Y-%m-%d %H:%M}" if len(equity) else ""
    end = f"{equity.index[-1]:%Y-%m-%d %H:%M}" if len(equity) else ""
    return PerformanceMetrics(start=start, end=end, exposure=exposure, **values)  # type: ignore[arg-type]


def segment_metrics(result: BacktestResult, start: pd.Timestamp, end: pd.Timestamp, initial: float | None = None) -> PerformanceMetrics:
    """Metrik untuk sebagian periode: kurva ekuitas di [start, end) dan trade yang DIBUKA di rentang itu."""
    mask = (result.equity.index > start) & (result.equity.index <= end)
    equity = result.equity[mask]
    before = result.equity[result.equity.index <= start]
    base = initial if initial is not None else (float(before.iloc[-1]) if len(before) else result.config.initial_capital)
    trades = [t for t in result.trades if t.opened_at and start <= pd.Timestamp(t.opened_at) < end]
    return compute_metrics(equity, trades, base, result.positions_open[mask])


def breakdown(trades: Sequence[Position], key: Callable[[Position], str], min_trades: int = 1) -> list[BreakdownRow]:
    groups: dict[str, list[Position]] = {}
    for trade in trades:
        groups.setdefault(key(trade) or "(kosong)", []).append(trade)
    rows = [
        BreakdownRow(
            key=name,
            trades=len(items),
            win_rate=sum(1 for t in items if t.realized_pnl > 0) / len(items),
            avg_r=float(np.mean([t.r_multiple for t in items])),
            pnl=float(sum(t.realized_pnl for t in items)),
        )
        for name, items in groups.items()
        if len(items) >= min_trades
    ]
    return sorted(rows, key=lambda row: (row.trades, row.pnl), reverse=True)


def monthly_returns(equity: pd.Series, initial: float) -> pd.Series:
    if equity.empty:
        return pd.Series(dtype="float64")
    closes = equity.resample("ME").last()
    previous = closes.shift(1)
    previous.iloc[0] = initial
    returns = closes / previous - 1
    returns.index = returns.index.strftime("%Y-%m")
    return returns


def buy_and_hold(prices: pd.Series) -> dict[str, float]:
    prices = prices.dropna()
    if len(prices) < 2:
        return {"return": float("nan"), "max_drawdown": float("nan")}
    max_dd, _ = drawdown_stats(prices)
    return {"return": float(prices.iloc[-1] / prices.iloc[0] - 1), "max_drawdown": max_dd}


# ----------------------------------------------------------------------
# Format teks
# ----------------------------------------------------------------------
def _pct(value: float, sign: bool = True) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "n/a"
    return f"{value:+.2%}" if sign else f"{value:.2%}"


def _num(value: float, fmt: str = ".2f") -> str:
    if value is None or math.isnan(value):
        return "n/a"
    if math.isinf(value):
        return "tak hingga"
    return format(value, fmt)


def _usd(value: float) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{'-' if value < 0 else ''}${abs(value):,.2f}"


METRIC_ROWS: tuple[tuple[str, Callable[[PerformanceMetrics], str]], ...] = (
    ("Total return", lambda m: _pct(m.total_return)),
    ("CAGR (disetahunkan)", lambda m: _pct(m.cagr)),
    ("Max drawdown", lambda m: _pct(m.max_drawdown)),
    ("Drawdown terlama (hari)", lambda m: _num(m.max_drawdown_days, ".1f")),
    ("Sharpe (harian)", lambda m: _num(m.sharpe)),
    ("Sortino (harian)", lambda m: _num(m.sortino)),
    ("Calmar", lambda m: _num(m.calmar)),
    ("Jumlah trade", lambda m: str(m.trades)),
    ("Win rate", lambda m: _pct(m.win_rate, sign=False)),
    ("Profit factor", lambda m: _num(m.profit_factor)),
    ("Rata rata R", lambda m: _num(m.avg_r, "+.2f")),
    ("Median R", lambda m: _num(m.median_r, "+.2f")),
    ("t-stat rata rata R", lambda m: _num(m.r_tstat)),
    ("Expectancy per trade", lambda m: _usd(m.expectancy)),
    ("Rata rata menang", lambda m: _usd(m.avg_win)),
    ("Rata rata kalah", lambda m: _usd(m.avg_loss)),
    ("Trade terbaik", lambda m: _usd(m.best_trade)),
    ("Trade terburuk", lambda m: _usd(m.worst_trade)),
    ("Kalah beruntun terpanjang", lambda m: str(m.max_consecutive_losses)),
    ("Rata rata lama posisi (jam)", lambda m: _num(m.avg_hold_hours, ".1f")),
    ("Exposure (waktu ada posisi)", lambda m: _pct(m.exposure, sign=False)),
    ("Total fee", lambda m: _usd(m.fees)),
)


@dataclass(frozen=True)
class BacktestReport:
    full: PerformanceMetrics
    segment_a: PerformanceMetrics
    segment_b: PerformanceMetrics
    benchmark: dict[str, float]
    by_exit: list[BreakdownRow]
    by_pattern: list[BreakdownRow]
    by_symbol: list[BreakdownRow]
    by_regime: list[BreakdownRow]
    monthly: dict[str, float]
    text: str


def build_report(result: BacktestResult) -> BacktestReport:
    config = result.config
    full = compute_metrics(result.equity, result.trades, config.initial_capital, result.positions_open)
    split = config.oos_start
    seg_a = segment_metrics(result, config.start, split, initial=config.initial_capital)
    seg_b = segment_metrics(result, split, config.end + pd.Timedelta(minutes=5))
    bench = buy_and_hold(result.benchmark)
    by_exit = breakdown(result.trades, lambda t: t.exit_reason)
    by_pattern = breakdown(result.trades, lambda t: t.pattern, min_trades=3)
    by_symbol = breakdown(result.trades, lambda t: t.symbol)
    by_regime = breakdown(result.trades, lambda t: t.regime)
    monthly = monthly_returns(result.equity, config.initial_capital)
    report = BacktestReport(full, seg_a, seg_b, bench, by_exit, by_pattern, by_symbol, by_regime, monthly.to_dict(), "")
    return replace(report, text=format_report(result, report))


def _rows(rows: Sequence[BreakdownRow], limit: int) -> list[str]:
    return [
        f"  {row.key[:34]:<34} {row.trades:>5} trade | win {row.win_rate:>4.0%} | R {row.avg_r:+.2f} | PnL {_usd(row.pnl)}"
        for row in rows[:limit]
    ]


def format_report(result: BacktestResult, report: BacktestReport) -> str:
    config, s = result.config, result.settings
    universe_sizes = [len(v) for v in result.universe.values()] or [0]
    lines = [
        "=" * 78,
        "HASIL BACKTEST (simulasi pada data historis, bukan jaminan hasil masa depan)",
        "=" * 78,
        f"Periode   : {report.full.start} s/d {report.full.end} UTC ({report.full.days:.0f} hari, langkah candle 5m)",
        f"Universe  : rata rata {np.mean(universe_sizes):.0f} coin per hari (maks {config.universe_size}) dari "
        f"{len(result.symbols)} kandidat, dipilih harian dari volume hari sebelumnya >= ${config.min_quote_volume:,.0f}",
        f"Modal     : awal {_usd(config.initial_capital)} | akhir {_usd(report.full.final_equity)}",
        f"Biaya     : fee {float(s.get('fee_rate', 0)):.2%} per transaksi, slippage {float(s.get('slippage_rate', 0)):.2%}, "
        "filter min notional dan presisi tiap pair",
        f"Risiko    : {float(s.get('risk_per_trade', 0)):.1%} per trade, maks {s.get('max_open_positions')} posisi, "
        f"batas rugi harian {float(s.get('daily_loss_limit', 0)):.0%} / mingguan {float(s.get('weekly_loss_limit', 0)):.0%}",
        "Catatan   : parameter strategi belum dioptimasi pada data ini, jadi seluruh periode adalah",
        "            out of sample untuk strategi dasar. Segmen B disiapkan sebagai data uji learning Fase 5.",
        "",
        f"{'Metrik':<30}{'Seluruh periode':>17}{'Segmen A':>15}{'Segmen B (OOS)':>16}",
        f"{'':<30}{'':>17}{report.segment_a.start[:10] + '..':>15}{report.segment_b.start[:10] + '..':>16}",
    ]
    for label, render in METRIC_ROWS:
        lines.append(f"{label:<30}{render(report.full):>17}{render(report.segment_a):>15}{render(report.segment_b):>16}")
    total_pnl = sum(t.realized_pnl for t in result.trades)
    lines += [
        "",
        f"Total PnL trade {_usd(total_pnl)} | dust (sisa coin di bawah step size, tetap di akun) {_usd(result.dust_value)}",
        f"Pembanding buy and hold BTC: return {_pct(report.benchmark['return'])}, "
        f"max drawdown {_pct(report.benchmark['max_drawdown'])}",
        "",
        f"Sinyal: {result.evaluations:,} evaluasi | {result.entry_stats.signals:,} sinyal entry | "
        f"{result.entry_stats.opened:,} posisi dibuka",
    ]
    if result.entry_stats.skipped:
        lines.append("Sinyal entry yang tidak dieksekusi:")
        lines += [f"  {count:>7,}  {reason}" for reason, count in result.entry_stats.skipped.most_common(8)]
    if result.rejections:
        lines.append("Alasan sinyal ditolak engine (satu sinyal bisa punya beberapa alasan):")
        lines += [f"  {count:>9,}  {reason}" for reason, count in result.rejections.most_common()]
    if report.by_exit:
        lines += ["", "Alasan exit:"] + _rows(report.by_exit, 10)
    if report.by_regime:
        lines += ["", "Regime coin saat entry:"] + _rows(report.by_regime, 5)
    if report.by_pattern:
        lines += ["", "Pola setup terbanyak (minimal 3 trade):"] + _rows(report.by_pattern, 10)
    if report.by_symbol:
        best = sorted(report.by_symbol, key=lambda r: r.pnl, reverse=True)
        lines += ["", "Coin dengan PnL terbaik:"] + _rows(best, 5)
        lines += ["Coin dengan PnL terburuk:"] + _rows(best[::-1], 5)
    if report.monthly:
        lines += ["", "Return bulanan:"]
        lines += [f"  {month}  {_pct(value)}" for month, value in report.monthly.items()]
    if report.full.trades < 30:
        lines += ["", f"PERINGATAN: hanya {report.full.trades} trade, terlalu sedikit untuk kesimpulan statistik."]
    lines += [
        "",
        "Bias yang tersisa: kandidat coin diambil dari coin yang masih diperdagangkan sekarang (coin yang",
        "sudah delisting tidak ikut), entry dianggap terisi di harga penutupan candle + slippage, dan",
        "TP/stop dalam satu candle diasumsikan pesimis (stop lebih dulu).",
    ]
    timing = ", ".join(f"{name} {seconds:.0f} dtk" for name, seconds in result.timings.items())
    if timing:
        lines.append(f"Waktu proses: {timing}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Simpan dan baca
# ----------------------------------------------------------------------
def _json_safe(value: object) -> object:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    return value


def save_report(result: BacktestResult, report: BacktestReport, directory: Path) -> Path:
    """Simpan report.txt, report.json, trades.csv, equity.csv, events.log, dan journal.db."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.txt").write_text(report.text + "\n", encoding="utf-8")
    payload = {
        "config": {
            "start": str(result.config.start), "end": str(result.config.end),
            "initial_capital": result.config.initial_capital, "universe_size": result.config.universe_size,
            "min_quote_volume": result.config.min_quote_volume, "oos_start": str(result.config.oos_start),
        },
        "settings": result.settings,
        "weights": result.weights,
        "symbols": result.symbols,
        "metrics": {"full": asdict(report.full), "segment_a": asdict(report.segment_a), "segment_b": asdict(report.segment_b)},
        "benchmark_btc": report.benchmark,
        "monthly": report.monthly,
        "by_exit": [asdict(r) for r in report.by_exit],
        "by_pattern": [asdict(r) for r in report.by_pattern],
        "by_symbol": [asdict(r) for r in report.by_symbol],
        "by_regime": [asdict(r) for r in report.by_regime],
        "entry_stats": {"signals": result.entry_stats.signals, "opened": result.entry_stats.opened,
                        "skipped": dict(result.entry_stats.skipped)},
        "dust_value": result.dust_value,
        "evaluations": result.evaluations,
        "rejections": dict(result.rejections),
        "timings": result.timings,
    }
    (directory / "report.json").write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    rows = position_rows(result.trades)
    with (directory / "trades.csv").open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    frame = pd.DataFrame({"equity": result.equity, "positions": result.positions_open, "btc": result.benchmark})
    frame.to_csv(directory / "equity.csv", index_label="timestamp")
    (directory / "events.log").write_text("\n".join(result.events) + "\n", encoding="utf-8")
    target = sqlite3.connect(directory / "journal.db")
    try:
        result.database.backup(target)
    finally:
        target.close()
    return directory


def latest_run(root: Path) -> Path | None:
    runs = sorted(p for p in root.glob("*") if (p / "report.txt").exists()) if root.exists() else []
    return runs[-1] if runs else None


def metrics_from_json(directory: Path) -> Mapping[str, object]:
    return json.loads((directory / "report.json").read_text(encoding="utf-8"))
