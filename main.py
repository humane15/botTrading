"""CLI bot trading spot Binance.

Contoh pemakaian:
    python main.py universe              tampilkan 75 coin (cache 24 jam)
    python main.py universe --refresh    paksa pilih ulang dari Binance
    python main.py fetch SOL/USDT        ambil candle 5m, 15m, 30m, 1h
    python main.py analyze SOL/USDT      analisis lengkap satu coin (skor, alasan, rencana)
    python main.py scan --top 10         analisis semua coin universe, tampilkan sinyal teratas
    python main.py positions             posisi aktif dan riwayat posisi (dari database)
    python main.py backtest              backtest (Fase 4)
    python main.py train                 latih model ML (Fase 5)
    python main.py report                laporan performa (Fase 4 dan 5)
    python main.py paper                 paper trading (Fase 6)
    python main.py live --confirm-live   live trading (Fase 7, butuh TRADING_MODE=live)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, TypeVar

import ccxt
import pandas as pd
from pydantic import ValidationError

from analysis.confluence import (
    ConfluenceEngine,
    EngineParams,
    SignalResult,
    describe_signal,
    format_signal,
    prepare_frames,
    rank_signals,
)
from analysis.regime import CircuitBreaker, MarketContext, analyze_market
from analysis.weights import WeightStore
from config.logging_setup import setup_logging
from config.settings import (
    DEFAULT_ENV_FILE,
    LiveTradingNotAllowed,
    Settings,
    ensure_live_allowed,
    load_settings,
)
from core.data_feed import DataFeed
from core.exchange import ExchangeClient
from core.universe import UniverseSelector
from risk.positions import DUST, PENDING, Position, PositionStore

T = TypeVar("T")

# Perintah yang belum diimplementasikan beserta fase pengerjaannya.
PENDING_PHASE = {"backtest": 4, "report": 4, "train": 5, "paper": 6, "live": 7}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py", description="Bot trading spot Binance multi timeframe")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="lokasi file .env (default: .env di root proyek)")
    sub = parser.add_subparsers(dest="command", required=True, metavar="PERINTAH")

    p_universe = sub.add_parser("universe", help="tampilkan atau perbarui daftar coin")
    p_universe.add_argument("--refresh", action="store_true", help="paksa pilih ulang dari Binance")
    p_universe.add_argument("--top", type=int, default=None, help="jumlah baris yang ditampilkan")

    p_fetch = sub.add_parser("fetch", help="ambil candle multi timeframe untuk satu simbol")
    p_fetch.add_argument("symbol", help="contoh: SOL/USDT")

    p_analyze = sub.add_parser("analyze", help="analisis teknikal lengkap satu simbol")
    p_analyze.add_argument("symbol", help="contoh: SOL/USDT")

    p_scan = sub.add_parser("scan", help="analisis semua coin universe dan tampilkan sinyal teratas")
    p_scan.add_argument("--top", type=int, default=10, help="jumlah sinyal yang ditampilkan (default 10)")

    p_positions = sub.add_parser("positions", help="posisi aktif dan riwayat posisi dari database")
    p_positions.add_argument("--limit", type=int, default=10, help="jumlah posisi tertutup terakhir yang ditampilkan (default 10)")

    sub.add_parser("backtest", help="jalankan backtest (Fase 4)")
    sub.add_parser("report", help="laporan performa (Fase 4 dan 5)")
    sub.add_parser("train", help="latih ulang model machine learning (Fase 5)")
    sub.add_parser("paper", help="paper trading dengan harga live dan saldo virtual (Fase 6)")
    p_live = sub.add_parser("live", help="live trading dengan uang sungguhan (Fase 7)")
    p_live.add_argument("--confirm-live", action="store_true", help="konfirmasi eksplisit untuk mengaktifkan mode live")
    return parser


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    if sys.platform == "win32":
        # aiohttp (dipakai ccxt) paling stabil dengan SelectorEventLoop di Windows.
        if sys.version_info >= (3, 12):
            return asyncio.run(coro, loop_factory=asyncio.SelectorEventLoop)
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # pragma: no cover
    return asyncio.run(coro)


async def _with_client(settings: Settings, action: Callable[[ExchangeClient], Awaitable[int]]) -> int:
    async with ExchangeClient(settings) as client:
        return await action(client)


async def run_universe(settings: Settings, client: ExchangeClient, refresh: bool = False, top: int | None = None) -> int:
    selector = UniverseSelector(client, settings)
    universe = await selector.get_universe(force_refresh=refresh)
    now = datetime.now(timezone.utc)
    origin = "cache" if universe.from_cache else "baru"
    print(
        f"Universe: {len(universe.symbols)} coin | sumber {universe.source} ({origin}) | "
        f"dibuat {universe.generated_at:%Y-%m-%d %H:%M} UTC | umur {universe.age(now).total_seconds() / 3600:.1f} jam"
    )
    if not universe.validated:
        print("PERINGATAN: daftar cadangan belum tervalidasi terhadap market Binance")
    volumes = {row.get("symbol"): row.get("quote_volume") for row in universe.details}
    rows = universe.symbols if top is None else universe.symbols[:top]
    print(f"{'#':>3}  {'Simbol':<14} {'Volume 24 jam':>16}")
    for rank, symbol in enumerate(rows, start=1):
        volume = volumes.get(symbol)
        volume_text = f"${volume / 1e6:,.1f} jt" if volume else "-"
        print(f"{rank:>3}  {symbol:<14} {volume_text:>16}")
    if universe.excluded:
        summary = ", ".join(f"{reason} {len(items)}" for reason, items in sorted(universe.excluded.items()))
        print(f"Dikecualikan: {summary}")
    return 0


async def run_fetch(settings: Settings, client: ExchangeClient, symbol: str) -> int:
    symbol = symbol.upper()
    markets = await client.load_markets()
    if symbol not in markets:
        print(f"Simbol {symbol} tidak ditemukan di Binance spot", file=sys.stderr)
        return 1
    try:
        await client.sync_time()
    except ccxt.BaseError as exc:
        print(f"Sinkron waktu gagal, memakai jam lokal: {exc}", file=sys.stderr)
    feed = DataFeed.from_settings(client, settings)
    frames = await feed.get_multi_timeframe(symbol)
    print(f"{symbol} | candle tertutup per timeframe:")
    for timeframe, frame in frames.items():
        if frame.empty:
            print(f"  {timeframe:>4} | tidak ada data")
            continue
        last = frame.iloc[-1]
        print(
            f"  {timeframe:>4} | {len(frame):>4} candle | terakhir {frame.index[-1]:%Y-%m-%d %H:%M} UTC | "
            f"O {last['open']:.8g} H {last['high']:.8g} L {last['low']:.8g} C {last['close']:.8g} V {last['volume']:.6g}"
        )
    print(f"Request API: {client.stats.requests} | retry: {client.stats.retries} | rate limit: {client.stats.rate_limit_hits}")
    return 0


def build_engine(settings: Settings) -> ConfluenceEngine:
    """Confluence engine dengan bobot dari database (data/bot.db)."""
    store = WeightStore.open(settings.db_path)
    try:
        weights = store.snapshot()
    finally:
        store.conn.close()
    return ConfluenceEngine(weights, EngineParams.from_settings(settings))


def build_breaker(settings: Settings) -> CircuitBreaker:
    return CircuitBreaker(drop_threshold=settings.circuit_breaker_drop, cooldown=pd.Timedelta(hours=settings.circuit_breaker_hours))


async def market_context(
    feed: DataFeed, engine: ConfluenceEngine, settings: Settings, breaker: CircuitBreaker, btc_frames: Mapping[str, Any] | None = None
) -> MarketContext:
    """Regime BTC dan status circuit breaker (BTC sebagai acuan pasar)."""
    raw = btc_frames if btc_frames is not None else await feed.get_multi_timeframe(settings.btc_symbol)
    frames = prepare_frames(raw, engine.params.indicators)
    return analyze_market(
        frames,
        breaker,
        trend_timeframe=engine.params.timeframe_of("trend"),
        breaker_timeframe=engine.params.timeframe_of("trigger"),
        params=engine.params.regime,
    )


async def run_analyze(settings: Settings, client: ExchangeClient, symbol: str) -> int:
    symbol = symbol.upper()
    markets = await client.load_markets()
    if symbol not in markets:
        print(f"Simbol {symbol} tidak ditemukan di Binance spot", file=sys.stderr)
        return 1
    try:
        await client.sync_time()
    except ccxt.BaseError as exc:
        print(f"Sinkron waktu gagal, memakai jam lokal: {exc}", file=sys.stderr)
    feed = DataFeed.from_settings(client, settings)
    engine = build_engine(settings)
    market = await market_context(feed, engine, settings, build_breaker(settings))
    frames = prepare_frames(await feed.get_multi_timeframe(symbol), engine.params.indicators)
    print(describe_signal(engine.evaluate(symbol, frames, market)))
    return 0


async def run_scan(settings: Settings, client: ExchangeClient, top: int = 10) -> int:
    universe = await UniverseSelector(client, settings).get_universe()
    try:
        await client.sync_time()
    except ccxt.BaseError as exc:
        print(f"Sinkron waktu gagal, memakai jam lokal: {exc}", file=sys.stderr)
    feed = DataFeed.from_settings(client, settings)
    engine = build_engine(settings)
    symbols = list(dict.fromkeys([*universe.symbols, settings.btc_symbol]))
    refresh = await feed.refresh(symbols)
    complete = set(refresh.complete_symbols(engine.params.timeframes))
    if settings.btc_symbol in complete:
        btc_frames = refresh.data[settings.btc_symbol]
        market = await market_context(feed, engine, settings, build_breaker(settings), btc_frames)
    else:
        print(f"Data {settings.btc_symbol} tidak lengkap: regime BTC dan circuit breaker tidak diketahui", file=sys.stderr)
        market = MarketContext()

    results: list[SignalResult] = []
    for symbol in universe.symbols:
        if symbol in complete:
            frames = prepare_frames(refresh.data[symbol], engine.params.indicators)
            results.append(engine.evaluate(symbol, frames, market))
    ranked = rank_signals(results)
    breaker = "AKTIF" if market.circuit_breaker_active else "tidak aktif"
    print(f"[SCAN] {len(universe.symbols)} coin | Regime BTC: {market.label} | Circuit breaker: {breaker}")
    if refresh.errors:
        failed = sorted({symbol for symbol, _ in refresh.errors})
        print(f"Data gagal diambil untuk {len(failed)} coin: {', '.join(failed[:10])}")
    print("Sinyal teratas:")
    for rank, result in enumerate(ranked[:top], start=1):
        print(f"  {rank:>2}. {format_signal(result)}")
    entries = sum(1 for r in results if r.is_entry)
    print(f"Entry valid: {entries} dari {len(results)} coin yang dianalisis (belum ada order, analisis saja)")
    return 0


def _short_time(iso: str | None) -> str:
    return f"{datetime.fromisoformat(iso):%Y-%m-%d %H:%M} UTC" if iso else "tidak tercatat"


def _format_active(position: Position) -> str:
    if position.status == PENDING:
        return f"#{position.id} {position.symbol} | menunggu hasil order entry (dicek ulang saat bot start)"
    stop_kind = "stop breakeven/trailing" if position.breakeven else "stop loss"
    return (
        f"#{position.id} {position.symbol} | {position.status} | entry {position.entry_price:.8g} | "
        f"sisa {position.qty:.8g} dari {position.initial_qty:.8g} | {stop_kind} {position.stop_price:.8g} | "
        f"TP1 {position.tp1_price:.8g} | dibuka {_short_time(position.opened_at)}"
    )


def _format_closed(position: Position, quote: str) -> str:
    return (
        f"#{position.id} {position.symbol} | PnL {position.realized_pnl:+.2f} {quote} ({position.r_multiple:+.2f}R) | "
        f"{position.exit_reason} | ditutup {_short_time(position.closed_at)}"
    )


def run_positions(settings: Settings, limit: int = 10) -> int:
    """Posisi aktif dan riwayat posisi dari database, tanpa koneksi ke Binance."""
    store = PositionStore.open(settings.db_path)
    try:
        mode = settings.trading_mode
        active = store.active(mode)
        stuck = store.with_status([DUST], mode)
        closed = store.closed(mode)
    finally:
        store.conn.close()
    quote = settings.quote_asset
    label = "paper (saldo virtual)" if mode == "paper" else "live (uang sungguhan)"
    kill = "AKTIF, tidak ada entry baru (hapus file STOP untuk melanjutkan)" if settings.kill_switch_file.exists() else "tidak aktif"
    print(f"Posisi mode {label} | database {settings.db_path}")
    print(f"Kill switch: {kill}")
    print(f"Posisi aktif ({len(active)}):" if active else "Posisi aktif: tidak ada")
    for position in active:
        print(f"  {_format_active(position)}")
    if stuck:
        print(f"Perlu dicek manual ({len(stuck)}):")
        for position in stuck:
            print(f"  #{position.id} {position.symbol} | sisa {position.qty:.8g} {position.base} | {position.exit_reason}")
    if not closed:
        print("Belum ada posisi tertutup")
        return 0
    shown = closed[: max(limit, 0)]
    print(f"Posisi tertutup terakhir ({len(shown)} dari {len(closed)}):")
    for position in shown:
        print(f"  {_format_closed(position, quote)}")
    wins = sum(1 for p in closed if p.realized_pnl > 0)
    total = sum(p.realized_pnl for p in closed)
    print(
        f"Ringkasan {len(closed)} posisi tertutup: menang {wins}, kalah {len(closed) - wins}, "
        f"win rate {wins / len(closed):.1%}, total PnL {total:+.2f} {quote}"
    )
    return 0


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(args.env_file, environ=environ)
    except ValidationError as exc:
        print(f"Konfigurasi .env tidak valid:\n{exc}", file=sys.stderr)
        return 1
    setup_logging(settings.log_level, settings.logs_dir, settings.secret_values())

    if args.command == "live":
        try:
            ensure_live_allowed(settings, args.confirm_live)
        except LiveTradingNotAllowed as exc:
            print(exc, file=sys.stderr)
            return 1

    if args.command in PENDING_PHASE:
        print(f"Perintah '{args.command}' baru tersedia di Fase {PENDING_PHASE[args.command]}.")
        return 2

    settings.ensure_dirs()
    if args.command == "positions":
        return run_positions(settings, args.limit)
    try:
        if args.command == "universe":
            return run_async(_with_client(settings, lambda c: run_universe(settings, c, args.refresh, args.top)))
        if args.command == "fetch":
            return run_async(_with_client(settings, lambda c: run_fetch(settings, c, args.symbol)))
        if args.command == "analyze":
            return run_async(_with_client(settings, lambda c: run_analyze(settings, c, args.symbol)))
        if args.command == "scan":
            return run_async(_with_client(settings, lambda c: run_scan(settings, c, args.top)))
    except ccxt.BaseError as exc:
        print(f"Gagal terhubung ke Binance: {exc}", file=sys.stderr)
        return 1
    return 1  # pragma: no cover  (argparse sudah membatasi pilihan perintah)


if __name__ == "__main__":
    sys.exit(main())
