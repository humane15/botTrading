"""CLI bot trading spot Binance.

Contoh pemakaian:
    python main.py universe              tampilkan 75 coin (cache 24 jam)
    python main.py universe --refresh    paksa pilih ulang dari Binance
    python main.py fetch SOL/USDT        ambil candle 5m, 15m, 30m, 1h
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
from pydantic import ValidationError

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
    try:
        if args.command == "universe":
            return run_async(_with_client(settings, lambda c: run_universe(settings, c, args.refresh, args.top)))
        if args.command == "fetch":
            return run_async(_with_client(settings, lambda c: run_fetch(settings, c, args.symbol)))
    except ccxt.BaseError as exc:
        print(f"Gagal terhubung ke Binance: {exc}", file=sys.stderr)
        return 1
    return 1  # pragma: no cover  (argparse sudah membatasi pilihan perintah)


if __name__ == "__main__":
    sys.exit(main())
