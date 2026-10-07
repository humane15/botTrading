"""Helper test: settings terisolasi, jam palsu, dan data pasar sintetis."""

from __future__ import annotations

import asyncio
import zlib
from typing import Any

import ccxt

from config.settings import TIMEFRAME_MS, Settings, load_settings

HOUR_MS = 3_600_000
# Titik awal waktu yang sejajar dengan jam penuh (UTC), memudahkan hitungan candle.
T0 = (1_760_000_000_000 // HOUR_MS) * HOUR_MS


def make_settings(tmp_path, **overrides: Any) -> Settings:
    """Settings terisolasi: tanpa .env, tanpa environment variable, folder di tmp_path."""
    overrides.setdefault("data_dir", tmp_path / "data")
    overrides.setdefault("logs_dir", tmp_path / "logs")
    return load_settings(env_file=None, environ={}, **overrides)


ASYNC_METHODS = (
    "load_markets", "fetch_tickers", "fetch_ticker", "fetch_ohlcv", "fetch_time", "fetch_balance",
    "fetch", "close", "load_time_difference", "create_order", "sapi_get_spot_delist_schedule",
    "sapi_get_account_apirestrictions",
)


def install_markets(exchange: Any, markets: dict[str, dict[str, Any]]) -> None:
    """Atur load_markets mock agar mengisi exchange.markets seperti ccxt asli."""

    async def _load_markets(reload: bool = False, params: dict | None = None) -> dict[str, dict[str, Any]]:
        exchange.markets = markets
        return markets

    exchange.load_markets.side_effect = _load_markets


class FakeTime:
    """Jam palsu untuk ExchangeClient: sleep() langsung memajukan waktu."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay
        await asyncio.sleep(0)


class ManualTime:
    """Jam palsu yang hanya maju jika test memajukannya (untuk uji cooldown)."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        target = self.now + delay
        while self.now < target:
            await asyncio.sleep(0)


# ----------------------------------------------------------------------
# Data pasar sintetis
# ----------------------------------------------------------------------
def make_market(
    base: str,
    quote: str = "USDT",
    *,
    active: bool = True,
    spot: bool = True,
    status: str = "TRADING",
    permissions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": f"{base}{quote}",
        "symbol": f"{base}/{quote}",
        "base": base,
        "quote": quote,
        "spot": spot,
        "type": "spot" if spot else "swap",
        "active": active,
        "precision": {"amount": 0.001, "price": 0.01},
        "limits": {"cost": {"min": 5.0}},
        "info": {"status": status, "isSpotTradingAllowed": spot, "permissions": permissions or ["SPOT"]},
    }


def make_ticker(symbol: str, quote_volume: float | None, last: float = 10.0, range_pct: float = 0.08) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "last": last,
        "high": last * (1 + range_pct / 2),
        "low": last * (1 - range_pct / 2),
        "quoteVolume": quote_volume,
        "baseVolume": (quote_volume or 0) / last,
    }


class FakeClock:
    """Jam dalam milidetik untuk DataFeed."""

    def __init__(self, now_ms: int) -> None:
        self.now_ms = now_ms

    def __call__(self) -> int:
        return self.now_ms

    def advance(self, ms: int) -> None:
        self.now_ms += ms


class FakeKlineServer:
    """Simulasi endpoint klines Binance untuk dipasang sebagai side_effect AsyncMock.

    Candle dibuat deterministik per (simbol, timeframe, open time). Candle yang
    masih berjalan dikembalikan dengan harga berbeda supaya test bisa
    memastikan candle tersebut tidak pernah masuk cache.
    """

    def __init__(self, clock: FakeClock, lag_candles: int = 0) -> None:
        self.clock = clock
        self.lag_candles = lag_candles
        self.fail_symbols: set[str] = set()
        self.in_flight = 0
        self.max_in_flight = 0

    @staticmethod
    def candle(symbol: str, tf_ms: int, open_ms: int, forming: bool = False) -> list[float]:
        base = 100 + zlib.crc32(symbol.encode()) % 50
        step = (open_ms // tf_ms) % 17
        open_ = base + step * 0.1
        close = open_ + (0.05 if step % 2 else -0.05) + (0.5 if forming else 0.0)
        high = max(open_, close) + 0.2
        low = min(open_, close) - 0.2
        volume = 1_000 + step
        return [open_ms, open_, high, low, close, volume]

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", since: int | None = None, limit: int | None = None, params: dict | None = None) -> list[list[float]]:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0)
            if symbol in self.fail_symbols:
                raise ccxt.NetworkError(f"binance simulasi gangguan jaringan untuk {symbol}")
            tf_ms = TIMEFRAME_MS[timeframe]
            current_open = self.clock() // tf_ms * tf_ms
            newest = current_open - self.lag_candles * tf_ms
            limit = limit or 500
            start = newest - (limit - 1) * tf_ms if since is None else -(-since // tf_ms) * tf_ms
            end = min(newest, start + (limit - 1) * tf_ms)
            return [self.candle(symbol, tf_ms, t, forming=(t == current_open)) for t in range(start, end + 1, tf_ms)]
        finally:
            self.in_flight -= 1
