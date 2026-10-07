"""Koneksi ke Binance spot lewat ccxt async.

Semua panggilan API melewati ExchangeClient.call() sehingga:

* jumlah request paralel dibatasi Semaphore (default 5);
* error jaringan dan rate limit (HTTP 429, juga 418 ban IP) diulang dengan
  exponential backoff plus jitter;
* saat kena rate limit, SEMUA request ikut menunggu (cooldown global),
  karena batas Binance dihitung per IP, bukan per request;
* operasi yang tidak idempotent (membuat order) tidak diulang ketika
  hasilnya tidak pasti (timeout atau HTTP 5xx), supaya tidak terjadi order
  ganda. Penolakan rate limit tetap aman diulang karena Binance menolak
  request tersebut sebelum diproses.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

import ccxt
import ccxt.async_support as ccxt_async

from config.settings import Settings

log = logging.getLogger(__name__)

# Binance mem-ban IP minimal 2 menit setelah HTTP 418 (ban bertambah lama jika diulang).
IP_BAN_MIN_DELAY = 120.0
RATE_LIMIT_ERRORS = (ccxt.DDoSProtection, ccxt.RateLimitExceeded)
_IP_BAN_PATTERN = re.compile(r"\b418\b")


class UnsafeApiKeyError(RuntimeError):
    """API key punya izin berbahaya (withdraw) atau tidak dibatasi IP whitelist."""


@dataclass
class RequestStats:
    """Statistik request untuk status console dan diagnosa rate limit."""

    requests: int = 0
    retries: int = 0
    rate_limit_hits: int = 0
    failures: int = 0


@dataclass(frozen=True)
class ApiPermissionReport:
    """Ringkasan izin API key dari endpoint /sapi/v1/account/apiRestrictions."""

    can_read: bool
    can_trade_spot: bool
    can_withdraw: bool
    ip_restricted: bool
    extra_permissions: tuple[str, ...] = ()

    _EXTRA_FLAGS = (
        ("enableFutures", "futures"),
        ("enableMargin", "margin"),
        ("enableVanillaOptions", "options"),
        ("enablePortfolioMarginTrading", "portfolio margin"),
        ("enableInternalTransfer", "transfer internal"),
        ("permitsUniversalTransfer", "universal transfer"),
    )

    @classmethod
    def from_response(cls, data: Mapping[str, Any]) -> ApiPermissionReport:
        def flag(key: str) -> bool:
            return data.get(key) is True

        return cls(
            can_read=flag("enableReading"),
            can_trade_spot=flag("enableSpotAndMarginTrading"),
            can_withdraw=flag("enableWithdrawals"),
            ip_restricted=flag("ipRestrict"),
            extra_permissions=tuple(name for key, name in cls._EXTRA_FLAGS if flag(key)),
        )

    @property
    def problems(self) -> list[str]:
        """Masalah yang membuat API key tidak boleh dipakai untuk live trading."""
        problems = []
        if self.can_withdraw:
            problems.append("izin Withdraw AKTIF, matikan di pengaturan API Binance")
        if not self.ip_restricted:
            problems.append("API key tidak dibatasi IP whitelist")
        if not self.can_read:
            problems.append("izin Read tidak aktif")
        if not self.can_trade_spot:
            problems.append("izin Spot Trading tidak aktif")
        return problems

    @property
    def warnings(self) -> list[str]:
        return [f"izin {name} aktif padahal tidak dibutuhkan bot spot" for name in self.extra_permissions]

    @property
    def is_safe(self) -> bool:
        return not self.problems


def build_exchange(settings: Settings) -> ccxt_async.binance:
    """Buat instance ccxt async Binance spot dari settings."""
    config: dict[str, Any] = {
        "enableRateLimit": True,
        "timeout": settings.request_timeout_ms,
        "options": {
            "defaultType": "spot",
            # Hanya muat market spot: startup lebih cepat dan hemat weight API.
            "fetchMarkets": {"types": ["spot"]},
            "adjustForTimeDifference": True,
        },
    }
    if settings.binance_api_key is not None and settings.binance_api_secret is not None:
        config["apiKey"] = settings.binance_api_key.get_secret_value()
        config["secret"] = settings.binance_api_secret.get_secret_value()
    exchange = ccxt_async.binance(config)
    if settings.binance_testnet:
        exchange.set_sandbox_mode(True)
    return exchange


def _wall_clock_ms() -> int:
    return int(time.time() * 1000)


class ExchangeClient:
    """Pembungkus ccxt async dengan pembatas konkurensi, retry, dan cooldown global."""

    def __init__(
        self,
        settings: Settings,
        exchange: Any | None = None,
        *,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock_ms: Callable[[], int] = _wall_clock_ms,
        jitter: float = 0.2,
        rng: random.Random | None = None,
    ) -> None:
        self.settings = settings
        self.exchange = exchange if exchange is not None else build_exchange(settings)
        self.stats = RequestStats()
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
        self._sleep = sleep
        self._monotonic = monotonic
        self._wall_clock_ms = wall_clock_ms
        self._jitter = jitter
        self._rng = rng or random.Random()
        self._cooldown_until = 0.0
        self._time_offset_ms = 0

    async def __aenter__(self) -> ExchangeClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def close(self) -> None:
        close = getattr(self.exchange, "close", None)
        if close is not None:
            await close()

    # ------------------------------------------------------------------
    # Waktu
    # ------------------------------------------------------------------
    def now_ms(self) -> int:
        """Waktu sekarang menurut jam server Binance (jam lokal + offset)."""
        return self._wall_clock_ms() + self._time_offset_ms

    async def sync_time(self) -> int:
        """Ukur selisih jam lokal terhadap server Binance, kembalikan offset (ms)."""
        before = self._wall_clock_ms()
        server_ms = int(await self.call("fetch_time"))
        after = self._wall_clock_ms()
        self._time_offset_ms = server_ms - (before + after) // 2
        if abs(self._time_offset_ms) > 1_000:
            log.warning("Jam lokal selisih %d ms dengan server Binance, offset dipakai otomatis", self._time_offset_ms)
        return self._time_offset_ms

    # ------------------------------------------------------------------
    # Inti: semua request lewat sini
    # ------------------------------------------------------------------
    async def call(self, method: str, *args: Any, idempotent: bool = True, retries: int | None = None, **kwargs: Any) -> Any:
        """Panggil method ccxt dengan pembatas konkurensi dan retry.

        idempotent=False dipakai untuk operasi seperti membuat order: hanya
        penolakan rate limit yang diulang, timeout dan 5xx tidak diulang.
        """
        func = getattr(self.exchange, method)
        max_retries = self.settings.max_retries if retries is None else retries
        attempt = 0
        resynced = False
        while True:
            await self._wait_for_cooldown()
            async with self._semaphore:
                # Cooldown bisa dipasang task lain selama kita menunggu slot.
                await self._wait_for_cooldown()
                self.stats.requests += 1
                try:
                    return await func(*args, **kwargs)
                except ccxt.BaseError as exc:
                    error = exc

            if not resynced and "-1021" in str(error):
                # Timestamp di luar recvWindow: sinkronkan ulang jam lalu coba lagi sekali.
                resynced = True
                await self._resync_exchange_time()
                continue

            delay = self._backoff_delay(attempt)
            if isinstance(error, RATE_LIMIT_ERRORS):
                # Batas Binance berlaku per IP: semua request ikut jeda,
                # termasuk ketika request ini sendiri sudah menyerah.
                self.stats.rate_limit_hits += 1
                delay = self._rate_limit_delay(delay, error)
                self._start_cooldown(delay)
            if attempt >= max_retries or not self._is_retryable(error, idempotent):
                self.stats.failures += 1
                raise error
            attempt += 1
            self.stats.retries += 1
            log.warning(
                "%s gagal (%s: %s), percobaan ulang %d/%d dalam %.1f detik",
                method, type(error).__name__, str(error)[:200], attempt, max_retries, delay,
            )
            await self._sleep(delay)

    @staticmethod
    def _is_retryable(error: Exception, idempotent: bool) -> bool:
        if isinstance(error, RATE_LIMIT_ERRORS):
            return True
        if isinstance(error, ccxt.NetworkError):
            return idempotent
        return False

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff: base, 2x base, 4x base, ... dibatasi retry_max_delay, plus jitter."""
        delay = min(self.settings.retry_max_delay, self.settings.retry_base_delay * (2 ** attempt))
        if self._jitter:
            delay *= 1 + self._rng.uniform(0, self._jitter)
        return delay

    def _rate_limit_delay(self, delay: float, error: Exception) -> float:
        retry_after = self._retry_after_seconds()
        if retry_after is not None:
            return max(delay, retry_after)
        if _IP_BAN_PATTERN.search(str(error)):
            return max(delay, IP_BAN_MIN_DELAY)
        return delay

    def _retry_after_seconds(self) -> float | None:
        """Baca header Retry-After dari respons terakhir (best effort)."""
        headers = getattr(self.exchange, "last_response_headers", None)
        if not isinstance(headers, Mapping):
            return None
        value = headers.get("Retry-After") or headers.get("retry-after")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def cooldown_remaining(self) -> float:
        """Sisa waktu (detik) jeda global akibat rate limit, 0 jika tidak sedang jeda."""
        return max(0.0, self._cooldown_until - self._monotonic())

    def _start_cooldown(self, delay: float) -> None:
        self._cooldown_until = max(self._cooldown_until, self._monotonic() + delay)

    async def _wait_for_cooldown(self) -> None:
        while True:
            remaining = self._cooldown_until - self._monotonic()
            if remaining <= 0:
                return
            await self._sleep(remaining)

    async def _resync_exchange_time(self) -> None:
        loader = getattr(self.exchange, "load_time_difference", None)
        if loader is None:
            return
        try:
            await loader()
        except ccxt.BaseError as exc:
            log.warning("Gagal sinkron waktu dengan Binance: %s", exc)

    # ------------------------------------------------------------------
    # Data pasar
    # ------------------------------------------------------------------
    @property
    def markets(self) -> dict[str, dict[str, Any]]:
        """Market yang sudah dimuat (kosong jika load_markets belum dipanggil)."""
        markets = getattr(self.exchange, "markets", None)
        return markets if isinstance(markets, dict) else {}

    async def load_markets(self, reload: bool = False) -> dict[str, dict[str, Any]]:
        return await self.call("load_markets", reload)

    async def fetch_tickers(self, symbols: list[str] | None = None) -> dict[str, dict[str, Any]]:
        return await self.call("fetch_tickers", symbols)

    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        return await self.call("fetch_ticker", symbol)

    async def fetch_ohlcv(self, symbol: str, timeframe: str, since: int | None = None, limit: int | None = None) -> list[list[float]]:
        return await self.call("fetch_ohlcv", symbol, timeframe, since, limit)

    async def fetch_json(self, url: str, retries: int | None = None) -> Any:
        """GET mentah ke URL publik Binance (dipakai untuk data tag coin)."""
        return await self.call("fetch", url, "GET", retries=retries)

    async def fetch_delisting_ids(self) -> set[str]:
        """ID market (misal 'XYZUSDT') yang dijadwalkan delisting.

        Endpoint ini butuh API key; tanpa API key (atau di testnet)
        dikembalikan set kosong dan bot bergantung pada status market.
        """
        if not self.settings.has_api_credentials or self.settings.binance_testnet:
            return set()
        rows = await self.call("sapi_get_spot_delist_schedule", retries=1)
        ids: set[str] = set()
        for row in rows or []:
            ids.update(str(symbol) for symbol in row.get("symbols") or [])
        return ids

    # ------------------------------------------------------------------
    # Akun
    # ------------------------------------------------------------------
    async def fetch_balance(self) -> dict[str, Any]:
        return await self.call("fetch_balance")

    async def fetch_free_balance(self, asset: str) -> float:
        balance = await self.fetch_balance()
        free = balance.get("free") or {}
        return float(free.get(asset) or 0.0)

    async def check_api_permissions(self) -> ApiPermissionReport:
        if not self.settings.has_api_credentials:
            raise UnsafeApiKeyError("API key belum diisi")
        data = await self.call("sapi_get_account_apirestrictions")
        return ApiPermissionReport.from_response(data or {})

    async def assert_api_key_safe(self) -> ApiPermissionReport:
        """Tolak API key yang punya izin withdraw atau tanpa IP whitelist."""
        report = await self.check_api_permissions()
        for warning in report.warnings:
            log.warning("API key: %s", warning)
        if not report.is_safe:
            raise UnsafeApiKeyError("API key tidak aman: " + "; ".join(report.problems))
        return report
