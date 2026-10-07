"""Data feed OHLCV multi timeframe dengan cache.

Keputusan desain:

* Hanya candle yang SUDAH TERTUTUP yang dipakai. Candle yang masih berjalan
  dibuang supaya sinyal tidak berubah ubah (repaint) dan perilaku live sama
  persis dengan backtest.
* Cache per (simbol, timeframe). Sebuah timeframe baru di-fetch ulang setelah
  candle barunya tertutup, jadi candle 1h cukup diperbarui sekali per jam
  walaupun scan berjalan tiap 5 menit.
* Update inkremental: hanya candle baru yang diminta (parameter since) lalu
  digabung dengan cache, sehingga hemat weight API.
* Fetch paralel dengan asyncio.gather. Batas konkurensi (Semaphore) dan
  exponential backoff untuk HTTP 429 ditangani oleh ExchangeClient.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config.settings import TIMEFRAME_MS, Settings
from core.exchange import ExchangeClient

log = logging.getLogger(__name__)

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
BINANCE_MAX_KLINES = 1000


def timeframe_to_ms(timeframe: str) -> int:
    try:
        return TIMEFRAME_MS[timeframe]
    except KeyError:
        raise ValueError(f"timeframe tidak didukung: {timeframe}") from None


def last_closed_open_ms(now_ms: int, tf_ms: int, grace_ms: int = 0) -> int:
    """Open time candle terakhir yang sudah tertutup dan melewati masa grace.

    Contoh timeframe 5m, grace 2 detik: pada 12:00:03 hasilnya 11:55 (candle
    11:55 tutup pukul 12:00:00), sedangkan pada 12:00:01 hasilnya masih 11:50.
    """
    return ((now_ms - grace_ms) // tf_ms) * tf_ms - tf_ms


def empty_ohlcv_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    return pd.DataFrame({col: pd.Series(dtype="float64") for col in OHLCV_COLUMNS}, index=index)


def ohlcv_to_frame(rows: Sequence[Sequence[float]]) -> pd.DataFrame:
    """Ubah list OHLCV ccxt menjadi DataFrame berindeks waktu UTC (open time)."""
    if not rows:
        return empty_ohlcv_frame()
    frame = pd.DataFrame([list(row[:6]) for row in rows], columns=["timestamp", *OHLCV_COLUMNS])
    frame = frame.dropna()
    frame["timestamp"] = frame["timestamp"].astype("int64")
    frame = frame.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    for col in OHLCV_COLUMNS:
        frame[col] = frame[col].astype("float64")
    timestamps = frame.pop("timestamp")
    frame.index = pd.DatetimeIndex(pd.to_datetime(timestamps, unit="ms", utc=True), name="timestamp")
    return frame


def frame_open_ms(frame: pd.DataFrame) -> np.ndarray:
    """Open time tiap candle dalam milidetik (int64)."""
    return frame.index.as_unit("ms").asi8


def drop_unclosed(frame: pd.DataFrame, tf_ms: int, now_ms: int, grace_ms: int = 0) -> pd.DataFrame:
    cutoff = last_closed_open_ms(now_ms, tf_ms, grace_ms)
    return frame[frame_open_ms(frame) <= cutoff]


@dataclass
class CacheEntry:
    frame: pd.DataFrame
    last_open_ms: int | None  # open time candle tertutup terakhir di cache
    fetched_at_ms: int


@dataclass
class FeedStats:
    api_calls: int = 0
    cache_hits: int = 0
    full_fetches: int = 0
    incremental_fetches: int = 0
    errors: int = 0


@dataclass
class RefreshResult:
    """Hasil refresh banyak simbol. Simbol yang gagal tidak dimasukkan ke data.

    `stale` berisi (simbol, timeframe) yang datanya ada tetapi belum memuat
    candle tertutup terbaru (bursa telat merilis candle).
    """

    data: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    errors: dict[tuple[str, str], Exception] = field(default_factory=dict)
    stale: set[tuple[str, str]] = field(default_factory=set)

    def complete_symbols(self, timeframes: Iterable[str]) -> list[str]:
        """Simbol dengan data lengkap dan terbaru untuk semua timeframe (siap dianalisis)."""
        required = set(timeframes)
        stale_symbols = {symbol for symbol, _ in self.stale}
        return [
            symbol for symbol, frames in self.data.items()
            if required <= frames.keys() and symbol not in stale_symbols
        ]


class DataFeed:
    """Mengambil dan menyimpan cache candle tertutup untuk banyak simbol dan timeframe."""

    def __init__(
        self,
        client: ExchangeClient,
        timeframes: Sequence[str],
        limit: int = 300,
        *,
        grace_ms: int = 2_000,
        min_refetch_ms: int = 15_000,
        clock: Callable[[], int] | None = None,
    ) -> None:
        for timeframe in timeframes:
            timeframe_to_ms(timeframe)
        if not 1 <= limit <= BINANCE_MAX_KLINES:
            raise ValueError(f"limit harus antara 1 dan {BINANCE_MAX_KLINES}")
        self.client = client
        self.timeframes = tuple(timeframes)
        self.limit = limit
        self.grace_ms = grace_ms
        self.min_refetch_ms = min_refetch_ms
        self._clock = clock or client.now_ms
        self._cache: dict[tuple[str, str], CacheEntry] = {}
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self.stats = FeedStats()

    @classmethod
    def from_settings(cls, client: ExchangeClient, settings: Settings) -> DataFeed:
        return cls(client, settings.timeframes, settings.ohlcv_limit, grace_ms=settings.candle_close_grace_ms)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    def cached(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        entry = self._cache.get((symbol, timeframe))
        return entry.frame if entry is not None else None

    def is_current(self, symbol: str, timeframe: str, now_ms: int | None = None) -> bool:
        """True jika cache sudah memuat candle tertutup terbaru."""
        entry = self._cache.get((symbol, timeframe))
        if entry is None or entry.last_open_ms is None:
            return False
        now = self._clock() if now_ms is None else now_ms
        return entry.last_open_ms >= last_closed_open_ms(now, timeframe_to_ms(timeframe), self.grace_ms)

    def needs_update(self, symbol: str, timeframe: str, now_ms: int | None = None) -> bool:
        """True jika ada candle baru yang sudah tertutup tetapi belum ada di cache."""
        entry = self._cache.get((symbol, timeframe))
        if entry is None:
            return True
        now = self._clock() if now_ms is None else now_ms
        if self.is_current(symbol, timeframe, now):
            return False
        # Data tertinggal (candle baru tutup, atau bursa telat merilis candle):
        # fetch ulang, tetapi dengan jeda minimal supaya tidak membanjiri API.
        return now - entry.fetched_at_ms >= self.min_refetch_ms

    def prune(self, keep_symbols: Iterable[str]) -> int:
        """Hapus cache simbol yang sudah keluar dari universe. Mengembalikan jumlah entri terhapus."""
        keep = set(keep_symbols)
        stale_keys = [key for key in self._cache if key[0] not in keep]
        for key in stale_keys:
            del self._cache[key]
            self._locks.pop(key, None)
        return len(stale_keys)

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------
    async def get_ohlcv(self, symbol: str, timeframe: str, *, force: bool = False) -> pd.DataFrame:
        """Candle tertutup terbaru (maksimal `limit` baris) untuk satu simbol dan timeframe."""
        key = (symbol, timeframe)
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            now = self._clock()
            entry = self._cache.get(key)
            if entry is not None and not force and not self.needs_update(symbol, timeframe, now):
                self.stats.cache_hits += 1
                return entry.frame

            tf_ms = timeframe_to_ms(timeframe)
            expected = last_closed_open_ms(now, tf_ms, self.grace_ms)
            try:
                if entry is None or force or entry.last_open_ms is None or (expected - entry.last_open_ms) // tf_ms >= self.limit:
                    frame = await self._fetch_full(symbol, timeframe, now)
                    self.stats.full_fetches += 1
                else:
                    frame = await self._fetch_incremental(symbol, timeframe, entry.frame, entry.last_open_ms, now)
                    self.stats.incremental_fetches += 1
            except Exception:
                self.stats.errors += 1
                raise

            last_open = int(frame_open_ms(frame)[-1]) if len(frame) else None
            self._cache[key] = CacheEntry(frame=frame, last_open_ms=last_open, fetched_at_ms=now)
            return frame

    async def _fetch_full(self, symbol: str, timeframe: str, now: int) -> pd.DataFrame:
        # +2: satu candle berjalan dan satu candle yang mungkin masih dalam masa grace
        # (dibatasi 1000, batas satu request Binance).
        rows = await self.client.fetch_ohlcv(symbol, timeframe, limit=min(self.limit + 2, BINANCE_MAX_KLINES))
        self.stats.api_calls += 1
        frame = drop_unclosed(ohlcv_to_frame(rows), timeframe_to_ms(timeframe), now, self.grace_ms)
        return frame.iloc[-self.limit :]

    async def _fetch_incremental(
        self, symbol: str, timeframe: str, cached: pd.DataFrame, last_open_ms: int, now: int
    ) -> pd.DataFrame:
        tf_ms = timeframe_to_ms(timeframe)
        missing = (last_closed_open_ms(now, tf_ms, self.grace_ms) - last_open_ms) // tf_ms
        rows = await self.client.fetch_ohlcv(
            symbol, timeframe, since=last_open_ms + tf_ms, limit=min(missing + 2, BINANCE_MAX_KLINES)
        )
        self.stats.api_calls += 1
        new = drop_unclosed(ohlcv_to_frame(rows), tf_ms, now, self.grace_ms)
        if new.empty:
            return cached
        merged = pd.concat([cached, new])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        return merged.iloc[-self.limit :]

    async def get_multi_timeframe(self, symbol: str) -> dict[str, pd.DataFrame]:
        frames = await asyncio.gather(*(self.get_ohlcv(symbol, tf) for tf in self.timeframes))
        return dict(zip(self.timeframes, frames, strict=True))

    async def refresh(self, symbols: Iterable[str], timeframes: Sequence[str] | None = None) -> RefreshResult:
        """Perbarui semua (simbol, timeframe) secara paralel.

        Kegagalan satu simbol tidak menghentikan simbol lain. Simbol yang gagal
        tidak dimasukkan ke hasil (data basi tidak dipakai untuk sinyal),
        tetapi cache lamanya tetap disimpan untuk update inkremental berikutnya.
        """
        tfs = tuple(timeframes or self.timeframes)
        keys = [(symbol, tf) for symbol in dict.fromkeys(symbols) for tf in tfs]
        results = await asyncio.gather(*(self.get_ohlcv(symbol, tf) for symbol, tf in keys), return_exceptions=True)

        now = self._clock()
        data: dict[str, dict[str, pd.DataFrame]] = defaultdict(dict)
        errors: dict[tuple[str, str], Exception] = {}
        stale: set[tuple[str, str]] = set()
        for (symbol, tf), result in zip(keys, results, strict=True):
            if isinstance(result, BaseException):
                if not isinstance(result, Exception):
                    raise result  # CancelledError / KeyboardInterrupt tidak boleh ditelan
                errors[(symbol, tf)] = result
                log.warning("Gagal mengambil candle %s %s: %s", symbol, tf, result)
                continue
            data[symbol][tf] = result
            if not self.is_current(symbol, tf, now):
                stale.add((symbol, tf))
        return RefreshResult(data=dict(data), errors=errors, stale=stale)

    async def fetch_history(
        self, symbol: str, timeframe: str, start_ms: int, end_ms: int | None = None, page_limit: int = BINANCE_MAX_KLINES
    ) -> pd.DataFrame:
        """Ambil candle tertutup pada rentang [start_ms, end_ms) dengan paginasi.

        Dipakai untuk pemanasan indikator dan pengunduhan data backtest.
        Hasil tidak masuk cache.
        """
        tf_ms = timeframe_to_ms(timeframe)
        now = self._clock()
        end = now if end_ms is None else min(end_ms, now)
        frames: list[pd.DataFrame] = []
        since = start_ms
        while since < end:
            rows = await self.client.fetch_ohlcv(symbol, timeframe, since=since, limit=page_limit)
            self.stats.api_calls += 1
            if not rows:
                break
            frames.append(ohlcv_to_frame(rows))
            next_since = int(rows[-1][0]) + tf_ms
            if next_since <= since:
                break  # pengaman: bursa mengembalikan data lama, hentikan agar tidak berputar
            since = next_since

        if not frames:
            return empty_ohlcv_frame()
        merged = pd.concat(frames)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        opens = frame_open_ms(merged)
        mask = (opens >= start_ms) & (opens < end) & (opens <= last_closed_open_ms(now, tf_ms, self.grace_ms))
        return merged[mask]
