"""Pemilihan universe 75 coin secara dinamis.

Langkah pemilihan:

1. Ambil semua pair spot /USDT yang aktif (status TRADING).
2. Buang stablecoin, mata uang fiat, token leverage (UP/DOWN/BULL/BEAR),
   token pembungkus (WBTC, WBETH, dan sejenisnya), simbol yang dikecualikan
   lewat EXCLUDED_SYMBOLS, coin ber-tag Monitoring, dan coin yang
   dijadwalkan delisting.
3. Urutkan berdasarkan quoteVolume 24 jam dengan minimum 5 juta USD.
4. Ambil 75 teratas lalu simpan ke data/universe.json beserta tanggalnya.

Jika API gagal, dipakai universe.json terakhir (walau sudah lewat 24 jam).
Jika file itu juga tidak ada, dipakai daftar cadangan di
config/fallback_symbols.py yang tetap divalidasi terhadap load_markets().
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import ccxt

from config.fallback_symbols import FALLBACK_SYMBOLS, SYMBOL_RENAMES
from config.settings import Settings
from core.exchange import ExchangeClient

log = logging.getLogger(__name__)

# Endpoint publik (tidak resmi) yang dipakai halaman Binance untuk menampilkan
# tag coin seperti "Monitoring" dan "Seed". Dipakai secara best effort.
BINANCE_PRODUCTS_URL = "https://www.binance.com/bapi/asset/v2/public/asset-service/product/get-products?includeEtf=true"

STABLECOIN_BASES = frozenset({
    "USDC", "FDUSD", "TUSD", "DAI", "BUSD", "USDP", "PAX", "USDD", "PYUSD", "USD1", "USDE",
    "USDS", "RLUSD", "XUSD", "BFUSD", "SUSD", "LUSD", "GUSD", "HUSD", "FRAX", "UST", "USDJ",
    "USDX", "EURI", "AEUR", "EURC", "EURT",
})
FIAT_BASES = frozenset({
    "EUR", "GBP", "AUD", "TRY", "BRL", "ARS", "MXN", "ZAR", "JPY", "UAH", "RUB", "PLN",
    "RON", "CZK", "COP", "NGN", "IDR", "IDRT", "BIDR", "BVND",
})
# Token pembungkus/staking yang harganya hanya mengikuti BTC, ETH, SOL, atau BNB.
# Dibuang agar eksposur tidak dobel dengan aset aslinya.
WRAPPED_BASES = frozenset({"WBTC", "WBETH", "BETH", "WETH", "STETH", "WSTETH", "CBBTC", "BNSOL", "BTCB"})
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")

# Alasan yang tidak dicatat di laporan karena jumlahnya ribuan dan tidak relevan.
_SILENT_REASONS = frozenset({"bukan_spot", "quote_lain"})


class UniverseError(RuntimeError):
    """Pemilihan universe dinamis gagal."""


@dataclass(frozen=True)
class FilterContext:
    """Data pendukung untuk menilai kelayakan sebuah market."""

    quote: str
    known_bases: frozenset[str] = frozenset()
    excluded_symbols: frozenset[str] = frozenset()
    excluded_tags: frozenset[str] = frozenset()  # huruf kecil
    symbol_tags: Mapping[str, frozenset[str]] = field(default_factory=dict)  # market id -> tag huruf kecil
    delisting_ids: frozenset[str] = frozenset()


def normalize_symbol(raw: str, quote: str) -> str:
    """Ubah 'matic', 'MATICUSDT', atau 'MATIC/USDT' menjadi 'MATIC/USDT'."""
    text = raw.strip().upper()
    if "/" in text:
        return text
    if text.endswith(quote) and len(text) > len(quote):
        return f"{text[: -len(quote)]}/{quote}"
    return f"{text}/{quote}"


def apply_rename(symbol: str) -> str:
    """Petakan ticker lama ke ticker baru, misal MATIC/USDT menjadi POL/USDT."""
    base, _, quote = symbol.partition("/")
    for _ in range(5):  # ikuti rantai rename, dengan batas agar tidak berputar
        new_base = SYMBOL_RENAMES.get(base)
        if new_base is None or new_base == base:
            break
        base = new_base
    return f"{base}/{quote}"


def _market_permissions(info: Mapping[str, Any]) -> set[str]:
    permissions = {str(p) for p in info.get("permissions") or []}
    for permission_set in info.get("permissionSets") or []:
        permissions.update(str(p) for p in permission_set or [])
    return permissions


def is_stablecoin(base: str) -> bool:
    return base in STABLECOIN_BASES or base.startswith("USD") or base.endswith("USD")


def is_leveraged_token(base: str, known_bases: Iterable[str], info: Mapping[str, Any] | None = None) -> bool:
    """Deteksi token leverage seperti BTCUP atau ETHDOWN.

    Hanya dianggap token leverage jika sisa namanya adalah aset yang memang
    terdaftar (BTCUP -> BTC), sehingga coin seperti JUP atau SYRUP aman.
    """
    if info is not None and "LEVERAGED" in _market_permissions(info):
        return True
    known = known_bases if isinstance(known_bases, (set, frozenset)) else set(known_bases)
    for suffix in LEVERAGED_SUFFIXES:
        if base.endswith(suffix):
            underlying = base[: -len(suffix)]
            if len(underlying) >= 2 and underlying in known:
                return True
    return False


def classify_market(market: Mapping[str, Any], ctx: FilterContext) -> str | None:
    """Kembalikan None jika market layak, atau alasan penolakannya."""
    if not market.get("spot", False):
        return "bukan_spot"
    if market.get("quote") != ctx.quote:
        return "quote_lain"
    info = market.get("info") or {}
    status = info.get("status")
    if not market.get("active", False) or (status is not None and status != "TRADING"):
        return "tidak_aktif"
    if info.get("isSpotTradingAllowed") is False:
        return "tidak_aktif"
    base = str(market.get("base", ""))
    if is_stablecoin(base):
        return "stablecoin"
    if base in FIAT_BASES:
        return "fiat"
    if is_leveraged_token(base, ctx.known_bases, info):
        return "token_leverage"
    if base in WRAPPED_BASES:
        return "token_pembungkus"
    symbol = str(market.get("symbol", ""))
    if symbol in ctx.excluded_symbols:
        return "dikecualikan_user"
    market_id = str(market.get("id", ""))
    if market_id in ctx.delisting_ids:
        return "jadwal_delisting"
    tags = ctx.symbol_tags.get(market_id, frozenset()) & ctx.excluded_tags
    if tags:
        return "tag_" + sorted(tags)[0]
    return None


def quote_volume(ticker: Mapping[str, Any]) -> float:
    """Volume 24 jam dalam quote (USDT). Dihitung dari baseVolume jika kosong."""
    value = ticker.get("quoteVolume")
    if value is not None:
        return float(value)
    base_volume = ticker.get("baseVolume")
    price = ticker.get("vwap") or ticker.get("last") or ticker.get("close")
    if base_volume is None or price is None:
        return 0.0
    return float(base_volume) * float(price)


def looks_like_usd_peg(ticker: Mapping[str, Any]) -> bool:
    """Deteksi stablecoin baru yang belum ada di daftar: harga ~1 USD dan hampir tidak bergerak."""
    last, high, low = ticker.get("last"), ticker.get("high"), ticker.get("low")
    if not last or high is None or low is None:
        return False
    last, high, low = float(last), float(high), float(low)
    return abs(last - 1.0) <= 0.02 and (high - low) / last <= 0.01


@dataclass(frozen=True)
class Universe:
    """Hasil pemilihan coin beserta asal usul dan alasan pengecualian."""

    symbols: list[str]
    generated_at: datetime
    source: str  # "binance" atau "fallback"
    details: list[dict[str, Any]] = field(default_factory=list)
    excluded: dict[str, list[str]] = field(default_factory=dict)
    validated: bool = True
    from_cache: bool = False

    def age(self, now: datetime) -> timedelta:
        return now - self.generated_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "source": self.source,
            "count": len(self.symbols),
            "validated": self.validated,
            "symbols": self.symbols,
            "details": self.details,
            "excluded": self.excluded,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Universe:
        generated_at = datetime.fromisoformat(str(data["generated_at"]))
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=timezone.utc)
        symbols = [str(s) for s in data["symbols"]]
        if not symbols:
            raise ValueError("daftar simbol kosong")
        return cls(
            symbols=symbols,
            generated_at=generated_at,
            source=str(data.get("source", "binance")),
            details=list(data.get("details") or []),
            excluded={str(k): list(v) for k, v in (data.get("excluded") or {}).items()},
            validated=bool(data.get("validated", True)),
        )


def validate_symbols(
    symbols: Iterable[str], markets: Mapping[str, Mapping[str, Any]], ctx: FilterContext
) -> tuple[list[str], dict[str, str], list[str]]:
    """Validasi daftar simbol terhadap load_markets().

    Mengembalikan (simbol valid, peta rename yang diterapkan, simbol yang dibuang).
    """
    valid: list[str] = []
    renamed: dict[str, str] = {}
    dropped: list[str] = []
    seen: set[str] = set()
    for raw in symbols:
        symbol = normalize_symbol(raw, ctx.quote)
        target = apply_rename(symbol)
        market = markets.get(target)
        if market is None or classify_market(market, ctx) is not None:
            dropped.append(symbol)
            continue
        if target != symbol:
            renamed[symbol] = target
        if target not in seen:
            seen.add(target)
            valid.append(target)
    return valid, renamed, dropped


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UniverseSelector:
    """Mengelola universe coin: cache file, refresh 24 jam, dan fallback."""

    def __init__(
        self,
        client: ExchangeClient,
        settings: Settings,
        *,
        path: Path | None = None,
        clock: Callable[[], datetime] = _utcnow,
        failure_backoff: timedelta = timedelta(minutes=30),
    ) -> None:
        self.client = client
        self.settings = settings
        self.path = path or settings.universe_file
        self._clock = clock
        self._failure_backoff = failure_backoff
        self._next_attempt_at: datetime | None = None
        self._current: Universe | None = None

    @property
    def current(self) -> Universe | None:
        return self._current

    # ------------------------------------------------------------------
    # Cache file
    # ------------------------------------------------------------------
    def load_cached(self) -> Universe | None:
        if not self.path.is_file():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return replace(Universe.from_dict(data), from_cache=True)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.warning("File universe %s rusak, diabaikan: %s", self.path, exc)
            return None

    def save(self, universe: Universe) -> None:
        """Tulis secara atomik (file sementara lalu rename) agar tidak setengah jadi."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(self.path.name + ".tmp")
        tmp_path.write_text(json.dumps(universe.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp_path, self.path)

    def is_fresh(self, universe: Universe) -> bool:
        return universe.source != "fallback" and universe.age(self._clock()) < timedelta(hours=self.settings.universe_refresh_hours)

    def needs_refresh(self) -> bool:
        universe = self._current or self.load_cached()
        if universe is not None and self.is_fresh(universe):
            return False
        return self._next_attempt_at is None or self._clock() >= self._next_attempt_at

    # ------------------------------------------------------------------
    # Alur utama
    # ------------------------------------------------------------------
    async def get_universe(self, force_refresh: bool = False) -> Universe:
        """Universe siap pakai: cache jika masih segar, selain itu bangun ulang."""
        cached = self.load_cached()
        if not force_refresh:
            if cached is not None and self.is_fresh(cached):
                self._current = self._revalidate(cached)
                return self._current
            if self._next_attempt_at is not None and self._clock() < self._next_attempt_at and self._current is not None:
                return self._current

        try:
            universe = await self.build()
        except Exception as exc:  # noqa: BLE001  # semua kegagalan API diarahkan ke cadangan
            self._next_attempt_at = self._clock() + self._failure_backoff
            # Traceback hanya untuk error tak terduga (kemungkinan bug), bukan gangguan API.
            expected = isinstance(exc, (ccxt.BaseError, UniverseError))
            log.error("Gagal memilih universe dari Binance: %s", exc, exc_info=not expected)
            if cached is not None:
                hours = cached.age(self._clock()).total_seconds() / 3600
                log.warning("Memakai universe terakhir dari %s (umur %.1f jam)", self.path, hours)
                self._current = self._revalidate(cached)
            else:
                self._current = await self.build_fallback()
            return self._current

        self._next_attempt_at = None
        self.save(universe)
        self._current = universe
        return universe

    async def build(self) -> Universe:
        """Pilih coin secara dinamis dari data Binance."""
        markets = await self.client.load_markets(reload=True)
        if not markets:
            raise UniverseError("load_markets tidak mengembalikan market")
        symbol_tags = await self._fetch_symbol_tags()
        delisting_ids = await self._fetch_delisting_ids()
        tickers = await self.client.fetch_tickers()
        if not tickers:
            raise UniverseError("fetch_tickers tidak mengembalikan data")

        ctx = self._context(markets, symbol_tags, delisting_ids)
        excluded: dict[str, list[str]] = defaultdict(list)
        candidates: list[tuple[str, float]] = []
        for symbol in sorted(markets):
            market = markets[symbol]
            reason = classify_market(market, ctx)
            if reason is None:
                ticker = tickers.get(symbol)
                if ticker is None:
                    reason = "tanpa_ticker"
                elif looks_like_usd_peg(ticker):
                    reason = "mirip_stablecoin"
                elif quote_volume(ticker) < self.settings.min_quote_volume_usd:
                    reason = "volume_rendah"
                else:
                    candidates.append((symbol, quote_volume(ticker)))
                    continue
            if reason not in _SILENT_REASONS:
                excluded[reason].append(symbol)

        if not candidates:
            raise UniverseError("tidak ada coin yang lolos filter")
        candidates.sort(key=lambda item: item[1], reverse=True)
        selected = candidates[: self.settings.universe_size]
        if len(selected) < self.settings.universe_size:
            log.warning(
                "Hanya %d coin yang memenuhi volume minimal $%s (target %d)",
                len(selected), f"{self.settings.min_quote_volume_usd:,.0f}", self.settings.universe_size,
            )
        log.info("Universe baru: %d coin, teratas %s", len(selected), ", ".join(s for s, _ in selected[:5]))
        return Universe(
            symbols=[symbol for symbol, _ in selected],
            generated_at=self._clock(),
            source="binance",
            details=[{"symbol": symbol, "quote_volume": round(volume, 2)} for symbol, volume in selected],
            excluded=dict(excluded),
        )

    async def build_fallback(self) -> Universe:
        """Universe dari daftar cadangan, divalidasi terhadap load_markets() bila bisa."""
        markets = self.client.markets
        if not markets:
            try:
                markets = await self.client.load_markets()
            except Exception as exc:  # noqa: BLE001
                log.error("load_markets juga gagal, daftar cadangan dipakai tanpa validasi: %s", exc)
                markets = {}

        size = self.settings.universe_size
        if markets:
            symbols, renamed, dropped = validate_symbols(FALLBACK_SYMBOLS, markets, self._context(markets))
            for old, new in renamed.items():
                log.info("Simbol cadangan %s sudah berganti nama menjadi %s", old, new)
            if dropped:
                log.warning("Simbol cadangan tidak valid dan dibuang: %s", ", ".join(dropped))
            excluded = {"tidak_valid": dropped} if dropped else {}
            universe = Universe(symbols[:size], self._clock(), "fallback", excluded=excluded, validated=True)
        else:
            symbols = list(dict.fromkeys(apply_rename(s) for s in FALLBACK_SYMBOLS))
            universe = Universe(symbols[:size], self._clock(), "fallback", validated=False)
        log.warning("Memakai daftar coin cadangan (%d coin)", len(universe.symbols))
        return universe

    # ------------------------------------------------------------------
    # Pendukung
    # ------------------------------------------------------------------
    def _context(
        self,
        markets: Mapping[str, Mapping[str, Any]],
        symbol_tags: Mapping[str, frozenset[str]] | None = None,
        delisting_ids: Iterable[str] = (),
    ) -> FilterContext:
        quote = self.settings.quote_asset
        return FilterContext(
            quote=quote,
            known_bases=frozenset(str(m.get("base", "")) for m in markets.values()),
            excluded_symbols=frozenset(normalize_symbol(s, quote) for s in self.settings.excluded_symbols),
            excluded_tags=frozenset(tag.lower() for tag in self.settings.excluded_tags),
            symbol_tags=symbol_tags or {},
            delisting_ids=frozenset(delisting_ids),
        )

    def _revalidate(self, universe: Universe) -> Universe:
        """Buang simbol cache yang sudah tidak aktif, jika data market sudah dimuat."""
        markets = self.client.markets
        if not markets:
            return universe
        valid, _, dropped = validate_symbols(universe.symbols, markets, self._context(markets))
        if not dropped:
            return universe
        log.warning("Simbol di universe cache sudah tidak aktif dan dibuang: %s", ", ".join(dropped))
        return replace(universe, symbols=valid)

    async def _fetch_symbol_tags(self) -> dict[str, frozenset[str]]:
        if not self.settings.excluded_tags or self.settings.binance_testnet:
            return {}
        try:
            payload = await self.client.fetch_json(BINANCE_PRODUCTS_URL, retries=1)
        except Exception as exc:  # noqa: BLE001  # endpoint tidak resmi, jangan gagalkan universe
            log.warning("Data tag coin (Monitoring) tidak bisa diambil, filter tag dilewati: %s", exc)
            return {}
        rows = payload.get("data") if isinstance(payload, Mapping) else None
        tags: dict[str, frozenset[str]] = {}
        for row in rows or []:
            if isinstance(row, Mapping) and row.get("s"):
                tags[str(row["s"])] = frozenset(str(tag).lower() for tag in row.get("tags") or [])
        return tags

    async def _fetch_delisting_ids(self) -> set[str]:
        try:
            return await self.client.fetch_delisting_ids()
        except Exception as exc:  # noqa: BLE001
            log.warning("Jadwal delisting tidak bisa diambil, dilewati: %s", exc)
            return set()
