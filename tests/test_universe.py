"""Test pemilihan universe coin dinamis, cache 24 jam, dan fallback."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import ccxt
import pytest
from helpers import install_markets, make_market, make_settings, make_ticker

from config.fallback_symbols import FALLBACK_SYMBOLS, SYMBOL_RENAMES
from core.exchange import ExchangeClient
from core.universe import (
    BINANCE_PRODUCTS_URL,
    FilterContext,
    Universe,
    UniverseSelector,
    apply_rename,
    classify_market,
    is_leveraged_token,
    is_stablecoin,
    looks_like_usd_peg,
    normalize_symbol,
    quote_volume,
    validate_symbols,
)

NOW = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)
FAKE_KEY = {"binance_api_key": "kunci-palsu-123", "binance_api_secret": "rahasia-palsu-456"}


class MutableClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def build_market_data():
    """100 coin normal + kasus khusus yang harus disaring."""
    markets, tickers = {}, {}

    def add(base, volume, quote="USDT", last=10.0, range_pct=0.08, ticker=True, **market_kwargs):
        market = make_market(base, quote, **market_kwargs)
        markets[market["symbol"]] = market
        if ticker:
            tickers[market["symbol"]] = make_ticker(market["symbol"], volume, last=last, range_pct=range_pct)

    add("BTC", 5_000_000_000, last=60_000)
    add("ETH", 3_000_000_000, last=3_000)
    for i in range(100):  # C000 volume terbesar, C099 terkecil, semuanya > 5 juta
        add(f"C{i:03d}", 900_000_000 - i * 8_000_000)
    add("JUP", 950_000_000, last=0.5)       # berakhiran UP tapi bukan token leverage
    add("SYRUP", 940_000_000, last=0.4)     # berakhiran UP tapi bukan token leverage
    add("USDC", 9_000_000_000, last=1.0, range_pct=0.001)
    add("FDUSD", 8_000_000_000, last=1.0, range_pct=0.001)
    add("USD1", 7_000_000_000, last=1.0, range_pct=0.001)
    add("NEWPEG", 6_000_000_000, last=1.0002, range_pct=0.002)  # stablecoin baru yang belum terdaftar
    add("EUR", 2_000_000_000, last=1.1, range_pct=0.005)
    add("BTCUP", 2_000_000_000, last=5)
    add("ETHDOWN", 2_000_000_000, last=5)
    add("WBTC", 2_000_000_000, last=60_000)
    add("OLD", 2_000_000_000, active=False, status="BREAK")
    add("MON", 2_000_000_000)                # ber-tag Monitoring
    add("DEL", 2_000_000_000)                # dijadwalkan delisting
    add("LOW", 4_900_000)                    # di bawah 5 juta
    add("NOTICK", 2_000_000_000, ticker=False)
    add("FUT", 2_000_000_000, spot=False)    # bukan spot
    add("SOL", 2_000_000_000, quote="BTC")   # quote bukan USDT
    return markets, tickers


@pytest.fixture
def market_data(mock_exchange):
    markets, tickers = build_market_data()
    install_markets(mock_exchange, markets)
    mock_exchange.fetch_tickers.return_value = tickers
    mock_exchange.fetch.return_value = {
        "data": [{"s": "MONUSDT", "tags": ["Monitoring"]}, {"s": "BTCUSDT", "tags": ["pow"]}],
        "success": True,
    }
    mock_exchange.sapi_get_spot_delist_schedule.return_value = [{"delistTime": 1, "symbols": ["DELUSDT"]}]
    return markets, tickers


def make_selector(settings, mock_exchange, fake_time, clock=None):
    client = ExchangeClient(settings, mock_exchange, sleep=fake_time.sleep, monotonic=fake_time.monotonic, jitter=0)
    return UniverseSelector(client, settings, clock=clock or MutableClock(NOW))


async def test_memilih_75_coin_teratas_berdasarkan_volume(tmp_path, mock_exchange, fake_time, market_data):
    settings = make_settings(tmp_path, **FAKE_KEY)
    selector = make_selector(settings, mock_exchange, fake_time)
    universe = await selector.get_universe()

    assert len(universe.symbols) == 75
    assert universe.source == "binance" and not universe.from_cache
    assert universe.symbols[:5] == ["BTC/USDT", "ETH/USDT", "JUP/USDT", "SYRUP/USDT", "C000/USDT"]
    volumes = [row["quote_volume"] for row in universe.details]
    assert volumes == sorted(volumes, reverse=True)
    assert min(volumes) >= 5_000_000

    excluded = universe.excluded
    assert set(excluded["stablecoin"]) == {"USDC/USDT", "FDUSD/USDT", "USD1/USDT"}
    assert excluded["mirip_stablecoin"] == ["NEWPEG/USDT"]
    assert excluded["fiat"] == ["EUR/USDT"]
    assert set(excluded["token_leverage"]) == {"BTCUP/USDT", "ETHDOWN/USDT"}
    assert excluded["token_pembungkus"] == ["WBTC/USDT"]
    assert excluded["tidak_aktif"] == ["OLD/USDT"]
    assert excluded["tag_monitoring"] == ["MON/USDT"]
    assert excluded["jadwal_delisting"] == ["DEL/USDT"]
    assert excluded["volume_rendah"] == ["LOW/USDT"]
    assert excluded["tanpa_ticker"] == ["NOTICK/USDT"]
    assert "FUT/USDT" not in str(excluded) and "SOL/BTC" not in str(excluded)


async def test_universe_disimpan_ke_json_beserta_tanggal(settings, mock_exchange, fake_time, market_data):
    selector = make_selector(settings, mock_exchange, fake_time)
    universe = await selector.get_universe()
    saved = json.loads(settings.universe_file.read_text(encoding="utf-8"))
    assert saved["generated_at"] == NOW.isoformat()
    assert saved["symbols"] == universe.symbols
    assert saved["count"] == 75
    assert not settings.universe_file.with_name("universe.json.tmp").exists()


async def test_cache_dipakai_selama_24_jam_lalu_diperbarui(settings, mock_exchange, fake_time, market_data):
    clock = MutableClock(NOW)
    selector = make_selector(settings, mock_exchange, fake_time, clock)
    await selector.get_universe()
    assert mock_exchange.fetch_tickers.await_count == 1

    clock.now = NOW + timedelta(hours=23)
    cached = await selector.get_universe()
    assert cached.from_cache
    assert mock_exchange.fetch_tickers.await_count == 1  # tidak ada request baru
    assert not selector.needs_refresh()

    clock.now = NOW + timedelta(hours=24, minutes=1)
    assert selector.needs_refresh()
    fresh = await selector.get_universe()
    assert not fresh.from_cache
    assert mock_exchange.fetch_tickers.await_count == 2


async def test_simbol_dikecualikan_user(tmp_path, mock_exchange, fake_time, market_data):
    settings = make_settings(tmp_path, excluded_symbols="C000,c001/usdt,JUPUSDT")
    universe = await make_selector(settings, mock_exchange, fake_time).get_universe()
    assert set(universe.excluded["dikecualikan_user"]) == {"C000/USDT", "C001/USDT", "JUP/USDT"}
    assert "C000/USDT" not in universe.symbols


async def test_tag_monitoring_gagal_diambil_tidak_menggagalkan_universe(settings, mock_exchange, fake_time, market_data):
    mock_exchange.fetch.side_effect = ccxt.ExchangeNotAvailable("binance bapi down")
    universe = await make_selector(settings, mock_exchange, fake_time).get_universe()
    assert len(universe.symbols) == 75
    assert "tag_monitoring" not in universe.excluded
    assert mock_exchange.fetch.await_args.args[0] == BINANCE_PRODUCTS_URL


async def test_api_gagal_memakai_cache_lama(settings, mock_exchange, fake_time, market_data):
    stale = Universe(["BTC/USDT", "ETH/USDT"], NOW - timedelta(hours=30), "binance")
    settings.universe_file.parent.mkdir(parents=True)
    settings.universe_file.write_text(json.dumps(stale.to_dict()), encoding="utf-8")
    mock_exchange.fetch_tickers.side_effect = ccxt.NetworkError("binance tidak bisa dihubungi")

    universe = await make_selector(settings, mock_exchange, fake_time).get_universe()
    assert universe.from_cache
    assert universe.symbols == ["BTC/USDT", "ETH/USDT"]


async def test_api_gagal_tanpa_cache_memakai_fallback_tervalidasi(settings, mock_exchange, fake_time):
    # Market hanya berisi sebagian simbol cadangan; sisanya harus dibuang.
    markets = {s: make_market(s.split("/")[0]) for s in FALLBACK_SYMBOLS[:70]}
    install_markets(mock_exchange, markets)
    mock_exchange.fetch_tickers.side_effect = ccxt.NetworkError("binance tidak bisa dihubungi")

    universe = await make_selector(settings, mock_exchange, fake_time).get_universe()
    assert universe.source == "fallback" and universe.validated
    assert universe.symbols == list(FALLBACK_SYMBOLS[:70])
    assert universe.excluded["tidak_valid"] == list(FALLBACK_SYMBOLS[70:])
    assert not settings.universe_file.exists()  # fallback tidak disimpan sebagai cache


async def test_semua_api_gagal_fallback_tanpa_validasi(settings, mock_exchange, fake_time):
    mock_exchange.load_markets.side_effect = ccxt.NetworkError("binance tidak bisa dihubungi")
    universe = await make_selector(settings, mock_exchange, fake_time).get_universe()
    assert universe.source == "fallback"
    assert universe.validated is False
    assert len(universe.symbols) == 75


async def test_setelah_gagal_tidak_langsung_mencoba_lagi(settings, mock_exchange, fake_time, market_data):
    clock = MutableClock(NOW)
    selector = make_selector(settings, mock_exchange, fake_time, clock)
    mock_exchange.fetch_tickers.side_effect = ccxt.NetworkError("binance tidak bisa dihubungi")
    await selector.get_universe()
    calls = mock_exchange.fetch_tickers.await_count

    clock.now = NOW + timedelta(minutes=10)
    await selector.get_universe()
    assert mock_exchange.fetch_tickers.await_count == calls  # masih dalam jeda 30 menit

    clock.now = NOW + timedelta(minutes=31)
    mock_exchange.fetch_tickers.side_effect = None
    universe = await selector.get_universe()
    assert universe.source == "binance"


async def test_cache_divalidasi_ulang_jika_market_sudah_dimuat(settings, mock_exchange, fake_time, market_data):
    markets, _ = market_data
    cached = Universe(["BTC/USDT", "OLD/USDT", "GONE/USDT"], NOW - timedelta(hours=1), "binance")
    settings.universe_file.parent.mkdir(parents=True)
    settings.universe_file.write_text(json.dumps(cached.to_dict()), encoding="utf-8")
    mock_exchange.markets = markets
    universe = await make_selector(settings, mock_exchange, fake_time).get_universe()
    assert universe.symbols == ["BTC/USDT"]


def test_file_universe_rusak_diabaikan(settings, mock_exchange, fake_time):
    settings.universe_file.parent.mkdir(parents=True)
    settings.universe_file.write_text("{bukan json", encoding="utf-8")
    assert make_selector(settings, mock_exchange, fake_time).load_cached() is None


# ----------------------------------------------------------------------
# Fungsi murni
# ----------------------------------------------------------------------
def test_validasi_simbol_dan_rename_ticker():
    markets = {m["symbol"]: m for m in (make_market(b) for b in ("POL", "S", "RENDER", "BTC", "FET"))}
    ctx = FilterContext(quote="USDT", known_bases=frozenset({"POL", "S", "RENDER", "BTC", "FET"}))
    valid, renamed, dropped = validate_symbols(
        ["MATIC/USDT", "FTM", "rndr", "BTCUSDT", "AGIX/USDT", "OCEAN/USDT", "XYZ/USDT"], markets, ctx
    )
    assert valid == ["POL/USDT", "S/USDT", "RENDER/USDT", "BTC/USDT", "FET/USDT"]
    assert renamed == {
        "MATIC/USDT": "POL/USDT",
        "FTM/USDT": "S/USDT",
        "RNDR/USDT": "RENDER/USDT",
        "AGIX/USDT": "FET/USDT",
        "OCEAN/USDT": "FET/USDT",
    }
    assert dropped == ["XYZ/USDT"]


def test_daftar_fallback_75_coin_valid():
    assert len(FALLBACK_SYMBOLS) == 75
    assert len(set(FALLBACK_SYMBOLS)) == 75
    for symbol in FALLBACK_SYMBOLS:
        base, quote = symbol.split("/")
        assert quote == "USDT"
        assert not is_stablecoin(base), symbol
        assert base not in SYMBOL_RENAMES, f"{symbol} memakai ticker lama"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("matic", "MATIC/USDT"), ("MATICUSDT", "MATIC/USDT"), ("matic/usdt", "MATIC/USDT"), (" sol ", "SOL/USDT")],
)
def test_normalize_symbol(raw, expected):
    assert normalize_symbol(raw, "USDT") == expected


def test_apply_rename():
    assert apply_rename("MATIC/USDT") == "POL/USDT"
    assert apply_rename("FTM/USDT") == "S/USDT"
    assert apply_rename("BTC/USDT") == "BTC/USDT"


@pytest.mark.parametrize(
    ("base", "expected"),
    [("BTCUP", True), ("ETHDOWN", True), ("BNBBULL", True), ("XRPBEAR", True), ("JUP", False), ("SYRUP", False), ("SETUP", False)],
)
def test_deteksi_token_leverage(base, expected):
    # JUP, SYRUP, SETUP: sisa nama (J, SYR, SET) bukan aset terdaftar, jadi bukan token leverage.
    assert is_leveraged_token(base, {"BTC", "ETH", "BNB", "XRP", "JUP"}) is expected


def test_token_leverage_dari_permission_binance():
    assert is_leveraged_token("ABCXYZ", set(), {"permissions": ["LEVERAGED"]})
    assert is_leveraged_token("ABCXYZ", set(), {"permissionSets": [["SPOT", "LEVERAGED"]]})


def test_classify_market_quote_dan_status():
    ctx = FilterContext(quote="USDT")
    assert classify_market(make_market("SOL", "BTC"), ctx) == "quote_lain"
    assert classify_market(make_market("SOL", spot=False), ctx) == "bukan_spot"
    assert classify_market(make_market("SOL", status="HALT"), ctx) == "tidak_aktif"
    assert classify_market(make_market("SOL"), ctx) is None


def test_volume_dan_deteksi_peg():
    assert quote_volume({"quoteVolume": 123.0}) == 123.0
    assert quote_volume({"quoteVolume": None, "baseVolume": 10, "last": 2.5}) == 25.0
    assert quote_volume({}) == 0.0
    assert looks_like_usd_peg({"last": 1.0001, "high": 1.002, "low": 0.999})
    assert not looks_like_usd_peg({"last": 1.0, "high": 1.08, "low": 0.95})   # coin biasa di harga $1
    assert not looks_like_usd_peg({"last": 60_000, "high": 60_100, "low": 59_900})
