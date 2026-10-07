"""Test ExchangeClient: retry, exponential backoff 429, cooldown global, dan semaphore.

Semua respons ccxt disimulasikan dengan AsyncMock; tidak ada request ke Binance.
"""

from __future__ import annotations

import asyncio

import ccxt
import pytest
from helpers import ManualTime, make_settings

from core.exchange import (
    ApiPermissionReport,
    ExchangeClient,
    UnsafeApiKeyError,
    build_exchange,
)

RATE_LIMIT_429 = ccxt.DDoSProtection("binance 429 Too Many Requests")
ROWS = [[1_700_000_000_000, 1.0, 2.0, 0.5, 1.5, 100.0]]


async def test_retry_429_dengan_exponential_backoff(client, mock_exchange, fake_time):
    mock_exchange.fetch_ohlcv.side_effect = [RATE_LIMIT_429, RATE_LIMIT_429, ROWS]
    result = await client.fetch_ohlcv("BTC/USDT", "5m", limit=1)
    assert result == ROWS
    assert mock_exchange.fetch_ohlcv.await_count == 3
    assert fake_time.sleeps == [1.0, 2.0]  # base 1 detik, lalu dua kali lipat
    assert client.stats.retries == 2
    assert client.stats.rate_limit_hits == 2


async def test_kode_binance_1003_juga_diulang(client, mock_exchange, fake_time):
    mock_exchange.fetch_tickers.side_effect = [ccxt.RateLimitExceeded('binance {"code":-1003}'), {"BTC/USDT": {}}]
    assert await client.fetch_tickers() == {"BTC/USDT": {}}
    assert fake_time.sleeps == [1.0]


async def test_menyerah_setelah_batas_retry(tmp_path, mock_exchange, fake_time):
    settings = make_settings(tmp_path, max_retries=3)
    client = ExchangeClient(settings, mock_exchange, sleep=fake_time.sleep, monotonic=fake_time.monotonic, jitter=0)
    mock_exchange.fetch_ohlcv.side_effect = RATE_LIMIT_429
    with pytest.raises(ccxt.DDoSProtection):
        await client.fetch_ohlcv("BTC/USDT", "5m")
    assert mock_exchange.fetch_ohlcv.await_count == 4  # 1 percobaan + 3 retry
    assert fake_time.sleeps == [1.0, 2.0, 4.0]
    assert client.stats.failures == 1
    assert client.stats.rate_limit_hits == 4
    # Walau menyerah, request lain tetap harus jeda karena IP sedang dibatasi.
    assert client.cooldown_remaining() == pytest.approx(8.0)


async def test_backoff_dibatasi_retry_max_delay(tmp_path, mock_exchange, fake_time):
    settings = make_settings(tmp_path, max_retries=5, retry_base_delay=1, retry_max_delay=5)
    client = ExchangeClient(settings, mock_exchange, sleep=fake_time.sleep, monotonic=fake_time.monotonic, jitter=0)
    mock_exchange.fetch_ticker.side_effect = ccxt.RequestTimeout("binance timeout")
    with pytest.raises(ccxt.RequestTimeout):
        await client.fetch_ticker("BTC/USDT")
    assert fake_time.sleeps == [1.0, 2.0, 4.0, 5.0, 5.0]


async def test_jitter_menambah_jeda_secara_acak(settings, mock_exchange, fake_time):
    client = ExchangeClient(settings, mock_exchange, sleep=fake_time.sleep, monotonic=fake_time.monotonic, jitter=0.2)
    mock_exchange.fetch_ticker.side_effect = [ccxt.NetworkError("putus"), {"last": 1}]
    await client.fetch_ticker("BTC/USDT")
    assert 1.0 <= fake_time.sleeps[0] <= 1.2


@pytest.mark.parametrize(
    "error",
    [
        ccxt.AuthenticationError("binance invalid api key"),
        ccxt.BadSymbol("binance does not have market symbol XXX/USDT"),
        ccxt.InsufficientFunds("binance saldo kurang"),
        ccxt.InvalidOrder("binance LOT_SIZE"),
    ],
)
async def test_error_permanen_tidak_diulang(client, mock_exchange, fake_time, error):
    mock_exchange.fetch_ticker.side_effect = error
    with pytest.raises(type(error)):
        await client.fetch_ticker("BTC/USDT")
    assert mock_exchange.fetch_ticker.await_count == 1
    assert fake_time.sleeps == []


async def test_order_tidak_diulang_saat_timeout_agar_tidak_dobel(client, mock_exchange, fake_time):
    # Timeout saat membuat order: status order tidak pasti, jangan kirim ulang.
    mock_exchange.create_order.side_effect = ccxt.RequestTimeout("binance timeout")
    with pytest.raises(ccxt.RequestTimeout):
        await client.call("create_order", "BTC/USDT", "market", "buy", 0.001, idempotent=False)
    assert mock_exchange.create_order.await_count == 1


async def test_order_tetap_diulang_jika_ditolak_rate_limit(client, mock_exchange, fake_time):
    # HTTP 429 berarti request ditolak sebelum diproses, aman dikirim ulang.
    mock_exchange.create_order.side_effect = [RATE_LIMIT_429, {"id": "1"}]
    result = await client.call("create_order", "BTC/USDT", "market", "buy", 0.001, idempotent=False)
    assert result == {"id": "1"}
    assert mock_exchange.create_order.await_count == 2


async def test_header_retry_after_dihormati(client, mock_exchange, fake_time):
    mock_exchange.last_response_headers = {"Retry-After": "7"}
    mock_exchange.fetch_ticker.side_effect = [RATE_LIMIT_429, {"last": 1}]
    await client.fetch_ticker("BTC/USDT")
    assert fake_time.sleeps == [7.0]


async def test_ban_ip_418_menunggu_minimal_dua_menit(client, mock_exchange, fake_time):
    mock_exchange.fetch_ticker.side_effect = [ccxt.DDoSProtection("binance 418 I'm a teapot"), {"last": 1}]
    await client.fetch_ticker("BTC/USDT")
    assert fake_time.sleeps[0] >= 120


async def test_cooldown_global_menahan_request_lain(settings, mock_exchange):
    clock = ManualTime()
    client = ExchangeClient(settings, mock_exchange, sleep=clock.sleep, monotonic=clock.monotonic, jitter=0)
    calls: list[tuple[str, float]] = []

    async def fake_ticker(symbol, params=None):
        calls.append((symbol, clock.now))
        if symbol == "A/USDT" and len(calls) == 1:
            raise RATE_LIMIT_429
        return {"symbol": symbol}

    mock_exchange.fetch_ticker.side_effect = fake_ticker
    task_a = asyncio.create_task(client.fetch_ticker("A/USDT"))
    while client.cooldown_remaining() == 0:  # tunggu sampai A terkena 429
        await asyncio.sleep(0)
    task_b = asyncio.create_task(client.fetch_ticker("B/USDT"))
    for _ in range(20):
        await asyncio.sleep(0)
    assert [s for s, _ in calls] == ["A/USDT"], "B harus tertahan selama cooldown"

    while not (task_a.done() and task_b.done()):
        clock.now += 0.25
        await asyncio.sleep(0)
    b_time = next(t for s, t in calls if s == "B/USDT")
    assert b_time >= 1.0  # baru jalan setelah cooldown 1 detik selesai
    assert task_b.result() == {"symbol": "B/USDT"}


async def test_semaphore_membatasi_request_paralel(settings, mock_exchange):
    client = ExchangeClient(settings, mock_exchange, jitter=0)
    state = {"now": 0, "max": 0}
    release = asyncio.Event()

    async def slow_ticker(symbol, params=None):
        state["now"] += 1
        state["max"] = max(state["max"], state["now"])
        await release.wait()
        state["now"] -= 1
        return {"symbol": symbol}

    mock_exchange.fetch_ticker.side_effect = slow_ticker
    tasks = [asyncio.create_task(client.fetch_ticker(f"C{i}/USDT")) for i in range(20)]
    for _ in range(20):
        await asyncio.sleep(0)
    assert state["now"] == settings.max_concurrent_requests == 5
    release.set()
    results = await asyncio.gather(*tasks)
    assert len(results) == 20
    assert state["max"] == 5


async def test_error_timestamp_1021_sinkron_ulang_lalu_coba_lagi(client, mock_exchange, fake_time):
    mock_exchange.fetch_balance.side_effect = [
        ccxt.InvalidNonce('binance {"code":-1021,"msg":"Timestamp for this request is outside of the recvWindow."}'),
        {"free": {"USDT": 10.0}},
    ]
    assert await client.fetch_free_balance("USDT") == 10.0
    mock_exchange.load_time_difference.assert_awaited_once()
    assert client.stats.retries == 0  # sinkron ulang tidak dihitung sebagai retry


async def test_sync_time_menghitung_offset(settings, mock_exchange):
    client = ExchangeClient(settings, mock_exchange, wall_clock_ms=lambda: 1_000_000)
    mock_exchange.fetch_time.return_value = 1_005_000
    assert await client.sync_time() == 5_000
    assert client.now_ms() == 1_005_000


async def test_saldo_dari_mock(client, mock_exchange):
    mock_exchange.fetch_balance.return_value = {
        "free": {"USDT": 107.4, "BTC": 0.001},
        "used": {"USDT": 0.0},
        "total": {"USDT": 107.4, "BTC": 0.001},
    }
    assert await client.fetch_free_balance("USDT") == pytest.approx(107.4)
    assert await client.fetch_free_balance("ETH") == 0.0


def test_laporan_izin_api_key_aman():
    report = ApiPermissionReport.from_response(
        {"enableReading": True, "enableSpotAndMarginTrading": True, "enableWithdrawals": False, "ipRestrict": True}
    )
    assert report.is_safe
    assert report.warnings == []


def test_laporan_izin_api_key_berbahaya():
    report = ApiPermissionReport.from_response(
        {
            "enableReading": True,
            "enableSpotAndMarginTrading": True,
            "enableWithdrawals": True,
            "ipRestrict": False,
            "enableFutures": True,
        }
    )
    assert not report.is_safe
    assert any("Withdraw" in p for p in report.problems)
    assert any("IP whitelist" in p for p in report.problems)
    assert any("futures" in w for w in report.warnings)


async def test_assert_api_key_safe_menolak_izin_withdraw(tmp_path, mock_exchange):
    settings = make_settings(tmp_path, binance_api_key="kunci-palsu-123", binance_api_secret="rahasia-palsu-456")
    client = ExchangeClient(settings, mock_exchange)
    mock_exchange.sapi_get_account_apirestrictions.return_value = {
        "enableReading": True, "enableSpotAndMarginTrading": True, "enableWithdrawals": True, "ipRestrict": True,
    }
    with pytest.raises(UnsafeApiKeyError, match="Withdraw"):
        await client.assert_api_key_safe()


async def test_jadwal_delisting_butuh_api_key(client, mock_exchange, tmp_path):
    assert await client.fetch_delisting_ids() == set()
    mock_exchange.sapi_get_spot_delist_schedule.assert_not_awaited()

    settings = make_settings(tmp_path, binance_api_key="kunci-palsu-123", binance_api_secret="rahasia-palsu-456")
    keyed = ExchangeClient(settings, mock_exchange)
    mock_exchange.sapi_get_spot_delist_schedule.return_value = [
        {"delistTime": 1, "symbols": ["AAAUSDT", "BBBUSDT"]},
        {"delistTime": 2, "symbols": ["CCCUSDT"]},
    ]
    assert await keyed.fetch_delisting_ids() == {"AAAUSDT", "BBBUSDT", "CCCUSDT"}


async def test_context_manager_menutup_koneksi(settings, mock_exchange):
    async with ExchangeClient(settings, mock_exchange) as client:
        assert client.exchange is mock_exchange
    mock_exchange.close.assert_awaited_once()


async def test_build_exchange_tanpa_dan_dengan_api_key(tmp_path):
    public = build_exchange(make_settings(tmp_path))
    keyed = build_exchange(
        make_settings(tmp_path, binance_api_key="kunci-palsu-123", binance_api_secret="rahasia-palsu-456", binance_testnet=True)
    )
    try:
        assert not public.apiKey
        assert public.options["defaultType"] == "spot"
        assert public.options["fetchMarkets"]["types"] == ["spot"]
        assert public.enableRateLimit is True
        assert keyed.apiKey == "kunci-palsu-123"
        assert "testnet" in str(keyed.urls["api"])
    finally:
        await public.close()
        await keyed.close()
