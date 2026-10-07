"""Test DataFeed: candle tertutup, cache per timeframe, update inkremental, dan fetch paralel."""

from __future__ import annotations

import pandas as pd
import pytest
from helpers import T0, FakeClock, FakeKlineServer, make_settings

from config.settings import TIMEFRAME_MS
from core.data_feed import (
    DataFeed,
    frame_open_ms,
    last_closed_open_ms,
    ohlcv_to_frame,
    timeframe_to_ms,
)
from core.exchange import ExchangeClient

M5 = TIMEFRAME_MS["5m"]
SECOND = 1_000
TIMEFRAMES = ("5m", "15m", "30m", "1h")


@pytest.fixture
def clock():
    return FakeClock(T0 + 5 * SECOND)


@pytest.fixture
def server(clock, mock_exchange):
    srv = FakeKlineServer(clock)
    mock_exchange.fetch_ohlcv.side_effect = srv.fetch_ohlcv
    return srv


def make_feed(client, clock, limit=50, **kwargs):
    return DataFeed(client, TIMEFRAMES, limit, grace_ms=2 * SECOND, clock=clock, **kwargs)


def calls_by_timeframe(mock_exchange):
    return [call.args[1] for call in mock_exchange.fetch_ohlcv.await_args_list]


def test_last_closed_open_ms_dengan_masa_grace():
    # 12:00:03 -> candle 11:55 sudah final; 12:00:01 -> masih dalam grace, terakhir 11:50.
    assert last_closed_open_ms(T0 + 3 * SECOND, M5, 2 * SECOND) == T0 - M5
    assert last_closed_open_ms(T0 + 1 * SECOND, M5, 2 * SECOND) == T0 - 2 * M5
    assert last_closed_open_ms(T0, M5, 0) == T0 - M5


def test_timeframe_to_ms():
    assert timeframe_to_ms("5m") == 300_000
    assert timeframe_to_ms("1h") == 3_600_000
    with pytest.raises(ValueError):
        timeframe_to_ms("7m")


def test_ohlcv_to_frame_rapi_dan_bertipe_benar():
    rows = [
        [T0 + M5, 2, 3, 1, 2.5, 10],
        [T0, 1, 2, 0.5, 1.5, 5],
        [T0 + M5, 2, 3, 1, 2.6, 11],  # duplikat, versi terakhir dipakai
        [T0 + 2 * M5, None, 3, 1, 2, 1],  # baris rusak dibuang
    ]
    frame = ohlcv_to_frame(rows)
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert all(dtype == "float64" for dtype in frame.dtypes)
    assert str(frame.index.tz) == "UTC"
    assert list(frame_open_ms(frame)) == [T0, T0 + M5]
    assert frame["close"].iloc[-1] == 2.6
    assert ohlcv_to_frame([]).empty


async def test_fetch_awal_membuang_candle_berjalan(client, mock_exchange, server, clock):
    feed = make_feed(client, clock)
    clock.advance(2 * 60 * SECOND)  # 12:02:05, candle 12:00 masih berjalan
    frame = await feed.get_ohlcv("BTC/USDT", "5m")

    assert len(frame) == 50
    assert frame_open_ms(frame)[-1] == T0 - M5  # candle terakhir 11:55
    assert frame.index.is_monotonic_increasing
    assert (pd.Series(frame_open_ms(frame)).diff().dropna() == M5).all()
    expected_close = FakeKlineServer.candle("BTC/USDT", M5, T0 - M5)[4]
    assert frame["close"].iloc[-1] == pytest.approx(expected_close)
    mock_exchange.fetch_ohlcv.assert_awaited_once_with("BTC/USDT", "5m", None, 52)


async def test_cache_dipakai_dalam_candle_yang_sama(client, mock_exchange, server, clock):
    feed = make_feed(client, clock)
    first = await feed.get_ohlcv("BTC/USDT", "5m")
    clock.advance(60 * SECOND)
    second = await feed.get_ohlcv("BTC/USDT", "5m")
    assert second is first
    assert mock_exchange.fetch_ohlcv.await_count == 1
    assert feed.stats.cache_hits == 1


async def test_timeframe_besar_hanya_difetch_saat_candle_baru_tutup(client, mock_exchange, server, clock):
    feed = make_feed(client, clock)
    expected_calls = [
        (0, {"5m", "15m", "30m", "1h"}),  # 12:00:05 fetch awal semua timeframe
        (5, {"5m"}),                      # 12:05:05 hanya 5m
        (10, {"5m"}),
        (15, {"5m", "15m"}),              # 12:15:05 candle 15m baru tutup
        (20, {"5m"}),
        (30, {"5m", "15m", "30m"}),
        (60, {"5m", "15m", "30m", "1h"}), # 13:00:05 candle 1h baru tutup
    ]
    for minute, expected in expected_calls:
        clock.now_ms = T0 + minute * 60 * SECOND + 5 * SECOND
        before = mock_exchange.fetch_ohlcv.await_count
        result = await feed.refresh(["BTC/USDT"])
        new_calls = calls_by_timeframe(mock_exchange)[before:]
        assert set(new_calls) == expected, f"menit {minute}"
        assert len(new_calls) == len(expected)
        assert result.complete_symbols(TIMEFRAMES) == ["BTC/USDT"]


async def test_update_inkremental_memakai_since(client, mock_exchange, server, clock):
    feed = make_feed(client, clock)
    first = await feed.get_ohlcv("ETH/USDT", "5m")
    clock.advance(10 * 60 * SECOND)  # dua candle 5m baru tertutup
    frame = await feed.get_ohlcv("ETH/USDT", "5m")

    call = mock_exchange.fetch_ohlcv.await_args_list[-1]
    assert call.args == ("ETH/USDT", "5m", frame_open_ms(first)[-1] + M5, 4)
    assert len(frame) == 50
    assert frame_open_ms(frame)[-1] == T0 + M5
    assert not frame.index.has_duplicates
    # Candle yang tadinya berjalan kini masuk dengan nilai final, bukan nilai sementara.
    final = FakeKlineServer.candle("ETH/USDT", M5, T0)[4]
    assert frame.loc[pd.Timestamp(T0, unit="ms", tz="UTC"), "close"] == pytest.approx(final)
    assert feed.stats.incremental_fetches == 1


async def test_jeda_panjang_memicu_fetch_penuh(client, mock_exchange, server, clock):
    feed = make_feed(client, clock)
    await feed.get_ohlcv("BTC/USDT", "5m")
    clock.advance(60 * M5)  # 60 candle terlewat, lebih dari limit 50
    frame = await feed.get_ohlcv("BTC/USDT", "5m")
    assert mock_exchange.fetch_ohlcv.await_args_list[-1].args == ("BTC/USDT", "5m", None, 52)
    assert len(frame) == 50
    assert feed.stats.full_fetches == 2


async def test_candle_belum_final_selama_masa_grace(client, server, clock):
    feed = make_feed(client, clock)
    clock.now_ms = T0 + M5 + 1 * SECOND  # candle 12:00 baru tutup 1 detik lalu
    frame = await feed.get_ohlcv("BTC/USDT", "5m")
    assert frame_open_ms(frame)[-1] == T0 - M5
    assert feed.is_current("BTC/USDT", "5m")
    # Lewat masa grace (dan jeda fetch ulang minimal 15 detik): candle 12:00 masuk.
    clock.now_ms = T0 + M5 + 20 * SECOND
    assert not feed.is_current("BTC/USDT", "5m")
    frame = await feed.get_ohlcv("BTC/USDT", "5m")
    assert frame_open_ms(frame)[-1] == T0


async def test_bursa_telat_merilis_candle_fetch_ulang_dengan_jeda(client, mock_exchange, clock):
    # lag 2: candle berjalan DAN candle tertutup terbaru belum dirilis bursa.
    lagging = FakeKlineServer(clock, lag_candles=2)
    mock_exchange.fetch_ohlcv.side_effect = lagging.fetch_ohlcv
    feed = make_feed(client, clock, min_refetch_ms=15 * SECOND)
    await feed.get_ohlcv("BTC/USDT", "5m")
    assert not feed.is_current("BTC/USDT", "5m")  # masih tertinggal satu candle
    clock.advance(5 * SECOND)
    assert not feed.needs_update("BTC/USDT", "5m")
    await feed.get_ohlcv("BTC/USDT", "5m")
    assert mock_exchange.fetch_ohlcv.await_count == 1  # belum 15 detik, tidak fetch ulang
    clock.advance(11 * SECOND)
    assert feed.needs_update("BTC/USDT", "5m")
    await feed.get_ohlcv("BTC/USDT", "5m")
    assert mock_exchange.fetch_ohlcv.await_count == 2


async def test_refresh_75_coin_paralel_dibatasi_semaphore(settings, mock_exchange, server, clock):
    client = ExchangeClient(settings, mock_exchange, jitter=0)
    feed = make_feed(client, clock)
    symbols = [f"C{i:03d}/USDT" for i in range(75)]
    result = await feed.refresh(symbols)

    assert mock_exchange.fetch_ohlcv.await_count == 75 * 4
    assert result.errors == {}
    assert len(result.complete_symbols(TIMEFRAMES)) == 75
    assert server.max_in_flight == settings.max_concurrent_requests == 5


async def test_kegagalan_satu_simbol_tidak_mengganggu_yang_lain(tmp_path, mock_exchange, server, clock, fake_time):
    settings = make_settings(tmp_path, max_retries=2)
    client = ExchangeClient(settings, mock_exchange, sleep=fake_time.sleep, monotonic=fake_time.monotonic, jitter=0)
    server.fail_symbols = {"BAD/USDT"}
    feed = make_feed(client, clock)
    result = await feed.refresh(["BTC/USDT", "BAD/USDT", "ETH/USDT"])

    assert result.complete_symbols(TIMEFRAMES) == ["BTC/USDT", "ETH/USDT"]
    assert set(result.errors) == {("BAD/USDT", tf) for tf in TIMEFRAMES}
    assert "BAD/USDT" not in result.data
    assert feed.stats.errors == 4


async def test_refresh_menandai_data_yang_tertinggal(client, mock_exchange, clock):
    lagging = FakeKlineServer(clock, lag_candles=2)
    mock_exchange.fetch_ohlcv.side_effect = lagging.fetch_ohlcv
    feed = make_feed(client, clock)
    result = await feed.refresh(["BTC/USDT"])
    assert result.errors == {}
    assert ("BTC/USDT", "5m") in result.stale
    assert set(result.data["BTC/USDT"]) == set(TIMEFRAMES)
    assert result.complete_symbols(TIMEFRAMES) == []  # data basi tidak dipakai untuk sinyal


async def test_get_multi_timeframe(client, server, clock):
    feed = make_feed(client, clock)
    frames = await feed.get_multi_timeframe("SOL/USDT")
    assert list(frames) == list(TIMEFRAMES)
    for tf, frame in frames.items():
        assert len(frame) == 50
        assert frame_open_ms(frame)[-1] == last_closed_open_ms(clock(), TIMEFRAME_MS[tf], 2 * SECOND)


async def test_prune_menghapus_simbol_yang_keluar_universe(client, server, clock):
    feed = make_feed(client, clock)
    await feed.refresh(["BTC/USDT", "OLD/USDT"])
    assert feed.prune(["BTC/USDT"]) == 4
    assert feed.cached("OLD/USDT", "5m") is None
    assert feed.cached("BTC/USDT", "5m") is not None


async def test_fetch_history_dengan_paginasi(client, mock_exchange, server, clock):
    feed = make_feed(client, clock)
    clock.now_ms = T0 + 10_000 * M5
    start = T0
    end = T0 + 2_500 * M5
    frame = await feed.fetch_history("BTC/USDT", "5m", start, end, page_limit=1000)

    assert mock_exchange.fetch_ohlcv.await_count == 3
    sinces = [call.args[2] for call in mock_exchange.fetch_ohlcv.await_args_list]
    assert sinces == [start, start + 1000 * M5, start + 2000 * M5]
    assert len(frame) == 2_500
    opens = frame_open_ms(frame)
    assert opens[0] == start and opens[-1] == end - M5
    assert (pd.Series(opens).diff().dropna() == M5).all()


async def test_fetch_history_tidak_memuat_candle_berjalan(client, server, clock):
    feed = make_feed(client, clock)
    clock.now_ms = T0 + 100 * M5 + 30 * SECOND
    frame = await feed.fetch_history("BTC/USDT", "5m", T0)
    assert len(frame) == 100
    assert frame_open_ms(frame)[-1] == T0 + 99 * M5


def test_validasi_parameter_data_feed(client):
    with pytest.raises(ValueError):
        DataFeed(client, ["5m", "7m"])
    with pytest.raises(ValueError):
        DataFeed(client, ["5m"], limit=1001)
