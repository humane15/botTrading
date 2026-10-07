"""Test data historis backtest: penyimpanan .npz, aturan pair, dan unduhan inkremental (Binance di-mock)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from helpers import (
    T0,
    FakeClock,
    FakeKlineServer,
    install_markets,
    make_market,
    sol_rules,
)
from synthetic import random_ohlcv

from backtest.data import (
    HistoryStore,
    download_history,
    rules_from_dict,
    rules_to_dict,
    warmup_start_ms,
)
from config.settings import TIMEFRAME_MS
from core.data_feed import frame_open_ms
from core.exchange import ExchangeClient


def test_simpan_dan_baca_ulang_identik(tmp_path):
    store = HistoryStore(tmp_path)
    frame = random_ohlcv(500, freq="5m")
    store.save("SOL/USDT", "5m", frame, requested_from=123)
    loaded, requested = store._load_with_meta("SOL/USDT", "5m")
    assert requested == 123 and store.has("SOL/USDT", "5m") and not store.has("SOL/USDT", "1h")
    assert np.array_equal(frame_open_ms(loaded), frame_open_ms(frame))
    pd.testing.assert_frame_equal(loaded.reset_index(drop=True), frame.reset_index(drop=True))
    assert store.load("ETH/USDT", "5m").empty
    assert list(tmp_path.glob("*.tmp*")) == []  # tulis atomik, tidak ada sisa file sementara


def test_aturan_pair_dan_kandidat(tmp_path):
    store = HistoryStore(tmp_path)
    rules = sol_rules()  # max_qty tak hingga
    assert rules_from_dict(rules_to_dict(rules)) == rules
    store.save_rules({"SOL/USDT": rules})
    store.save_rules({"ETH/USDT": sol_rules(symbol="ETH/USDT", base="ETH", max_qty=9000.0)})
    loaded = store.load_rules()
    assert set(loaded) == {"SOL/USDT", "ETH/USDT"} and math.isinf(loaded["SOL/USDT"].max_qty)
    store.save_candidates(["SOL/USDT", "ETH/USDT"], "binance")
    assert store.load_candidates() == ["SOL/USDT", "ETH/USDT"]


@pytest.fixture
def kline_client(settings, mock_exchange):
    clock = FakeClock(T0)
    server = FakeKlineServer(clock)
    install_markets(mock_exchange, {s: make_market(s.split("/")[0]) for s in ("SOL/USDT", "BTC/USDT")})
    mock_exchange.fetch_ohlcv.side_effect = server.fetch_ohlcv
    client = ExchangeClient(settings, mock_exchange, jitter=0)
    return client, clock


async def test_unduh_dengan_pemanasan_lalu_hanya_bagian_baru(tmp_path, settings, kline_client):
    client, clock = kline_client
    store = HistoryStore(tmp_path / "history")
    start, end = T0 - 2 * 86_400_000, T0 - 86_400_000
    report = await download_history(client, settings, store, ["SOL/USDT", "NOPE/USDT"], start, end)
    assert report.failed == {"NOPE/USDT": "tidak ada di Binance spot"}
    for tf in settings.timeframes:
        opens = frame_open_ms(store.load("SOL/USDT", tf))
        assert opens[0] == warmup_start_ms(start, tf)  # 1000 candle pemanasan sebelum tanggal mulai
        assert opens[-1] == end - TIMEFRAME_MS[tf] and len(opens) == 1000 + 86_400_000 // TIMEFRAME_MS[tf]
        assert np.all(np.diff(opens) == TIMEFRAME_MS[tf])
    assert "SOL/USDT" in store.load_rules()

    calls = client.exchange.fetch_ohlcv.await_count
    again = await download_history(client, settings, store, ["SOL/USDT"], start, end)
    assert client.exchange.fetch_ohlcv.await_count == calls and again.series_complete == 4  # sudah lengkap

    later = await download_history(client, settings, store, ["SOL/USDT"], start, end + 3_600_000)
    assert later.new_candles == sum(3_600_000 // TIMEFRAME_MS[tf] for tf in settings.timeframes)
    assert client.exchange.fetch_ohlcv.await_count == calls + 4  # satu request per timeframe untuk bagian baru
