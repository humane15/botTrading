"""Universe coin point in time untuk backtest.

Bot live memilih 75 coin dengan volume 24 jam terbesar dan memperbaruinya tiap
24 jam. Backtest meniru itu TANPA melihat masa depan: universe untuk hari D
dipilih dari volume (close x volume candle 1h) pada hari D-1, dengan syarat
volume minimal yang sama (default 5 juta USD).

Memakai daftar 75 coin HARI INI untuk menguji 6 bulan ke belakang akan
membuat hasil terlalu bagus (coin yang sekarang ramai biasanya coin yang naik
di periode itu). Cara ini mengurangi bias tersebut, tetapi tidak
menghilangkannya: coin yang sudah delisting tidak bisa diunduh dari Binance,
dan kandidat tetap diambil dari coin yang masih diperdagangkan sekarang.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def daily_quote_volume(frame_1h: pd.DataFrame) -> pd.Series:
    """Volume USDT per hari UTC (jumlah close x volume candle 1h)."""
    if frame_1h.empty:
        return pd.Series(dtype="float64")
    hourly = frame_1h["close"] * frame_1h["volume"]
    return hourly.groupby(frame_1h.index.floor("D")).sum()


def daily_universe(
    frames_1h: Mapping[str, pd.DataFrame],
    days: pd.DatetimeIndex,
    size: int,
    min_quote_volume: float,
) -> dict[pd.Timestamp, tuple[str, ...]]:
    """{hari: simbol terpilih} memakai volume hari sebelumnya saja."""
    volumes = pd.DataFrame({symbol: daily_quote_volume(frame) for symbol, frame in frames_1h.items()})
    universe: dict[pd.Timestamp, tuple[str, ...]] = {}
    for day in days:
        previous = day - pd.Timedelta(days=1)
        if previous not in volumes.index:
            universe[day] = ()
            continue
        row = volumes.loc[previous].dropna()
        row = row[row >= min_quote_volume].sort_values(ascending=False, kind="stable")
        universe[day] = tuple(row.index[:size])
    return universe
