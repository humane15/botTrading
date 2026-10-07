"""Deteksi swing high dan swing low (pivot/fractal).

Sebuah swing high di candle i: high[i] lebih tinggi dari `left` candle di
kiri dan tidak terlampaui oleh `right` candle di kanan. Artinya pivot baru
TERKONFIRMASI setelah `right` candle berikutnya tertutup. Karena fungsi ini
hanya melihat frame yang diberikan, pivot yang belum terkonfirmasi tidak akan
pernah muncul (tidak ada look ahead).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(frozen=True)
class Pivot:
    position: int          # posisi baris di frame (0 = baris pertama)
    price: float
    kind: str              # "high" atau "low"
    volume_ratio: float = float("nan")


def _pivot_positions(values: np.ndarray, left: int, right: int, find_high: bool) -> np.ndarray:
    window = left + right + 1
    if values.size < window:
        return np.empty(0, dtype=int)
    data = values if find_high else -values
    windows = sliding_window_view(data, window)
    center = windows[:, left]
    left_max = windows[:, :left].max(axis=1)
    right_max = windows[:, left + 1 :].max(axis=1)
    # Kiri harus lebih rendah (ketat), kanan boleh sama: puncak datar dihitung sekali.
    mask = (center > left_max) & (center >= right_max)
    return np.flatnonzero(mask) + left


def find_pivots(
    frame: pd.DataFrame, left: int = 3, right: int = 3, lookback: int | None = None
) -> tuple[list[Pivot], list[Pivot]]:
    """Kembalikan (swing highs, swing lows) terurut dari yang paling lama.

    lookback membatasi pencarian ke `lookback` candle terakhir.
    """
    if left < 1 or right < 1:
        raise ValueError("left dan right minimal 1")
    start = max(0, len(frame) - lookback) if lookback else 0
    highs = frame["high"].to_numpy(dtype="float64")[start:]
    lows = frame["low"].to_numpy(dtype="float64")[start:]
    ratios = frame["volume_ratio"].to_numpy(dtype="float64")[start:] if "volume_ratio" in frame.columns else None

    def build(positions: np.ndarray, prices: np.ndarray, kind: str) -> list[Pivot]:
        return [
            Pivot(
                position=start + int(pos),
                price=float(prices[pos]),
                kind=kind,
                volume_ratio=float(ratios[pos]) if ratios is not None else float("nan"),
            )
            for pos in positions
        ]

    swing_highs = build(_pivot_positions(highs, left, right, True), highs, "high")
    swing_lows = build(_pivot_positions(lows, left, right, False), lows, "low")
    return swing_highs, swing_lows
