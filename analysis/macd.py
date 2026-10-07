"""MACD 12, 26, 9 (perpindahan momentum).

* Crossover garis MACD dan signal dalam beberapa candle terakhir.
* Posisi MACD terhadap garis nol.
* Perubahan histogram (momentum menguat atau melemah).
* Crossover ke atas di bawah garis nol saat tren 1h naik diberi skor lebih
  tinggi: momentum berbalik naik dari koreksi di dalam tren besar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.base import SignalScore, has_columns, is_valid, last, tail
from analysis.moving_average import TREND_UP


@dataclass(frozen=True)
class MACDParams:
    cross_lookback: int = 3          # cross dianggap baru jika terjadi <= 3 candle lalu
    cross_weight: float = 0.5
    below_zero_uptrend_bonus: float = 0.3
    zero_line_weight: float = 0.15
    histogram_weight: float = 0.15


def find_recent_cross(frame: pd.DataFrame, lookback: int) -> tuple[int, int] | None:
    """(arah, posisi) crossover MACD/signal terbaru dalam `lookback` candle, arah +1/-1."""
    count = min(len(frame), lookback + 1)
    diff = tail(frame, "macd", count) - tail(frame, "macd_signal", count)
    offset = len(frame) - count  # posisi baris pertama `diff` di frame
    latest = None
    for i in range(1, count):
        prev, curr = diff[i - 1], diff[i]
        if np.isnan(prev) or np.isnan(curr):
            continue
        if prev <= 0 < curr:
            latest = (1, offset + i)
        elif prev >= 0 > curr:
            latest = (-1, offset + i)
    return latest


def score_macd(frame: pd.DataFrame, higher_trend: str = "sideways", params: MACDParams = MACDParams()) -> SignalScore:
    name = "macd"
    if len(frame) < 2 or not has_columns(frame, "macd", "macd_signal", "macd_hist"):
        return SignalScore.neutral(name, "data MACD belum cukup")
    line, hist, hist_prev = last(frame, "macd"), last(frame, "macd_hist"), last(frame, "macd_hist", 2)
    if not is_valid(line, hist, hist_prev):
        return SignalScore.neutral(name, "data MACD belum cukup")

    score = 0.0
    tags: list[str] = []
    parts: list[str] = []
    label = ""
    cross = find_recent_cross(frame, params.cross_lookback)
    if cross is not None:
        direction, pos = cross
        below_zero = last(frame, "macd", len(frame) - pos) < 0
        if direction > 0:
            score += params.cross_weight
            tags.append("macd_cross_up")
            label = "MACD cross"
            parts.append("crossover naik")
            if below_zero:
                tags.append("macd_cross_below_zero")
                parts.append("di bawah garis nol")
                if higher_trend == TREND_UP:
                    score += params.below_zero_uptrend_bonus
                    label = "MACD cross di bawah nol"
                    parts.append("saat tren 1h naik")
        else:
            score -= params.cross_weight
            tags.append("macd_cross_down")
            parts.append("crossover turun")

    if line > 0:
        score += params.zero_line_weight
        tags.append("macd_above_zero")
        parts.append("MACD di atas nol")
    else:
        score -= params.zero_line_weight
        parts.append("MACD di bawah nol")
    if hist > hist_prev:
        score += params.histogram_weight
        tags.append("macd_hist_rising")
        parts.append("histogram menguat")
    else:
        score -= params.histogram_weight
        parts.append("histogram melemah")

    return SignalScore(
        name=name,
        score=score,
        reason="MACD: " + ", ".join(parts),
        label=label,
        tags=tuple(tags),
        details={
            "macd": line,
            "macd_hist": hist,
            "cross": cross[0] if cross else 0,
            "cross_bars_ago": (len(frame) - 1 - cross[1]) if cross else -1,
        },
    )
