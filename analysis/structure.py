"""Struktur pasar (peran timeframe 30m): higher high / higher low.

* Dua swing high dan dua swing low terakhir: HH + HL = struktur naik,
  LH + LL = struktur turun, kombinasi lain = transisi atau konsolidasi.
* Break of structure: close menembus swing high terakhir (naik) atau swing
  low terakhir (turun).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from analysis.base import SignalScore, is_valid, last
from analysis.pivots import find_pivots


@dataclass(frozen=True)
class StructureParams:
    left: int = 3
    right: int = 3
    lookback: int = 120
    trend_weight: float = 0.6
    break_weight: float = 0.4


def score_structure(frame: pd.DataFrame, params: StructureParams = StructureParams()) -> SignalScore:
    name = "structure"
    highs, lows = find_pivots(frame, params.left, params.right, params.lookback)
    if len(highs) < 2 or len(lows) < 2:
        return SignalScore.neutral(name, "swing belum cukup untuk membaca struktur")
    close = last(frame, "close")
    if not is_valid(close):
        return SignalScore.neutral(name, "data harga tidak valid")

    h1, h2 = highs[-2].price, highs[-1].price
    l1, l2 = lows[-2].price, lows[-1].price
    score = 0.0
    tags: list[str] = []
    label = ""
    if h2 > h1 and l2 > l1:
        score += params.trend_weight
        tags.append("hh_hl")
        label = "HH + HL"
        text = "higher high + higher low (struktur naik)"
    elif h2 < h1 and l2 < l1:
        score -= params.trend_weight
        tags.append("lh_ll")
        text = "lower high + lower low (struktur turun)"
    elif h2 > h1 and l2 < l1:
        text = "higher high + lower low (volatilitas melebar)"
    else:
        text = "lower high + higher low (menyempit/konsolidasi)"

    if close > h2:
        score += params.break_weight
        tags.append("bos_up")
        label = label or "Break of structure naik"
        text += ", close menembus swing high terakhir"
    elif close < l2:
        score -= params.break_weight
        tags.append("bos_down")
        text += ", close menembus swing low terakhir"

    return SignalScore(
        name=name,
        score=score,
        reason="Struktur: " + text,
        label=label,
        tags=tuple(tags),
        details={"last_swing_high": h2, "last_swing_low": l2, "prev_swing_high": h1, "prev_swing_low": l1},
    )
