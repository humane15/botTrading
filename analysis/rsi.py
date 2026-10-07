"""RSI 14 (kejenuhan pasar) yang dibaca sesuai konteks tren.

* Pasar sideways: di bawah 30 oversold (bullish), di atas 70 overbought (bearish).
* Uptrend kuat: RSI wajar bertahan di 60 sampai 80, jadi RSI tinggi tidak
  otomatis bearish; area 40 sampai 55 justru pullback sehat. RSI di atas 80
  ditandai rawan entry terlambat.
* Divergence: bullish jika harga membuat lower low tetapi RSI higher low,
  bearish jika harga higher high tetapi RSI lower high.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from analysis.base import SignalScore, has_columns, is_valid, last
from analysis.moving_average import TREND_DOWN, TREND_UP
from analysis.pivots import find_pivots


@dataclass(frozen=True)
class RSIParams:
    pivot_left: int = 3
    pivot_right: int = 3
    divergence_lookback: int = 60
    divergence_recent: int = 12   # pivot kedua maksimal 12 candle dari candle terakhir
    min_pivot_gap: int = 5
    divergence_bonus: float = 0.4


def detect_divergence(frame: pd.DataFrame, params: RSIParams = RSIParams()) -> str | None:
    """'bullish', 'bearish', atau None berdasarkan dua pivot terakhir."""
    if not has_columns(frame, "rsi") or len(frame) < params.pivot_left + params.pivot_right + 2:
        return None
    highs, lows = find_pivots(frame, params.pivot_left, params.pivot_right, params.divergence_lookback)
    rsi = frame["rsi"].to_numpy(dtype="float64")
    last = len(frame) - 1
    found: list[tuple[int, str]] = []

    if len(lows) >= 2:
        p1, p2 = lows[-2], lows[-1]
        r1, r2 = rsi[p1.position], rsi[p2.position]
        if (
            last - p2.position <= params.divergence_recent
            and p2.position - p1.position >= params.min_pivot_gap
            and is_valid(r1, r2)
            and p2.price < p1.price
            and r2 > r1
            and r1 < 50
        ):
            found.append((p2.position, "bullish"))
    if len(highs) >= 2:
        p1, p2 = highs[-2], highs[-1]
        r1, r2 = rsi[p1.position], rsi[p2.position]
        if (
            last - p2.position <= params.divergence_recent
            and p2.position - p1.position >= params.min_pivot_gap
            and is_valid(r1, r2)
            and p2.price > p1.price
            and r2 < r1
            and r1 > 50
        ):
            found.append((p2.position, "bearish"))
    if not found:
        return None
    return max(found)[1]  # pilih divergence yang paling baru


def _base_score(value: float, trend: str) -> tuple[float, str, list[str], str]:
    """Skor dasar RSI sesuai konteks tren: (skor, penjelasan, tag, label)."""
    if trend == TREND_UP:
        if value >= 80:
            return -0.5, "sangat jenuh beli walau dalam uptrend, rawan entry terlambat", ["rsi_overbought"], ""
        if value >= 70:
            return 0.0, "tinggi tetapi wajar dalam uptrend kuat", [], ""
        if value >= 55:
            return 0.2, "momentum uptrend sehat", [], ""
        if value >= 40:
            return 0.5, "pullback sehat dalam uptrend", ["rsi_pullback"], "RSI pullback"
        if value >= 30:
            return 0.3, "pullback dalam pada uptrend", ["rsi_pullback"], "RSI pullback dalam"
        return 0.1, "oversold dalam uptrend, tren bisa melemah", ["rsi_oversold"], ""
    if trend == TREND_DOWN:
        if value < 30:
            return 0.0, "oversold dalam downtrend, bisa terus turun", ["rsi_oversold"], ""
        if value >= 60:
            return -0.4, "memantul ke area tinggi dalam downtrend", [], ""
        return -0.2, "momentum masih condong turun", [], ""
    if value <= 30:
        return 0.6, "oversold di pasar sideways", ["rsi_oversold"], "RSI oversold"
    if value <= 40:
        return 0.3, "mendekati oversold", [], ""
    if value < 60:
        return 0.0, "netral", [], ""
    if value < 70:
        return -0.3, "mendekati overbought", [], ""
    return -0.6, "overbought di pasar sideways", ["rsi_overbought"], ""


def score_rsi(frame: pd.DataFrame, trend: str = "sideways", params: RSIParams = RSIParams()) -> SignalScore:
    name = "rsi"
    value = last(frame, "rsi")
    if not is_valid(value):
        return SignalScore.neutral(name, "data RSI belum cukup")
    score, text, tags, label = _base_score(value, trend)

    divergence = detect_divergence(frame, params)
    if divergence == "bullish":
        score += params.divergence_bonus
        tags.insert(0, "rsi_bull_div")
        label = "RSI bullish divergence"
        text += ", bullish divergence (harga lower low, RSI higher low)"
    elif divergence == "bearish":
        score -= params.divergence_bonus
        tags.append("rsi_bear_div")
        text += ", bearish divergence (harga higher high, RSI lower high)"

    return SignalScore(
        name=name,
        score=score,
        reason=f"RSI {value:.1f}: {text}",
        label=label,
        tags=tuple(tags),
        details={"rsi": value, "divergence": divergence or "", "trend_context": trend},
    )
