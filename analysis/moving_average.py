"""Moving Average sebagai garis tren: EMA 20, 50, 200.

* Harga di atas EMA dan EMA tersusun 20 > 50 > 200 berarti uptrend.
* Golden cross (EMA 50 memotong ke atas EMA 200) dan death cross.
* Kemiringan (slope) EMA 50, diukur dalam satuan ATR, untuk kekuatan tren.
* Pullback ke EMA 20/50 dalam uptrend ditandai sebagai setup (regime trending).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.base import SignalScore, has_columns, is_valid, last, tail

TREND_UP = "up"
TREND_DOWN = "down"
TREND_SIDEWAYS = "sideways"


@dataclass(frozen=True)
class MAParams:
    slope_lookback: int = 10     # slope EMA 50 diukur selama 10 candle
    cross_lookback: int = 20     # golden/death cross dianggap baru jika terjadi <= 20 candle lalu
    pullback_atr: float = 0.5    # low menyentuh EMA (+0.5 ATR) lalu close di atasnya = pullback
    pullback_bars: int = 3


def _sign(value: float) -> float:
    return 1.0 if value > 0 else -1.0 if value < 0 else 0.0


def _recent_cross(frame: pd.DataFrame, fast: str, slow: str, lookback: int) -> int:
    """+1 jika `fast` baru memotong ke atas `slow`, -1 jika ke bawah, 0 jika tidak ada."""
    diff = tail(frame, fast, lookback + 1) - tail(frame, slow, lookback + 1)
    diff = diff[~np.isnan(diff)]
    result = 0
    for prev, curr in zip(diff[:-1], diff[1:], strict=True):
        if prev <= 0 < curr:
            result = 1
        elif prev >= 0 > curr:
            result = -1
    return result


def score_moving_average(frame: pd.DataFrame, params: MAParams = MAParams()) -> SignalScore:
    name = "ma"
    if len(frame) < 2 or not has_columns(frame, "ema20", "ema50", "ema200", "atr"):
        return SignalScore.neutral(name, "data EMA belum cukup", trend=TREND_SIDEWAYS)
    close, e20, e50, e200, atr = (last(frame, c) for c in ("close", "ema20", "ema50", "ema200", "atr"))
    if not is_valid(close, e20, e50, atr) or atr <= 0:
        return SignalScore.neutral(name, "data EMA belum cukup", trend=TREND_SIDEWAYS)
    has200 = is_valid(e200)

    points = [_sign(close - e20), _sign(e20 - e50)]
    if has200:
        points += [_sign(e50 - e200), _sign(close - e200)]
    alignment = sum(points) / len(points)

    k = params.slope_lookback
    e50_prev = last(frame, "ema50", k + 1)
    slope_atr = (e50 - e50_prev) / atr if is_valid(e50_prev) else 0.0
    cross = _recent_cross(frame, "ema50", "ema200", params.cross_lookback) if has200 else 0

    score = 0.6 * alignment + 0.4 * math.tanh(slope_atr) + 0.2 * cross

    aligned_up = e20 > e50 and (not has200 or e50 > e200)
    aligned_down = e20 < e50 and (not has200 or e50 < e200)
    if aligned_up and close > e50 and slope_atr > 0:
        trend = TREND_UP
    elif aligned_down and close < e50 and slope_atr < 0:
        trend = TREND_DOWN
    else:
        trend = TREND_SIDEWAYS

    tags: list[str] = []
    label = ""
    if trend == TREND_UP:
        tags.append("ema_uptrend")
        label = "EMA uptrend"
        recent_low = float(tail(frame, "low", params.pullback_bars).min())
        # Tag setup diletakkan paling depan: tag pertama = kunci pola untuk memori pola.
        if recent_low <= e20 + params.pullback_atr * atr and close > e20:
            tags.insert(0, "pullback_ema20")
            label = "Pullback EMA 20"
        elif recent_low <= e50 + params.pullback_atr * atr and close > e50:
            tags.insert(0, "pullback_ema50")
            label = "Pullback EMA 50"
    elif trend == TREND_DOWN:
        tags.append("ema_downtrend")
    if cross > 0:
        tags.append("golden_cross")
        label = label or "Golden cross EMA 50/200"
    elif cross < 0:
        tags.append("death_cross")

    order = f"EMA20 {'>' if e20 > e50 else '<'} EMA50"
    if has200:
        order += f" {'>' if e50 > e200 else '<'} EMA200"
    reason = (
        f"{order}, harga {'di atas' if close > e20 else 'di bawah'} EMA20, "
        f"slope EMA50 {slope_atr:+.2f} ATR per {k} candle, tren {trend}"
    )
    if cross > 0:
        reason += ", golden cross baru terjadi"
    elif cross < 0:
        reason += ", death cross baru terjadi"
    return SignalScore(
        name=name,
        score=score,
        reason=reason,
        label=label,
        tags=tuple(tags),
        details={
            "trend": trend,
            "alignment": alignment,
            "slope_atr": slope_atr,
            "dist_ema20_atr": (close - e20) / atr,
            "dist_ema50_atr": (close - e50) / atr,
            "dist_ema200_atr": (close - e200) / atr if has200 else float("nan"),
            "golden_cross": cross > 0,
            "death_cross": cross < 0,
        },
    )
