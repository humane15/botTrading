"""Trigger candle untuk timing entry di timeframe 5m.

Pola yang dikenali pada candle tertutup terakhir:
* Bullish engulfing: candle hijau menelan badan candle merah sebelumnya.
* Hammer: ekor bawah panjang (penolakan harga rendah), close di separuh atas.
* Breakout candle: badan besar dan close di atas high candle sebelumnya.
Pola kebalikannya (bearish engulfing, shooting star, close di bawah low
sebelumnya) memberi skor negatif.
"""

from __future__ import annotations

import pandas as pd

from analysis.base import SignalScore, is_valid, last


def score_trigger_candle(frame: pd.DataFrame) -> SignalScore:
    name = "candle"
    if len(frame) < 2:
        return SignalScore.neutral(name, "data candle belum cukup")
    o, h, l, c = (last(frame, k) for k in ("open", "high", "low", "close"))
    po, ph, pl, pc = (last(frame, k, 2) for k in ("open", "high", "low", "close"))
    if not is_valid(o, h, l, c, po, ph, pl, pc):
        return SignalScore.neutral(name, "data candle belum cukup")
    full_range = h - l
    if full_range <= 0:
        return SignalScore.neutral(name, "candle tanpa pergerakan")

    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    bullish, bearish = c > o, c < o
    position = (c - l) / full_range  # 0 = close di low, 1 = close di high

    if bullish and pc < po and c >= po and o <= pc:
        return SignalScore(name, 0.7, "Bullish engulfing: candle naik menelan candle turun sebelumnya", "Bullish engulfing", ("bullish_engulfing",))
    if bearish and pc > po and c <= po and o >= pc:
        return SignalScore(name, -0.7, "Bearish engulfing: candle turun menelan candle naik sebelumnya", "", ("bearish_engulfing",))
    if lower_wick >= 2 * max(body, 1e-12) and upper_wick <= max(body, 0.1 * full_range) and position >= 0.5:
        return SignalScore(name, 0.5, "Hammer: ekor bawah panjang, harga rendah ditolak pembeli", "Hammer", ("hammer",))
    if upper_wick >= 2 * max(body, 1e-12) and lower_wick <= max(body, 0.1 * full_range) and position <= 0.5:
        return SignalScore(name, -0.5, "Shooting star: ekor atas panjang, harga tinggi ditolak penjual", "", ("shooting_star",))
    if bullish and body >= 0.6 * full_range and c > ph:
        return SignalScore(name, 0.5, "Candle breakout: badan besar dan close di atas high sebelumnya", "Breakout candle", ("breakout_candle",))
    if bearish and body >= 0.6 * full_range and c < pl:
        return SignalScore(name, -0.5, "Candle breakdown: badan besar dan close di bawah low sebelumnya", "", ("breakdown_candle",))
    if bullish:
        return SignalScore(name, 0.1, "Candle naik biasa")
    if bearish:
        return SignalScore(name, -0.1, "Candle turun biasa")
    return SignalScore.neutral(name, "Doji, belum ada arah")
