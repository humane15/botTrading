"""Support & Resistance: peta lantai dan atap harga.

* Swing high dan swing low (pivot) dideteksi pada 1h dan 15m.
* Pivot yang berdekatan (dalam 0.5 x ATR) digabung menjadi satu zona. Pivot
  high dan low digabung bersama karena resistance yang ditembus sering
  berubah menjadi support (role reversal).
* Kekuatan zona (0..1) = jumlah sentuhan (50%) + volume di sekitarnya (25%)
  + seberapa baru sentuhan terakhir (25%).
* Skor positif jika harga memantul dari support kuat, negatif jika harga
  mendekati resistance kuat (ruang naik sempit).
* Entry ditolak jika jarak ke resistance < 1.5 x jarak ke stop loss.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.base import SignalScore, has_columns, is_valid, last, tail
from analysis.pivots import Pivot, find_pivots


@dataclass(frozen=True)
class SRParams:
    pivot_left: int = 5
    pivot_right: int = 5
    lookback: int = 300
    cluster_atr: float = 0.5        # pivot dalam 0.5 ATR digabung jadi satu zona
    min_zone_atr: float = 0.2       # ketebalan minimal zona
    half_life_bars: float = 100.0   # kebaruan: bobot setengah setelah 100 candle
    touch_cap: int = 4              # 4 sentuhan atau lebih = skor sentuhan penuh
    near_atr: float = 1.0           # "dekat" jika jarak <= 1 ATR
    touch_buffer_atr: float = 0.25  # low menyentuh zona jika <= batas atas zona + 0.25 ATR
    bounce_bars: int = 3
    min_strength: float = 0.3       # zona lebih lemah diabaikan untuk support/resistance terdekat


@dataclass(frozen=True)
class Zone:
    low: float
    high: float
    touches: int
    strength: float
    last_touch_position: int
    volume_score: float = 0.0
    recency: float = 0.0
    timeframe: str = ""

    @property
    def center(self) -> float:
        return (self.low + self.high) / 2

    def describe(self) -> str:
        tf = f" {self.timeframe}" if self.timeframe else ""
        return f"zona{tf} {self.low:.6g}-{self.high:.6g} ({self.touches} sentuhan, kekuatan {self.strength:.2f})"


def _build_zone(cluster: Sequence[Pivot], atr: float, last_position: int, timeframe: str, params: SRParams) -> Zone:
    prices = [p.price for p in cluster]
    low, high = min(prices), max(prices)
    pad = max(0.0, (params.min_zone_atr * atr - (high - low)) / 2)
    ratios = [p.volume_ratio for p in cluster if is_valid(p.volume_ratio)]
    volume_score = min(float(np.mean(ratios)) / 2.0, 1.0) if ratios else 0.5
    last_touch = max(p.position for p in cluster)
    recency = math.exp(-math.log(2) * (last_position - last_touch) / params.half_life_bars)
    touch_score = min(len(cluster), params.touch_cap) / params.touch_cap
    strength = 0.5 * touch_score + 0.25 * volume_score + 0.25 * recency
    return Zone(
        low=low - pad,
        high=high + pad,
        touches=len(cluster),
        strength=round(strength, 4),
        last_touch_position=last_touch,
        volume_score=volume_score,
        recency=recency,
        timeframe=timeframe,
    )


def find_zones(frame: pd.DataFrame, timeframe: str = "", params: SRParams = SRParams()) -> list[Zone]:
    """Kelompokkan pivot menjadi zona S&R, terurut dari harga terendah."""
    if frame.empty or not has_columns(frame, "atr", "high", "low"):
        return []
    atr = last(frame, "atr")
    if not is_valid(atr) or atr <= 0:
        return []
    highs, lows = find_pivots(frame, params.pivot_left, params.pivot_right, params.lookback)
    pivots = sorted(highs + lows, key=lambda p: p.price)
    tolerance = params.cluster_atr * atr
    clusters: list[list[Pivot]] = []
    for pivot in pivots:
        # Rentang satu zona tidak boleh melebihi toleransi (mencegah zona "berantai").
        if clusters and pivot.price - clusters[-1][0].price <= tolerance:
            clusters[-1].append(pivot)
        else:
            clusters.append([pivot])
    last_position = len(frame) - 1
    return [_build_zone(cluster, atr, last_position, timeframe, params) for cluster in clusters]


def nearest_levels(price: float, zones: Iterable[Zone], min_strength: float = 0.0) -> tuple[Zone | None, Zone | None]:
    """(support terdekat di bawah harga, resistance terdekat di atas harga).

    Zona dengan titik tengah di bawah harga dianggap support, di atasnya resistance.
    """
    support: Zone | None = None
    resistance: Zone | None = None
    for zone in zones:
        if zone.strength < min_strength:
            continue
        if zone.center <= price:
            if support is None or zone.center > support.center:
                support = zone
        elif resistance is None or zone.center < resistance.center:
            resistance = zone
    return support, resistance


def suggest_stop_loss(
    entry: float,
    atr: float,
    support: Zone | None,
    min_atr: float = 1.5,
    max_atr: float = 2.5,
    buffer_atr: float = 0.2,
    default_atr: float = 2.0,
) -> float:
    """Stop di bawah zona support terdekat, dibatasi 1.5 sampai 2.5 x ATR.

    Support yang terlalu dekat membuat stop terlalu ketat (mudah tersapu noise),
    jadi jaraknya minimal 1.5 ATR; support yang terlalu jauh dibatasi 2.5 ATR.
    Tanpa support, stop diletakkan 2 ATR di bawah entry.
    """
    if support is not None and support.low < entry:
        distance = entry - (support.low - buffer_atr * atr)
        distance = min(max(distance, min_atr * atr), max_atr * atr)
    else:
        distance = default_atr * atr
    return entry - distance


def reward_risk(entry: float, stop: float, resistance: Zone | None) -> float:
    """Ruang naik ke resistance dibagi jarak ke stop. Tanpa resistance = tak terhingga."""
    risk = entry - stop
    if risk <= 0:
        return 0.0
    if resistance is None:
        return math.inf
    return max(resistance.low - entry, 0.0) / risk


def score_support_resistance(
    frame: pd.DataFrame, zones: Sequence[Zone], params: SRParams = SRParams()
) -> SignalScore:
    name = "sr"
    if frame.empty or not has_columns(frame, "atr", "close", "open", "low"):
        return SignalScore.neutral(name, "data S&R belum cukup")
    close, open_, atr = last(frame, "close"), last(frame, "open"), last(frame, "atr")
    if not is_valid(close, open_, atr) or atr <= 0:
        return SignalScore.neutral(name, "data S&R belum cukup")
    if not zones:
        return SignalScore.neutral(name, "belum ada zona S&R")

    support, resistance = nearest_levels(close, zones, params.min_strength)
    recent_low = float(tail(frame, "low", params.bounce_bars).min())
    previous_closes = tail(frame, "close", params.bounce_bars + 1)[:-1]
    lowest_previous = float(previous_closes.min()) if previous_closes.size else float("inf")
    score = 0.0
    tags: list[str] = []
    parts: list[str] = []
    label = ""

    if support is not None:
        dist_support = (close - support.high) / atr
        touched = recent_low <= support.high + params.touch_buffer_atr * atr
        # Pantulan = support diuji dari atas. Jika candle sebelumnya close di bawah
        # zona, itu breakout dari bawah (dinilai terpisah), bukan pantulan.
        held = lowest_previous >= support.low
        closed_above = close > support.high or (close >= support.low and close > open_)
        if touched and held and closed_above:
            score += support.strength
            tags.append("pantulan_support")
            label = "Pantulan support"
            parts.append(f"memantul dari support {support.describe()}")
        elif 0 <= dist_support <= params.near_atr:
            score += 0.3 * support.strength
            tags.append("dekat_support")
            parts.append(f"dekat support {support.describe()}")

    if resistance is not None:
        dist_resistance = (resistance.low - close) / atr
        if dist_resistance <= params.near_atr:
            score -= resistance.strength * (1 - max(dist_resistance, 0.0) / params.near_atr)
            tags.append("dekat_resistance")
            parts.append(f"mendekati resistance {resistance.describe()}")

    # Breakout: zona yang beberapa candle lalu masih di atas close kini sudah ditembus.
    for zone in zones:
        if zone.strength >= params.min_strength and zone.high < close and lowest_previous < zone.low:
            score += 0.5 * zone.strength
            if label:
                tags.append("breakout_resistance")
            else:
                tags.insert(0, "breakout_resistance")
                label = "Breakout resistance"
            parts.append(f"menembus {zone.describe()}")
            break

    details = {
        "support_low": support.low if support else float("nan"),
        "support_high": support.high if support else float("nan"),
        "support_strength": support.strength if support else 0.0,
        "resistance_low": resistance.low if resistance else float("nan"),
        "resistance_high": resistance.high if resistance else float("nan"),
        "resistance_strength": resistance.strength if resistance else 0.0,
        "dist_support_atr": (close - support.high) / atr if support else float("nan"),
        "dist_resistance_atr": (resistance.low - close) / atr if resistance else float("nan"),
        "zones": len(zones),
    }
    if not parts:
        parts.append("harga di antara zona S&R")
    return SignalScore(name=name, score=score, reason="S&R: " + "; ".join(parts), label=label, tags=tuple(tags), details=details)
