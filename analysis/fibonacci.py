"""Fibonacci retracement: titik pantul koreksi.

* Leg = swing low ke swing high terakhir yang signifikan (minimal 3 ATR) pada
  1h atau 15m. Puncak harus sudah lewat beberapa candle (sudah terkonfirmasi).
* Level 23.6%, 38.2%, 50%, 61.8% (golden ratio), dan 78.6%.
* Skor tertinggi jika harga memantul di zona 50% sampai 61.8% dan bertepatan
  dengan support atau EMA (confluence).
* Extension 127.2% dan 161.8% dipakai sebagai kandidat take profit.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.base import SignalScore, column_values, has_columns, is_valid, last, tail
from analysis.support_resistance import Zone

FIB_RATIOS = (0.236, 0.382, 0.5, 0.618, 0.786)
FIB_EXTENSIONS = (1.272, 1.618)


@dataclass(frozen=True)
class FibParams:
    lookback: int = 120
    confirm_bars: int = 3          # puncak minimal 3 candle lalu (sudah ada koreksi)
    min_swing_atr: float = 3.0     # leg signifikan jika >= 3 ATR
    tolerance: float = 0.03        # toleransi rasio di sekitar level
    bounce_bars: int = 3
    confluence_atr: float = 0.5    # level dianggap bertepatan jika jaraknya <= 0.5 ATR
    confluence_bonus: float = 0.3
    no_bounce_factor: float = 0.5  # belum memantul: skor dipotong setengah


@dataclass(frozen=True)
class FibLeg:
    low: float
    high: float
    low_position: int
    high_position: int
    timeframe: str = ""

    @property
    def size(self) -> float:
        return self.high - self.low

    def level(self, ratio: float) -> float:
        """Harga level retracement (0.618 = 61.8% turun dari puncak)."""
        return self.high - self.size * ratio

    def extension(self, ratio: float) -> float:
        """Harga level extension (1.272 = 127.2% dari leg, di atas puncak)."""
        return self.low + self.size * ratio

    def retracement(self, price: float) -> float:
        return (self.high - price) / self.size

    def extensions(self) -> tuple[float, ...]:
        return tuple(self.extension(r) for r in FIB_EXTENSIONS)


def find_fib_leg(frame: pd.DataFrame, timeframe: str = "", params: FibParams = FibParams()) -> FibLeg | None:
    """Leg naik terakhir: puncak tertinggi di jendela lookback dan low terendah sebelumnya."""
    if len(frame) < 10 or not has_columns(frame, "high", "low", "atr"):
        return None
    atr = last(frame, "atr")
    if not is_valid(atr) or atr <= 0:
        return None
    start = max(0, len(frame) - params.lookback)
    highs = column_values(frame, "high")[start:]
    lows = column_values(frame, "low")[start:]
    high_rel = int(np.argmax(highs))
    if high_rel == 0 or (len(highs) - 1 - high_rel) < params.confirm_bars:
        return None  # puncak di awal jendela (tidak ada low sebelumnya) atau masih membuat high baru
    low_rel = int(np.argmin(lows[: high_rel + 1]))
    if low_rel >= high_rel:
        return None
    leg = FibLeg(
        low=float(lows[low_rel]),
        high=float(highs[high_rel]),
        low_position=start + low_rel,
        high_position=start + high_rel,
        timeframe=timeframe,
    )
    if leg.size < params.min_swing_atr * atr:
        return None
    return leg


def _zone_for(ratio: float, tol: float) -> tuple[str, float, float] | None:
    """(tag, rasio level acuan, skor dasar) untuk kedalaman koreksi `ratio`."""
    if ratio < 0.236 - tol:
        return None
    if ratio < 0.382 - tol:
        return "fib_236", 0.236, 0.15
    if ratio < 0.5 - tol:
        return "fib_382", 0.382, 0.4
    if ratio < (0.5 + 0.618) / 2:
        return "fib_50", 0.5, 0.6
    if ratio <= 0.618 + tol:
        return "fib_618", 0.618, 0.6
    if ratio <= 0.786 + tol:
        return "fib_786", 0.786, 0.3
    return "fib_terlalu_dalam", 1.0, -0.2


def score_fibonacci(
    frame: pd.DataFrame,
    leg: FibLeg | None,
    support_zones: Sequence[Zone] = (),
    ema_levels: Iterable[float] = (),
    params: FibParams = FibParams(),
) -> SignalScore:
    """Skor pantulan Fibonacci memakai price action candle terakhir dari `frame`."""
    name = "fib"
    if leg is None:
        return SignalScore.neutral(name, "tidak ada swing signifikan untuk Fibonacci")
    if frame.empty or not has_columns(frame, "close", "open", "low", "atr"):
        return SignalScore.neutral(name, "data Fibonacci belum cukup")
    close, open_, atr = last(frame, "close"), last(frame, "open"), last(frame, "atr")
    if not is_valid(close, open_, atr) or atr <= 0:
        return SignalScore.neutral(name, "data Fibonacci belum cukup")

    deepest = float(tail(frame, "low", params.bounce_bars).min())
    r_now = leg.retracement(close)
    r_deep = leg.retracement(deepest)
    details = {
        "leg_low": leg.low,
        "leg_high": leg.high,
        "leg_timeframe": leg.timeframe,
        "retracement": r_now,
        "deepest_retracement": r_deep,
        "ext_1272": leg.extension(1.272),
        "ext_1618": leg.extension(1.618),
    }
    leg_text = f"leg {leg.timeframe} {leg.low:.6g}-{leg.high:.6g}".replace("  ", " ")

    if r_now > 1.0:
        return SignalScore(name, -0.5, f"Fibonacci: harga di bawah swing low {leg_text}, leg patah", tags=("fib_patah",), details=details)
    zone = _zone_for(r_deep, params.tolerance)
    if zone is None:
        return SignalScore(name, 0.0, f"Fibonacci: belum ada koreksi berarti ({r_deep:.1%}) pada {leg_text}", details=details)
    tag, ratio, base = zone
    ratio_text = f"{ratio * 100:.1f}%".replace(".0%", "%")
    if base < 0:
        return SignalScore(name, base, f"Fibonacci: koreksi terlalu dalam ({r_deep:.1%}) pada {leg_text}", tags=(tag,), details=details)

    bounce = close > open_ and r_now < r_deep
    score = base if bounce else base * params.no_bounce_factor
    tags = [tag]
    level_price = leg.level(ratio)
    near = params.confluence_atr * atr
    support_hits = [
        f"support {zone_.timeframe}".strip()
        for zone_ in support_zones
        if zone_.low - near <= level_price <= zone_.high + near
    ]
    ema_hits = [f"EMA {ema:.6g}" for ema in ema_levels if is_valid(ema) and abs(ema - level_price) <= near]
    confluence = support_hits + ema_hits
    if confluence:
        score += params.confluence_bonus
        tags.append("fib_confluence")

    # Contoh label: "Pantulan Fib 61.8% + support 1h" (timeframe = asal leg Fibonacci).
    label = f"{'Pantulan ' if bounce else ''}Fib {ratio_text}"
    if support_hits:
        label += " + support"
    elif ema_hits:
        label += " + EMA"
    if leg.timeframe:
        label += f" {leg.timeframe}"
    text = f"Fibonacci: koreksi {r_deep:.1%} ke level {ratio_text} ({level_price:.6g}) pada {leg_text}"
    text += ", sudah memantul" if bounce else ", belum ada pantulan"
    if confluence:
        text += ", bertepatan dengan " + ", ".join(dict.fromkeys(confluence))
    details.update({"level": ratio, "level_price": level_price, "bounce": bounce, "confluence": bool(confluence)})
    return SignalScore(name=name, score=score, reason=text, label=label, tags=tuple(tags), details=details)
