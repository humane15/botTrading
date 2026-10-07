"""Bollinger Bands 20, 2 (volatilitas).

* Bandwidth dan persentilnya dalam 100 candle terakhir. Persentil di bawah
  20% disebut squeeze: volatilitas sangat rendah, kemungkinan breakout.
* Breakout di atas band atas dengan volume tinggi setelah squeeze adalah
  sinyal kuat. Breakout tanpa volume ditandai lemah (rawan false breakout).
* Di pasar ranging, harga di dekat band bawah adalah peluang beli dan di dekat
  band atas adalah area jual (mean reversion).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from analysis.base import SignalScore, has_columns, is_valid, last, tail


@dataclass(frozen=True)
class BollingerParams:
    squeeze_pct: float = 0.20
    squeeze_lookback: int = 10      # squeeze dihitung jika terjadi dalam 10 candle terakhir
    breakout_volume_ratio: float = 1.5
    band_touch_pctb: float = 0.10


def score_bollinger(frame: pd.DataFrame, ranging: bool = False, params: BollingerParams = BollingerParams()) -> SignalScore:
    name = "bb"
    required = ("close", "bb_upper", "bb_lower", "bb_pctb", "bb_width_rank", "volume_ratio")
    if frame.empty or not has_columns(frame, *required):
        return SignalScore.neutral(name, "data Bollinger belum cukup")
    close, upper, lower, pct_b = (last(frame, c) for c in ("close", "bb_upper", "bb_lower", "bb_pctb"))
    if not is_valid(close, upper, lower, pct_b):
        return SignalScore.neutral(name, "data Bollinger belum cukup")
    width_rank = last(frame, "bb_width_rank")
    volume_ratio = last(frame, "volume_ratio")
    volume_ratio = volume_ratio if is_valid(volume_ratio) else 1.0
    # Squeeze dinilai juga SEBELUM candle terakhir agar candle breakout (yang
    # melebarkan band) tidak menghapus jejak squeeze-nya sendiri.
    ranks = tail(frame, "bb_width_rank", params.squeeze_lookback + 1)
    squeeze_now = is_valid(width_rank) and width_rank <= params.squeeze_pct
    squeeze_recent = bool((ranks[:-1] <= params.squeeze_pct).any()) or squeeze_now
    high_volume = volume_ratio >= params.breakout_volume_ratio

    tags: list[str] = []
    label = ""
    if close > upper:
        if high_volume and squeeze_recent:
            score, text = 0.8, f"breakout band atas setelah squeeze dengan volume {volume_ratio:.1f}x"
            tags += ["bb_squeeze_breakout", "bb_breakout"]
            label = "BB squeeze breakout"
        elif high_volume:
            score, text = 0.4, f"breakout band atas dengan volume {volume_ratio:.1f}x"
            tags.append("bb_breakout")
            label = "Breakout BB"
        else:
            score, text = 0.1, f"breakout band atas tanpa volume ({volume_ratio:.1f}x), rawan palsu"
            tags.append("bb_breakout_lemah")
    elif close < lower:
        if high_volume:
            score, text = (-0.8 if squeeze_recent else -0.5), f"breakdown band bawah dengan volume {volume_ratio:.1f}x"
            tags.append("bb_breakdown")
        elif ranging:
            score, text = 0.2, "di bawah band bawah tanpa volume di pasar ranging (jenuh jual)"
            tags.append("bb_lower_touch")
        else:
            score, text = -0.3, "ditutup di bawah band bawah"
    elif ranging and pct_b <= params.band_touch_pctb:
        score, text = 0.3, "dekat band bawah di pasar ranging"
        tags.append("bb_lower_touch")
        label = "Band bawah BB"
    elif ranging and pct_b >= 1 - params.band_touch_pctb:
        score, text = -0.3, "dekat band atas di pasar ranging"
        tags.append("bb_upper_touch")
    else:
        score, text = 0.0, f"di dalam band (%B {pct_b:.2f})"

    if squeeze_now:
        tags.append("bb_squeeze")
        text += f", squeeze aktif (bandwidth persentil {width_rank:.0%}), potensi breakout"
    return SignalScore(
        name=name,
        score=score,
        reason="Bollinger: " + text,
        label=label,
        tags=tuple(tags),
        details={
            "pct_b": pct_b,
            "bb_width_rank": width_rank,
            "squeeze": squeeze_now,
            "squeeze_recent": squeeze_recent,
            "volume_ratio": volume_ratio,
        },
    )
