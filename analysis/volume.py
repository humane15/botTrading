"""Volume: rasio terhadap rata rata 20 candle sebelumnya dan arah OBV.

* Lonjakan volume (>= 2x) pada candle naik = minat beli kuat; pada candle
  turun = tekanan jual.
* Volume lemah (< 0.7x) ditandai karena breakout tanpa volume rawan gagal.
* OBV di atas EMA-nya berarti akumulasi, di bawahnya distribusi.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from analysis.base import SignalScore, has_columns, is_valid, last


@dataclass(frozen=True)
class VolumeParams:
    spike_ratio: float = 2.0
    weak_ratio: float = 0.7
    spike_weight: float = 0.5
    obv_weight: float = 0.3


def score_volume(frame: pd.DataFrame, params: VolumeParams = VolumeParams()) -> SignalScore:
    name = "volume"
    if frame.empty or not has_columns(frame, "volume_ratio", "obv", "obv_ema", "open", "close"):
        return SignalScore.neutral(name, "data volume belum cukup")
    ratio = last(frame, "volume_ratio")
    if not is_valid(ratio):
        return SignalScore.neutral(name, "data volume belum cukup")
    bullish_candle = last(frame, "close") > last(frame, "open")

    score = 0.0
    tags: list[str] = []
    parts = [f"volume {ratio:.1f}x rata rata"]
    label = ""
    if ratio >= params.spike_ratio:
        if bullish_candle:
            score += params.spike_weight
            tags.append("volume_spike")
            label = f"Volume {ratio:.1f}x"
            parts.append("lonjakan volume beli")
        else:
            score -= params.spike_weight
            tags.append("volume_spike_jual")
            parts.append("lonjakan volume jual")
    elif ratio < params.weak_ratio:
        tags.append("volume_lemah")
        parts.append("volume lemah")

    obv, obv_ema = last(frame, "obv"), last(frame, "obv_ema")
    if is_valid(obv, obv_ema):
        if obv > obv_ema:
            score += params.obv_weight
            tags.append("obv_naik")
            parts.append("OBV di atas EMA (akumulasi)")
        else:
            score -= params.obv_weight
            tags.append("obv_turun")
            parts.append("OBV di bawah EMA (distribusi)")
    return SignalScore(
        name=name,
        score=score,
        reason="Volume: " + ", ".join(parts),
        label=label,
        tags=tuple(tags),
        details={"volume_ratio": ratio, "obv_above_ema": is_valid(obv, obv_ema) and obv > obv_ema},
    )
