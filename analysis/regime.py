"""Deteksi kondisi pasar (market regime) dan circuit breaker BTC.

Aturan klasifikasi (dihitung pada timeframe 1h):

* Trending Up: ADX >= 25, EMA tersusun naik (20 > 50 > 200) dan +DI >= -DI.
* Trending Down: ADX >= 25 dengan EMA tersusun turun (atau -DI dominan dan
  harga di bawah EMA 50), atau ADX 20 sampai 25 dengan EMA tersusun turun.
  Regime ini melarang posisi baru karena bot spot tidak bisa short.
* Ranging: ADX < 20, atau tren belum jelas.
* High Volatility: ATR (% dari harga) berada di persentil 90% ke atas.
  Ini lapisan tambahan di atas regime tren: ukuran posisi dipotong 50%.

Circuit breaker: jika BTC turun lebih dari 3% dalam 1 jam, entry baru
dihentikan selama 2 jam. Penurunan diukur dari harga penutupan tertinggi
dalam 1 jam terakhir ke harga sekarang, sehingga crash di tengah jam yang
sempat memantul tetap terdeteksi.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from analysis.base import is_valid, last

log = logging.getLogger(__name__)

TRENDING_UP = "trending_up"
RANGING = "ranging"
TRENDING_DOWN = "trending_down"

_TREND_LABELS = {TRENDING_UP: "Trending Up", RANGING: "Ranging", TRENDING_DOWN: "Trending Down"}


@dataclass(frozen=True)
class RegimeParams:
    adx_trend: float = 25.0
    adx_range: float = 20.0
    high_vol_pct: float = 0.90
    high_vol_size_multiplier: float = 0.5


@dataclass(frozen=True)
class Regime:
    trend: str
    high_volatility: bool
    adx: float
    atr_pct_rank: float
    reason: str
    size_multiplier: float = 1.0

    @property
    def label(self) -> str:
        text = _TREND_LABELS.get(self.trend, self.trend)
        return text + (" + High Volatility" if self.high_volatility else "")

    @property
    def allows_entry(self) -> bool:
        return self.trend != TRENDING_DOWN

    @property
    def strategy(self) -> str:
        """trending: beli pullback; ranging: beli di support, jual di resistance."""
        return {TRENDING_UP: "pullback", RANGING: "range"}.get(self.trend, "tidak_entry")

    def code(self) -> int:
        """Kode numerik untuk fitur ML: 1 naik, 0 ranging, -1 turun."""
        return {TRENDING_UP: 1, RANGING: 0, TRENDING_DOWN: -1}[self.trend]


def classify_regime(frame: pd.DataFrame, params: RegimeParams = RegimeParams()) -> Regime:
    if frame.empty:
        return Regime(RANGING, False, float("nan"), float("nan"), "data belum ada")
    adx, plus_di, minus_di = (last(frame, c) for c in ("adx", "plus_di", "minus_di"))
    close, e20, e50, e200 = (last(frame, c) for c in ("close", "ema20", "ema50", "ema200"))
    atr_rank = last(frame, "atr_pct_rank")
    high_vol = is_valid(atr_rank) and atr_rank >= params.high_vol_pct
    multiplier = params.high_vol_size_multiplier if high_vol else 1.0
    if not is_valid(adx, plus_di, minus_di, close, e20, e50):
        return Regime(RANGING, high_vol, adx, atr_rank, "data ADX/EMA belum cukup", multiplier)

    has200 = is_valid(e200)
    aligned_up = e20 > e50 and (not has200 or e50 > e200)
    aligned_down = e20 < e50 and (not has200 or e50 < e200)
    if adx >= params.adx_trend:
        if aligned_up and plus_di >= minus_di:
            trend, why = TRENDING_UP, f"ADX {adx:.1f} >= {params.adx_trend:g} dan EMA tersusun naik"
        elif aligned_down or (minus_di > plus_di and close < e50):
            trend, why = TRENDING_DOWN, f"ADX {adx:.1f} >= {params.adx_trend:g} dengan tekanan turun"
        else:
            trend, why = RANGING, f"ADX {adx:.1f} tinggi tetapi arah tren belum jelas"
    elif adx >= params.adx_range:
        if aligned_down and close < e50:
            trend, why = TRENDING_DOWN, f"ADX {adx:.1f} (transisi) dengan EMA tersusun turun"
        else:
            trend, why = RANGING, f"ADX {adx:.1f} di zona transisi, tren belum terkonfirmasi"
    else:
        trend, why = RANGING, f"ADX {adx:.1f} < {params.adx_range:g}, pasar bergerak menyamping"
    if high_vol:
        why += f", ATR di persentil {atr_rank:.0%} (volatilitas tinggi, ukuran posisi x{multiplier:g})"
    return Regime(trend, high_vol, adx, atr_rank, why, multiplier)


@dataclass
class CircuitBreaker:
    """Hentikan entry baru selama `cooldown` jika BTC turun >= `drop_threshold` dalam `window`."""

    drop_threshold: float = 0.03
    window: pd.Timedelta = field(default_factory=lambda: pd.Timedelta(hours=1))
    cooldown: pd.Timedelta = field(default_factory=lambda: pd.Timedelta(hours=2))
    active_until: pd.Timestamp | None = None
    last_drop: float = 0.0
    last_change: float = 0.0

    def update(self, btc: pd.DataFrame, now: pd.Timestamp | None = None) -> bool:
        """Perbarui status dari candle BTC tertutup (disarankan 5m). Mengembalikan status aktif."""
        if len(btc) < 2:
            return self.is_active(now)
        step = btc.index[-1] - btc.index[-2]
        close_time = btc.index[-1] + step
        now = close_time if now is None else now
        recent = btc[btc.index >= btc.index[-1] - self.window]
        closes = recent["close"]
        peak, last, first = float(closes.max()), float(closes.iloc[-1]), float(closes.iloc[0])
        self.last_drop = (peak - last) / peak if peak > 0 else 0.0
        self.last_change = last / first - 1 if first > 0 else 0.0
        if self.last_drop >= self.drop_threshold:
            until = close_time + self.cooldown
            if self.active_until is None or until > self.active_until:
                log.warning(
                    "Circuit breaker aktif: BTC turun %.2f%% dalam 1 jam, entry baru dihentikan sampai %s",
                    self.last_drop * 100, until,
                )
                self.active_until = until
        return self.is_active(now)

    def is_active(self, now: pd.Timestamp | None) -> bool:
        return self.active_until is not None and now is not None and now < self.active_until


@dataclass(frozen=True)
class MarketContext:
    """Kondisi pasar umum dengan BTC sebagai acuan."""

    btc_regime: Regime | None = None
    circuit_breaker_active: bool = False
    btc_change_1h: float = 0.0
    btc_drop_1h: float = 0.0
    breaker_until: pd.Timestamp | None = None

    @property
    def label(self) -> str:
        return self.btc_regime.label if self.btc_regime else "tidak diketahui"


def analyze_market(
    btc_frames: Mapping[str, pd.DataFrame],
    breaker: CircuitBreaker,
    trend_timeframe: str = "1h",
    breaker_timeframe: str = "5m",
    now: pd.Timestamp | None = None,
    params: RegimeParams = RegimeParams(),
) -> MarketContext:
    """Bangun konteks pasar dari frame indikator BTC (timeframe tren dan 5m)."""
    regime = classify_regime(btc_frames[trend_timeframe], params) if trend_timeframe in btc_frames else None
    active = breaker.update(btc_frames[breaker_timeframe], now) if breaker_timeframe in btc_frames else breaker.is_active(now)
    return MarketContext(
        btc_regime=regime,
        circuit_breaker_active=active,
        btc_change_1h=breaker.last_change,
        btc_drop_1h=breaker.last_drop,
        breaker_until=breaker.active_until,
    )
