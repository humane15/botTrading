"""Confluence engine: gabungan analisis multi timeframe menjadi skor 0..100.

Peran timeframe (top down seperti trader profesional):
* trend (1h): arah tren utama dan S&R besar. BUY hanya jika tren 1h tidak bearish.
* structure (30m): konfirmasi struktur pasar (higher high / higher low).
* setup (15m): pullback ke support, Fibonacci, BB squeeze.
* trigger (5m): timing entry (trigger candle, MACD cross, lonjakan volume).

Perhitungan:
1. Skor timeframe = rata rata tertimbang skor modulnya (bobot dari database).
2. Skor mentah = rata rata tertimbang skor timeframe (-1..1).
3. Skor akhir = 50 x (1 + skor mentah - penalti), dibatasi 0..100.

Gerbang entry (semua harus lolos):
* minimal 3 dari 4 timeframe searah (bullish);
* tren 1h tidak bearish dan regime coin bukan Trending Down;
* circuit breaker BTC tidak aktif;
* ada setup yang sesuai regime (trending: pullback/Fibonacci/EMA,
  ranging: beli di support);
* ruang ke resistance >= 1.5 x jarak ke stop loss;
* skor akhir >= skor minimum.

Fungsi evaluate() murni: hasilnya hanya bergantung pada frame yang diberikan,
sehingga backtest dan live memakai kode sinyal yang sama persis.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from analysis.base import SignalScore, is_valid, last
from analysis.bollinger import BollingerParams, score_bollinger
from analysis.candles import score_trigger_candle
from analysis.fibonacci import FibLeg, FibParams, find_fib_leg, score_fibonacci
from analysis.indicators import DEFAULT_PARAMS, IndicatorParams, compute_indicators
from analysis.macd import MACDParams, score_macd
from analysis.moving_average import TREND_DOWN, TREND_UP, MAParams, score_moving_average
from analysis.regime import (
    RANGING,
    TRENDING_UP,
    MarketContext,
    Regime,
    RegimeParams,
    classify_regime,
)
from analysis.rsi import RSIParams, score_rsi
from analysis.structure import StructureParams, score_structure
from analysis.support_resistance import (
    SRParams,
    Zone,
    find_zones,
    nearest_levels,
    reward_risk,
    score_support_resistance,
    suggest_stop_loss,
)
from analysis.volume import VolumeParams, score_volume
from analysis.weights import DEFAULT_MODULE_WEIGHTS, default_weight_values
from config.settings import TIMEFRAME_MS, Settings

# Tag setup yang sah untuk tiap regime ("strategi berbeda per regime").
SETUP_TAGS: dict[str, frozenset[str]] = {
    TRENDING_UP: frozenset({
        "fib_382", "fib_50", "fib_618", "fib_786", "pullback_ema20", "pullback_ema50",
        "bb_squeeze_breakout", "bb_breakout", "bos_up", "pantulan_support", "breakout_resistance",
        "rsi_pullback", "rsi_bull_div",
    }),
    RANGING: frozenset({
        "pantulan_support", "dekat_support", "bb_lower_touch", "rsi_oversold", "rsi_bull_div", "fib_50", "fib_618",
    }),
}

REJECTION_TEXT = {
    "data_kurang": "data candle belum cukup",
    "tren_1h_bearish": "tren 1h bearish",
    "regime_turun": "regime coin Trending Down",
    "circuit_breaker": "circuit breaker BTC aktif",
    "tf_tidak_searah": "kurang dari 3 timeframe searah",
    "tanpa_setup": "tidak ada setup sesuai regime",
    "rr_kurang": "ruang ke resistance kurang dari 1.5x jarak stop",
    "skor_rendah": "skor di bawah minimum",
}


@dataclass(frozen=True)
class EngineParams:
    timeframes: tuple[str, str, str, str] = ("5m", "15m", "30m", "1h")  # trigger, setup, structure, trend
    min_score: float = 65.0
    min_aligned: int = 3
    align_threshold: float = 0.1     # skor timeframe > 0.1 dianggap bullish (searah)
    min_reward_risk: float = 1.5
    min_rows: int = 60
    stop_role: str = "setup"         # ATR timeframe setup (15m) dipakai untuk stop loss
    stop_min_atr: float = 1.5
    stop_max_atr: float = 2.5
    tp1_reward_risk: float = 1.5     # target 1 jika tidak ada resistance
    late_rsi: float = 70.0           # RSI setup >= 70 = entry terlambat
    weak_volume_ratio: float = 1.0   # volume trigger < rata rata = volume lemah
    near_resistance_rr: float = 2.0  # ruang < 2R ditandai dekat resistance
    btc_dump_change: float = -0.015  # BTC -1.5% dalam 1 jam ditandai btc_dump
    indicators: IndicatorParams = DEFAULT_PARAMS
    ma: MAParams = field(default_factory=MAParams)
    rsi: RSIParams = field(default_factory=RSIParams)
    macd: MACDParams = field(default_factory=MACDParams)
    bb: BollingerParams = field(default_factory=BollingerParams)
    volume: VolumeParams = field(default_factory=VolumeParams)
    structure: StructureParams = field(default_factory=StructureParams)
    sr: SRParams = field(default_factory=SRParams)
    fib: FibParams = field(default_factory=FibParams)
    regime: RegimeParams = field(default_factory=RegimeParams)

    @property
    def roles(self) -> dict[str, str]:
        """{timeframe: peran}, timeframe terkecil = trigger, terbesar = trend."""
        return dict(zip(self.timeframes, ("trigger", "setup", "structure", "trend"), strict=True))

    def timeframe_of(self, role: str) -> str:
        return {r: tf for tf, r in self.roles.items()}[role]

    @classmethod
    def from_settings(cls, settings: Settings, **overrides: object) -> EngineParams:
        if len(settings.timeframes) != 4:
            raise ValueError("confluence engine butuh tepat 4 timeframe (trigger, setup, structure, trend)")
        values = {
            "timeframes": tuple(settings.timeframes),
            "min_score": settings.min_signal_score,
            "min_aligned": settings.min_aligned_timeframes,
            "min_reward_risk": settings.min_reward_risk,
        }
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class TimeframeScore:
    timeframe: str
    role: str
    score: float
    aligned: bool
    signals: tuple[SignalScore, ...]
    weights: Mapping[str, float]

    def signal(self, name: str) -> SignalScore | None:
        return next((s for s in self.signals if s.name == name), None)


@dataclass(frozen=True)
class TradePlan:
    entry: float
    stop: float
    tp1: float
    targets: tuple[float, ...]
    atr: float
    reward_risk: float
    support: Zone | None = None
    resistance: Zone | None = None

    @property
    def risk_per_unit(self) -> float:
        return self.entry - self.stop

    @property
    def stop_pct(self) -> float:
        return self.risk_per_unit / self.entry if self.entry else float("nan")


@dataclass(frozen=True)
class SignalResult:
    symbol: str
    timestamp: pd.Timestamp | None
    score: float
    raw_score: float
    aligned: int
    timeframes: Mapping[str, TimeframeScore]
    regime: Regime
    market: MarketContext
    plan: TradePlan | None
    reasons: tuple[str, ...]
    pattern: str
    tags: tuple[str, ...]
    conditions: tuple[str, ...]
    penalty: float
    rejections: tuple[str, ...]
    features: Mapping[str, float]

    @property
    def is_entry(self) -> bool:
        return not self.rejections

    @property
    def summary(self) -> str:
        return " + ".join(self.reasons) if self.reasons else "-"

    @property
    def size_multiplier(self) -> float:
        return self.regime.size_multiplier

    def rejection_text(self) -> str:
        return ", ".join(REJECTION_TEXT.get(code, code) for code in self.rejections)


def prepare_frames(raw_frames: Mapping[str, pd.DataFrame], params: IndicatorParams = DEFAULT_PARAMS) -> dict[str, pd.DataFrame]:
    """Hitung indikator untuk setiap timeframe (frame OHLCV candle tertutup)."""
    return {tf: compute_indicators(frame, params) for tf, frame in raw_frames.items()}


def close_time(frame: pd.DataFrame, timeframe: str) -> pd.Timestamp | None:
    if frame.empty:
        return None
    return frame.index[-1] + pd.Timedelta(milliseconds=TIMEFRAME_MS[timeframe])


def _value(frame: pd.DataFrame, column: str) -> float:
    return last(frame, column)


def _label_with_timeframe(label: str, timeframe: str) -> str:
    """Tambahkan timeframe di akhir label, kecuali label sudah menyebut timeframe sendiri."""
    words = label.split()
    return label if words and words[-1] in TIMEFRAME_MS else f"{label} {timeframe}"


def _fingerprint(frame: pd.DataFrame) -> tuple[object, ...]:
    """Identitas isi frame candle tertutup, dipakai sebagai kunci cache analisis."""
    if frame.empty:
        return (0,)
    stamps = getattr(frame.index, "asi8", None)  # int64 langsung, tanpa membuat objek Timestamp
    first, newest = (int(stamps[0]), int(stamps[-1])) if stamps is not None else (frame.index[0], frame.index[-1])
    return (len(frame), first, newest, last(frame, "close"), last(frame, "volume"))


@dataclass(frozen=True)
class TrendContext:
    """Hasil analisis timeframe tren (1h) yang menjadi konteks timeframe lain."""

    regime: Regime
    ma: SignalScore
    direction: str
    zones: tuple[Zone, ...]
    signals: tuple[SignalScore, ...]

    @property
    def ranging(self) -> bool:
        return self.regime.trend == RANGING


@dataclass(frozen=True)
class SetupContext:
    signals: tuple[SignalScore, ...]
    zones: tuple[Zone, ...]  # zona gabungan 1h + 15m
    fib_leg: FibLeg | None


class ConfluenceEngine:
    """Menilai satu coin pada 4 timeframe dan memutuskan apakah layak entry.

    Analisis timeframe 1h, 30m, dan 15m disimpan di cache per simbol dan baru
    dihitung ulang saat candle timeframe itu berganti. Hasilnya identik dengan
    tanpa cache (diuji), tetapi scan tiap 5 menit dan backtest jauh lebih cepat.
    """

    def __init__(
        self, weights: Mapping[str, float] | None = None, params: EngineParams | None = None, cache: bool = True
    ) -> None:
        self.weights = default_weight_values() | dict(weights or {})
        self.params = params or EngineParams()
        self.use_cache = cache
        self._cache: dict[tuple[str, str], tuple[tuple[object, ...], object]] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def _cached(self, symbol: str, role: str, key: tuple[object, ...], compute):  # noqa: ANN001, ANN202
        if not self.use_cache:
            return compute()
        entry = self._cache.get((symbol, role))
        if entry is not None and entry[0] == key:
            return entry[1]
        value = compute()
        self._cache[(symbol, role)] = (key, value)
        return value

    # ------------------------------------------------------------------
    # Analisis per peran timeframe
    # ------------------------------------------------------------------
    def _analyze_trend(self, frame: pd.DataFrame) -> TrendContext:
        p = self.params
        tf = p.timeframe_of("trend")
        regime = classify_regime(frame, p.regime)
        ma = score_moving_average(frame, p.ma)
        direction = str(ma.details.get("trend", "sideways"))
        zones = tuple(find_zones(frame, tf, p.sr))
        builders = {
            "ma": lambda: ma,
            "sr": lambda: score_support_resistance(frame, zones, p.sr),
            "structure": lambda: score_structure(frame, p.structure),
            "macd": lambda: score_macd(frame, direction, p.macd),
            "rsi": lambda: score_rsi(frame, direction, p.rsi),
        }
        signals = tuple(builders[name]() for name in DEFAULT_MODULE_WEIGHTS["trend"])
        return TrendContext(regime=regime, ma=ma, direction=direction, zones=zones, signals=signals)

    def _analyze_structure(self, frame: pd.DataFrame, trend: TrendContext) -> tuple[SignalScore, ...]:
        p = self.params
        builders = {
            "structure": lambda: score_structure(frame, p.structure),
            "ma": lambda: score_moving_average(frame, p.ma),
            "macd": lambda: score_macd(frame, trend.direction, p.macd),
            "rsi": lambda: score_rsi(frame, trend.direction, p.rsi),
        }
        return tuple(builders[name]() for name in DEFAULT_MODULE_WEIGHTS["structure"])

    def _analyze_setup(self, frame: pd.DataFrame, trend_frame: pd.DataFrame, trend: TrendContext) -> SetupContext:
        p = self.params
        tf_setup, tf_trend = p.timeframe_of("setup"), p.timeframe_of("trend")
        zones = trend.zones + tuple(find_zones(frame, tf_setup, p.sr))
        # Fibonacci: leg dari 15m atau 1h, dinilai dengan price action 15m; ambil yang terbaik.
        ema_levels = [_value(frame, "ema20"), _value(frame, "ema50")]
        candidates: list[tuple[SignalScore, FibLeg]] = []
        for leg in (find_fib_leg(frame, tf_setup, p.fib), find_fib_leg(trend_frame, tf_trend, p.fib)):
            if leg is not None:
                candidates.append((score_fibonacci(frame, leg, zones, ema_levels, p.fib), leg))
        fib_leg: FibLeg | None = None
        if candidates:
            fib_signal, fib_leg = max(candidates, key=lambda item: item[0].score)
        else:
            fib_signal = SignalScore.neutral("fib", "tidak ada swing signifikan untuk Fibonacci")
        builders = {
            "fib": lambda: fib_signal,
            "sr": lambda: score_support_resistance(frame, zones, p.sr),
            "bb": lambda: score_bollinger(frame, trend.ranging, p.bb),
            "rsi": lambda: score_rsi(frame, trend.direction, p.rsi),
            "ma": lambda: score_moving_average(frame, p.ma),
        }
        signals = tuple(builders[name]() for name in DEFAULT_MODULE_WEIGHTS["setup"])
        return SetupContext(signals=signals, zones=zones, fib_leg=fib_leg)

    def _analyze_trigger(self, frame: pd.DataFrame, trend: TrendContext) -> tuple[SignalScore, ...]:
        p = self.params
        builders = {
            "macd": lambda: score_macd(frame, trend.direction, p.macd),
            "volume": lambda: score_volume(frame, p.volume),
            "candle": lambda: score_trigger_candle(frame),
            "rsi": lambda: score_rsi(frame, trend.direction, p.rsi),
            "bb": lambda: score_bollinger(frame, trend.ranging, p.bb),
        }
        return tuple(builders[name]() for name in DEFAULT_MODULE_WEIGHTS["trigger"])

    def _timeframe_score(self, tf: str, role: str, signals: Sequence[SignalScore]) -> TimeframeScore:
        module_weights = {s.name: max(self.weights.get(f"w.{role}.{s.name}", 0.0), 0.0) for s in signals}
        total = sum(module_weights.values())
        value = sum(module_weights[s.name] * s.score for s in signals) / total if total > 0 else 0.0
        return TimeframeScore(
            timeframe=tf,
            role=role,
            score=round(value, 6),
            aligned=value > self.params.align_threshold,
            signals=tuple(signals),
            weights=module_weights,
        )

    def _timeframe_scores(self, signals: Mapping[str, Sequence[SignalScore]]) -> dict[str, TimeframeScore]:
        return {tf: self._timeframe_score(tf, role, signals[tf]) for tf, role in self.params.roles.items()}

    def _timeframe_weights(self) -> tuple[dict[str, float], float]:
        weights = {tf: max(self.weights.get(f"tf.{role}", 0.0), 0.0) for tf, role in self.params.roles.items()}
        return weights, sum(weights.values()) or 1.0

    @staticmethod
    def _trend_rejections(trend: TrendContext, market: MarketContext) -> tuple[str, ...]:
        """Alasan tolak yang hanya bergantung pada tren 1h dan kondisi pasar (sama dengan evaluasi penuh)."""
        codes: list[str] = []
        if trend.direction == TREND_DOWN or trend.ma.score <= -0.3:
            codes.append("tren_1h_bearish")
        if not trend.regime.allows_entry:
            codes.append("regime_turun")
        if market.circuit_breaker_active:
            codes.append("circuit_breaker")
        return tuple(codes)

    def _certain_rejections(
        self, trend: TrendContext, structure: Sequence[SignalScore], setup: SetupContext, setup_f: pd.DataFrame, market: MarketContext
    ) -> tuple[str, ...]:
        """Alasan tolak yang SUDAH PASTI dari timeframe 1h/30m/15m dan kondisi pasar.

        Dipakai evaluate(fast_reject=True) untuk melewati analisis 5m yang hasilnya
        pasti ditolak. Setiap kode di sini pasti juga muncul pada evaluasi penuh:
        * jumlah timeframe searah paling banyak (searah di 1h/30m/15m) + 1;
        * skor paling tinggi dihitung dengan skor 5m = +1 dan penalti yang belum
          pasti dianggap serendah mungkin.
        """
        p = self.params
        tf_trend, tf_struct, tf_setup, tf_trig = (p.timeframe_of(r) for r in ("trend", "structure", "setup", "trigger"))
        codes = list(self._trend_rejections(trend, market))
        higher = {
            tf_trend: self._timeframe_score(tf_trend, "trend", trend.signals),
            tf_struct: self._timeframe_score(tf_struct, "structure", structure),
            tf_setup: self._timeframe_score(tf_setup, "setup", setup.signals),
        }
        tags = {tag for ts in higher.values() for signal in ts.signals for tag in signal.tags}
        if not tags & SETUP_TAGS.get(trend.regime.trend, frozenset()):
            codes.append("tanpa_setup")
        if sum(ts.aligned for ts in higher.values()) + 1 < p.min_aligned:
            codes.append("tf_tidak_searah")

        tf_weights, total_tf = self._timeframe_weights()
        raw_max = (sum(tf_weights[tf] * ts.score for tf, ts in higher.items()) + tf_weights[tf_trig] * 1.0) / total_tf
        known = []
        if trend.direction != TREND_UP:
            known.append("melawan_tren_1h")
        rsi_setup = _value(setup_f, "rsi")
        if is_valid(rsi_setup) and rsi_setup >= p.late_rsi:
            known.append("entry_terlambat")
        if market.btc_change_1h <= p.btc_dump_change:
            known.append("btc_dump")
        unknown = [c for c in ("dekat_resistance", "volume_lemah", "entry_terlambat") if c not in known]
        penalty_min = sum(self.weights.get(f"penalty.{c}", 0.0) for c in known)
        penalty_min += sum(min(self.weights.get(f"penalty.{c}", 0.0), 0.0) for c in unknown)
        if market.btc_regime is not None and not market.btc_regime.allows_entry:
            penalty_min += self.weights.get("penalty.btc_turun", 0.0)
        # 1e-6: cadangan pembulatan float agar batas atas tidak pernah di bawah skor sebenarnya.
        score_max = round(min(max(50.0 * (1.0 + raw_max - penalty_min) + 1e-6, 0.0), 100.0), 2)
        if score_max < p.min_score:
            codes.append("skor_rendah")
        return tuple(codes)

    # ------------------------------------------------------------------
    # Evaluasi utama
    # ------------------------------------------------------------------
    def evaluate(
        self, symbol: str, frames: Mapping[str, pd.DataFrame], market: MarketContext | None = None, *, fast_reject: bool = False
    ) -> SignalResult:
        """Nilai satu coin dari frame indikator (lihat prepare_frames) candle tertutup.

        fast_reject=True (dipakai backtest): jika timeframe 1h/30m/15m dan kondisi
        pasar sudah pasti menolak entry, analisis 5m dilewati dan hasilnya hanya
        berisi alasan tolak tersebut (skor dan fitur tidak dihitung). Keputusan
        entry selalu sama dengan evaluasi penuh.
        """
        p = self.params
        market = market or MarketContext()
        tf_trend, tf_struct, tf_setup, tf_trig = (p.timeframe_of(r) for r in ("trend", "structure", "setup", "trigger"))
        insufficient = [tf for tf in p.timeframes if tf not in frames or len(frames[tf]) < p.min_rows]
        timestamp = close_time(frames[tf_trig], tf_trig) if tf_trig in frames else None
        if insufficient:
            regime = classify_regime(frames[tf_trend], p.regime) if tf_trend in frames else classify_regime(pd.DataFrame())
            return SignalResult(
                symbol=symbol, timestamp=timestamp, score=0.0, raw_score=0.0, aligned=0, timeframes={}, regime=regime,
                market=market, plan=None, reasons=(), pattern="", tags=(), conditions=(), penalty=0.0,
                rejections=("data_kurang",), features={"missing_timeframes": float(len(insufficient))},
            )

        trend_f, struct_f, setup_f, trig_f = frames[tf_trend], frames[tf_struct], frames[tf_setup], frames[tf_trig]
        trend_key = _fingerprint(trend_f)
        trend: TrendContext = self._cached(symbol, "trend", trend_key, lambda: self._analyze_trend(trend_f))
        if fast_reject and (early := self._trend_rejections(trend, market)):
            return self._fast_rejected(symbol, timestamp, trend, market, early)
        structure = self._cached(
            symbol, "structure", (_fingerprint(struct_f), trend.direction), lambda: self._analyze_structure(struct_f, trend)
        )
        setup: SetupContext = self._cached(
            symbol, "setup", (_fingerprint(setup_f), trend_key), lambda: self._analyze_setup(setup_f, trend_f, trend)
        )
        if fast_reject and (certain := self._certain_rejections(trend, structure, setup, setup_f, market)):
            return self._fast_rejected(symbol, timestamp, trend, market, certain)
        trigger = self._analyze_trigger(trig_f, trend)
        regime, ma_trend, trend_dir = trend.regime, trend.ma, trend.direction
        signals = {tf_trend: trend.signals, tf_struct: structure, tf_setup: setup.signals, tf_trig: trigger}
        tf_scores = self._timeframe_scores(signals)

        tf_weights, total_tf = self._timeframe_weights()
        raw = sum(tf_weights[tf] * tf_scores[tf].score for tf in tf_scores) / total_tf
        aligned = sum(1 for ts in tf_scores.values() if ts.aligned)

        plan = self._trade_plan(frames, setup.zones, setup.fib_leg)
        conditions = self._conditions(trend_dir, plan, setup_f, trig_f, market)
        penalty = sum(self.weights.get(f"penalty.{c}", 0.0) for c in conditions)
        if market.btc_regime is not None and not market.btc_regime.allows_entry:
            penalty += self.weights.get("penalty.btc_turun", 0.0)
        score = round(min(max(50.0 * (1.0 + raw - penalty), 0.0), 100.0), 2)

        tags_by_tf = {tf: {tag for s in ts.signals for tag in s.tags} for tf, ts in tf_scores.items()}
        setup_scope = set().union(*(tags_by_tf[p.timeframe_of(r)] for r in ("trend", "structure", "setup")))
        setup_found = setup_scope & SETUP_TAGS.get(regime.trend, frozenset())

        rejections: list[str] = []
        if trend_dir == TREND_DOWN or ma_trend.score <= -0.3:
            rejections.append("tren_1h_bearish")
        if not regime.allows_entry:
            rejections.append("regime_turun")
        if market.circuit_breaker_active:
            rejections.append("circuit_breaker")
        if aligned < p.min_aligned:
            rejections.append("tf_tidak_searah")
        if not setup_found:
            rejections.append("tanpa_setup")
        if plan.reward_risk < p.min_reward_risk:
            rejections.append("rr_kurang")
        if score < p.min_score:
            rejections.append("skor_rendah")

        reasons, pattern = self._reasons(tf_scores, tf_weights, total_tf)
        all_tags = tuple(sorted(f"{tag}@{tf}" for tf, tags in tags_by_tf.items() for tag in tags))
        features = self._features(frames, tf_scores, regime, market, plan, raw, penalty, aligned, timestamp)
        return SignalResult(
            symbol=symbol,
            timestamp=timestamp,
            score=score,
            raw_score=round(raw, 6),
            aligned=aligned,
            timeframes=tf_scores,
            regime=regime,
            market=market,
            plan=plan,
            reasons=reasons,
            pattern=pattern,
            tags=all_tags,
            conditions=tuple(conditions),
            penalty=round(penalty, 6),
            rejections=tuple(rejections),
            features=features,
        )

    # ------------------------------------------------------------------
    # Pendukung
    # ------------------------------------------------------------------
    @staticmethod
    def _fast_rejected(
        symbol: str, timestamp: pd.Timestamp | None, trend: TrendContext, market: MarketContext, codes: tuple[str, ...]
    ) -> SignalResult:
        """Hasil ringkas penolakan cepat: hanya `rejections` yang bermakna."""
        return SignalResult(
            symbol=symbol, timestamp=timestamp, score=0.0, raw_score=0.0, aligned=0, timeframes={},
            regime=trend.regime, market=market, plan=None, reasons=(), pattern="", tags=(), conditions=(),
            penalty=0.0, rejections=codes, features={"fast_reject": 1.0},
        )

    def _trade_plan(self, frames: Mapping[str, pd.DataFrame], zones: Sequence[Zone], leg: FibLeg | None) -> TradePlan:
        p = self.params
        entry = _value(frames[p.timeframe_of("trigger")], "close")
        atr = _value(frames[p.timeframe_of(p.stop_role)], "atr")
        if not is_valid(atr) or atr <= 0:
            atr = abs(entry) * 0.01
        support, resistance = nearest_levels(entry, zones, p.sr.min_strength)
        stop = suggest_stop_loss(entry, atr, support, p.stop_min_atr, p.stop_max_atr)
        risk = entry - stop
        rr = reward_risk(entry, stop, resistance)
        if resistance is not None and resistance.low - 0.1 * atr > entry:
            tp1 = resistance.low - 0.1 * atr
        else:
            tp1 = entry + p.tp1_reward_risk * risk
        targets = tuple(sorted(t for t in (leg.extensions() if leg else ()) if t > entry))
        return TradePlan(entry=entry, stop=stop, tp1=tp1, targets=targets, atr=atr, reward_risk=rr, support=support, resistance=resistance)

    def _conditions(
        self, trend_dir: str, plan: TradePlan, setup_f: pd.DataFrame, trig_f: pd.DataFrame, market: MarketContext
    ) -> list[str]:
        """Kondisi berisiko saat entry (dipakai untuk penalti dan mistake analyzer)."""
        p = self.params
        conditions: list[str] = []
        if trend_dir != TREND_UP:
            conditions.append("melawan_tren_1h")
        if plan.reward_risk < p.near_resistance_rr:
            conditions.append("dekat_resistance")
        volume = _value(trig_f, "volume_ratio")
        if is_valid(volume) and volume < p.weak_volume_ratio:
            conditions.append("volume_lemah")
        rsi_setup, rsi_trigger = _value(setup_f, "rsi"), _value(trig_f, "rsi")
        if (is_valid(rsi_setup) and rsi_setup >= p.late_rsi) or (is_valid(rsi_trigger) and rsi_trigger >= p.late_rsi + 5):
            conditions.append("entry_terlambat")
        if market.btc_change_1h <= p.btc_dump_change:
            conditions.append("btc_dump")
        return conditions

    @staticmethod
    def _reasons(
        tf_scores: Mapping[str, TimeframeScore], tf_weights: Mapping[str, float], total_tf: float
    ) -> tuple[tuple[str, ...], str]:
        """Empat alasan dengan kontribusi positif terbesar, plus kunci pola untuk memori pola."""
        contributions: list[tuple[float, str, str]] = []
        for tf, ts in tf_scores.items():
            total_modules = sum(ts.weights.values()) or 1.0
            share = tf_weights[tf] / total_tf
            for signal in ts.signals:
                if signal.score > 0 and signal.label:
                    contribution = share * ts.weights.get(signal.name, 0.0) / total_modules * signal.score
                    if contribution > 0:
                        primary = signal.tags[0] if signal.tags else signal.name
                        contributions.append((contribution, _label_with_timeframe(signal.label, tf), primary))
        contributions.sort(key=lambda item: item[0], reverse=True)
        top = contributions[:4]
        reasons = tuple(text for _, text, _ in top)
        pattern = "+".join(sorted({primary for _, _, primary in top}))
        return reasons, pattern

    def _features(
        self,
        frames: Mapping[str, pd.DataFrame],
        tf_scores: Mapping[str, TimeframeScore],
        regime: Regime,
        market: MarketContext,
        plan: TradePlan,
        raw: float,
        penalty: float,
        aligned: int,
        timestamp: pd.Timestamp | None,
    ) -> dict[str, float]:
        """Snapshot fitur saat sinyal (untuk jurnal trade dan model ML di Fase 5).

        Semua fitur sudah bebas skala harga (skor, RSI, ADX, rasio, satuan ATR),
        sehingga bisa dibandingkan antar coin dengan harga berbeda.
        """
        features: dict[str, float] = {}
        for tf, ts in tf_scores.items():
            features[f"{tf}.score"] = ts.score
            for signal in ts.signals:
                features[f"{tf}.{signal.name}"] = signal.score
            frame = frames[tf]
            for column in ("rsi", "adx", "atr_pct", "volume_ratio", "bb_width_rank", "bb_pctb"):
                features[f"{tf}.{column}"] = _value(frame, column)
        setup_tf = self.params.timeframe_of("setup")
        fib = tf_scores[setup_tf].signal("fib")
        atr = plan.atr
        features.update({
            "raw_score": raw,
            "penalty": penalty,
            "aligned": float(aligned),
            "regime": float(regime.code()),
            "high_volatility": float(regime.high_volatility),
            "atr_pct_rank": regime.atr_pct_rank,
            "dist_support_atr": (plan.entry - plan.support.high) / atr if plan.support else float("nan"),
            "dist_resistance_atr": (plan.resistance.low - plan.entry) / atr if plan.resistance else float("nan"),
            "reward_risk": min(plan.reward_risk, 10.0),
            "stop_pct": plan.stop_pct,
            "fib_retracement": float(fib.details.get("retracement", float("nan"))) if fib else float("nan"),
            "btc_regime": float(market.btc_regime.code()) if market.btc_regime else 0.0,
            "btc_change_1h": market.btc_change_1h,
            "btc_drop_1h": market.btc_drop_1h,
            "hour": float(timestamp.hour) if timestamp is not None else float("nan"),
            "weekday": float(timestamp.weekday()) if timestamp is not None else float("nan"),
        })
        return {key: (float(value) if is_valid(value) else float("nan")) for key, value in features.items()}


def rank_signals(results: Sequence[SignalResult]) -> list[SignalResult]:
    """Urutkan hasil: entry valid dulu, lalu skor tertinggi."""
    return sorted(results, key=lambda r: (r.is_entry, r.score), reverse=True)


def format_signal(result: SignalResult) -> str:
    """Satu baris ringkas untuk console, contoh: 'SOL/USDT skor 84 | Fib 61.8% 15m + ...'."""
    status = "ENTRY" if result.is_entry else "tolak: " + result.rejection_text()
    return f"{result.symbol:<12} skor {result.score:5.1f} | {result.summary} | {result.regime.label} | {status}"


def describe_signal(result: SignalResult) -> str:
    """Laporan lengkap satu sinyal (dipakai perintah `analyze` dan log bot)."""
    status = "ENTRY" if result.is_entry else "DITOLAK: " + result.rejection_text()
    when = f"{result.timestamp:%Y-%m-%d %H:%M} UTC" if result.timestamp is not None else "-"
    breaker = "AKTIF" if result.market.circuit_breaker_active else "tidak aktif"
    lines = [
        f"{result.symbol} | skor {result.score:.1f}/100 | {status}",
        f"Candle {when} | Regime: {result.regime.label} | Regime BTC: {result.market.label} | Circuit breaker: {breaker}",
        f"Regime: {result.regime.reason}",
        f"Alasan utama: {result.summary}",
    ]
    if result.timeframes:
        lines.append(f"Timeframe searah: {result.aligned} dari {len(result.timeframes)}")
    for tf, ts in sorted(result.timeframes.items(), key=lambda item: TIMEFRAME_MS[item[0]], reverse=True):
        mark = "searah" if ts.aligned else "tidak searah"
        lines.append(f"  {tf:>4} ({ts.role}) {ts.score:+.2f} {mark}")
        for signal in ts.signals:
            weight = ts.weights.get(signal.name, 0.0)
            lines.append(f"      {signal.name:<9} {signal.score:+.2f} (bobot {weight:.2f}) {signal.reason}")
    if result.plan is not None:
        plan = result.plan
        rr = "tanpa resistance" if math.isinf(plan.reward_risk) else f"{plan.reward_risk:.2f}"
        targets = ", ".join(f"{t:.6g}" for t in plan.targets) or "-"
        lines.append(
            f"Rencana: entry {plan.entry:.6g} | SL {plan.stop:.6g} ({plan.stop_pct:.2%}) | TP1 {plan.tp1:.6g} | "
            f"ruang/risiko {rr} | target Fibonacci {targets}"
        )
    if result.conditions:
        lines.append("Kondisi berisiko: " + ", ".join(result.conditions) + f" (penalti {result.penalty:.2f})")
    if result.regime.high_volatility:
        lines.append(f"Volatilitas tinggi: ukuran posisi x{result.size_multiplier:g}")
    return "\n".join(lines)
