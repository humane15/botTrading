"""Risk manager: pemeriksaan terakhir sebelum posisi baru dibuka.

Entry baru DITOLAK jika salah satu kondisi ini terjadi:
* kill switch aktif (ada file STOP di root proyek);
* rugi hari ini >= 5% atau rugi minggu ini >= 10% dari nilai akun di awal
  hari/minggu (UTC, termasuk posisi yang masih terbuka). Entry dibuka lagi
  otomatis saat periode berikutnya dimulai;
* slot posisi penuh (maksimal 5, dikurangi otomatis jika modal kecil);
* coin itu sudah punya posisi, atau diblokir karena ada saldo coin tersebut
  di luar kendali bot;
* sudah ada 2 posisi pada coin yang korelasinya tinggi dengan BTC, dan coin
  baru ini juga berkorelasi tinggi.

Posisi yang sudah terbuka TIDAK ditutup oleh risk manager: stop loss, take
profit, dan trailing stop tetap berjalan seperti biasa.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import Settings
from risk.position_sizing import effective_max_positions, min_trade_notional
from risk.positions import Position, PositionStore

REASON_TEXT = {
    "kill_switch": "kill switch aktif (file STOP ada di root proyek)",
    "batas_rugi_harian": "batas rugi harian tercapai",
    "batas_rugi_mingguan": "batas rugi mingguan tercapai",
    "posisi_penuh": "slot posisi sudah penuh",
    "modal_terlalu_kecil": "modal terlalu kecil: tiap posisi butuh minimal 2 x min notional Binance x buffer",
    "sudah_ada_posisi": "coin ini sudah punya posisi terbuka",
    "simbol_diblokir": "ada saldo coin ini di luar kendali bot",
    "korelasi_btc_penuh": "sudah ada posisi maksimal pada coin berkorelasi tinggi dengan BTC",
}


def btc_correlation(closes: pd.Series, btc_closes: pd.Series, window: int = 168, min_periods: int = 48) -> float:
    """Korelasi return log 1h coin terhadap BTC (default 7 hari terakhir). NaN jika data kurang."""
    joined = pd.concat([closes.rename("coin"), btc_closes.rename("btc")], axis=1, join="inner").dropna()
    returns = np.log(joined.tail(window + 1)).diff().dropna()
    if len(returns) < min_periods or returns["coin"].std() == 0 or returns["btc"].std() == 0:
        return float("nan")
    return float(returns["coin"].corr(returns["btc"]))


@dataclass(frozen=True)
class LossStatus:
    equity: float
    day_start: float
    week_start: float

    @property
    def daily_change(self) -> float:
        return self.equity / self.day_start - 1 if self.day_start > 0 else 0.0

    @property
    def weekly_change(self) -> float:
        return self.equity / self.week_start - 1 if self.week_start > 0 else 0.0


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reasons: tuple[str, ...]
    max_positions: int
    open_positions: int

    def text(self) -> str:
        return ", ".join(REASON_TEXT.get(code, code) for code in self.reasons) or "boleh entry"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def period_keys(now: datetime) -> tuple[str, str]:
    """Kunci periode harian dan mingguan (ISO, senin sebagai awal minggu), dalam UTC."""
    now = now.astimezone(timezone.utc)
    iso = now.isocalendar()
    return f"day:{now:%Y-%m-%d}", f"week:{iso.year}-W{iso.week:02d}"


class RiskManager:
    def __init__(
        self,
        settings: Settings,
        store: PositionStore,
        mode: str,
        *,
        kill_file: Path | None = None,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self.settings = settings
        self.store = store
        self.mode = mode
        self.kill_file = kill_file or settings.kill_switch_file
        self._clock = clock
        self.blocked_symbols: set[str] = set()

    def kill_switch_active(self) -> bool:
        return self.kill_file.exists()

    def block_symbols(self, symbols: Iterable[str]) -> None:
        self.blocked_symbols.update(symbols)

    def loss_status(self, equity: float) -> LossStatus:
        """Ekuitas sekarang dibanding awal hari dan awal minggu (dicatat saat pertama kali terlihat)."""
        day_key, week_key = period_keys(self._clock())
        day_start = self.store.ensure_period_equity(self.mode, day_key, equity)
        week_start = self.store.ensure_period_equity(self.mode, week_key, equity)
        return LossStatus(equity=equity, day_start=day_start, week_start=week_start)

    def max_positions(self, equity: float, min_notional: float = 5.0) -> int:
        min_trade = min_trade_notional(min_notional, self.settings.min_notional_buffer)
        return effective_max_positions(equity, self.settings.max_open_positions, min_trade)

    def is_btc_correlated(self, correlation: float | None) -> bool:
        return correlation is not None and not math.isnan(correlation) and correlation >= self.settings.btc_correlation_threshold

    def check_entry(
        self,
        symbol: str,
        *,
        equity: float,
        open_positions: Sequence[Position],
        btc_corr: float | None = None,
        min_notional: float = 5.0,
    ) -> RiskDecision:
        reasons: list[str] = []
        active = [p for p in open_positions if p.is_active]
        slots = self.max_positions(equity, min_notional)

        if self.kill_switch_active():
            reasons.append("kill_switch")
        status = self.loss_status(equity)
        if status.daily_change <= -self.settings.daily_loss_limit:
            reasons.append("batas_rugi_harian")
        if status.weekly_change <= -self.settings.weekly_loss_limit:
            reasons.append("batas_rugi_mingguan")
        if slots == 0:
            reasons.append("modal_terlalu_kecil")
        elif len(active) >= slots:
            reasons.append("posisi_penuh")
        if any(p.symbol == symbol for p in active):
            reasons.append("sudah_ada_posisi")
        if symbol in self.blocked_symbols:
            reasons.append("simbol_diblokir")
        if self.is_btc_correlated(btc_corr):
            correlated = sum(1 for p in active if self.is_btc_correlated(p.btc_corr))
            if correlated >= self.settings.max_btc_correlated_positions:
                reasons.append("korelasi_btc_penuh")
        return RiskDecision(allowed=not reasons, reasons=tuple(reasons), max_positions=slots, open_positions=len(active))
