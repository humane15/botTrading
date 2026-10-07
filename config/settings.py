"""Konfigurasi bot.

Nilai dibaca dengan urutan prioritas: environment variable, lalu file .env,
lalu nilai default di kelas Settings. API key disimpan sebagai SecretStr
sehingga tidak ikut tercetak di log, repr, atau traceback.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from dotenv import dotenv_values
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

# Durasi timeframe dalam milidetik. Hanya timeframe yang sejajar dengan epoch
# UTC (sampai 1d) yang didukung, supaya penentuan candle tertutup akurat.
TIMEFRAME_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


class LiveTradingNotAllowed(RuntimeError):
    """Dilempar ketika syarat untuk mengaktifkan mode live belum terpenuhi."""


def _split_csv(value: Any) -> Any:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return value


class Settings(BaseModel):
    """Seluruh parameter bot. Nama variabel .env = nama field dalam huruf besar."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    # Mode trading. Mode live tetap butuh flag --confirm-live saat dijalankan.
    trading_mode: Literal["paper", "live"] = "paper"

    # Binance. Izin API key cukup Read + Spot Trading, tanpa Withdraw.
    binance_api_key: SecretStr | None = None
    binance_api_secret: SecretStr | None = None
    binance_testnet: bool = False
    request_timeout_ms: int = Field(default=15_000, ge=1_000, le=120_000)

    # Rate limit dan retry.
    max_concurrent_requests: int = Field(default=5, ge=1, le=20)
    max_retries: int = Field(default=5, ge=0, le=10)
    retry_base_delay: float = Field(default=1.0, gt=0, le=30)
    retry_max_delay: float = Field(default=60.0, gt=0, le=900)

    # Universe coin.
    quote_asset: str = "USDT"
    universe_size: int = Field(default=75, ge=1, le=300)
    min_quote_volume_usd: float = Field(default=5_000_000, ge=0)
    universe_refresh_hours: float = Field(default=24, gt=0, le=168)
    excluded_symbols: tuple[str, ...] = ()
    excluded_tags: tuple[str, ...] = ("Monitoring",)

    # Data candle.
    timeframes: tuple[str, ...] = ("5m", "15m", "30m", "1h")
    ohlcv_limit: int = Field(default=300, ge=50, le=1000)
    candle_close_grace_ms: int = Field(default=2_000, ge=0, le=60_000)

    # Manajemen risiko (dipakai mulai Fase 3). Risiko per trade dibatasi
    # maksimal 2% modal sesuai aturan; nilai lebih kecil tetap diizinkan.
    risk_per_trade: float = Field(default=0.01, gt=0, le=0.02)
    max_open_positions: int = Field(default=5, ge=1, le=20)
    daily_loss_limit: float = Field(default=0.05, gt=0, le=0.5)
    weekly_loss_limit: float = Field(default=0.10, gt=0, le=0.5)
    fee_rate: float = Field(default=0.001, ge=0, le=0.01)
    slippage_rate: float = Field(default=0.0005, ge=0, le=0.05)
    paper_start_balance: float = Field(default=100.0, gt=0)

    # Notifikasi Telegram (opsional).
    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str | None = None

    # Lokasi file dan logging.
    data_dir: Path = PROJECT_ROOT / "data"
    logs_dir: Path = PROJECT_ROOT / "logs"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("timeframes", "excluded_symbols", "excluded_tags", mode="before")
    @classmethod
    def _parse_csv(cls, value: Any) -> Any:
        return _split_csv(value)

    @field_validator("binance_api_key", "binance_api_secret", "telegram_bot_token", "telegram_chat_id", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        # Spasi hasil copy paste dibuang; string kosong berarti tidak diisi.
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @field_validator("trading_mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("log_level", "quote_asset", mode="before")
    @classmethod
    def _upper(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("excluded_symbols")
    @classmethod
    def _upper_symbols(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(item.upper() for item in value)

    @field_validator("timeframes")
    @classmethod
    def _check_timeframes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("TIMEFRAMES tidak boleh kosong")
        unknown = [tf for tf in value if tf not in TIMEFRAME_MS]
        if unknown:
            raise ValueError(f"timeframe tidak dikenal: {unknown}, pilihan: {list(TIMEFRAME_MS)}")
        if len(set(value)) != len(value):
            raise ValueError(f"timeframe duplikat: {list(value)}")
        # Urutkan dari timeframe terkecil (timing entry) ke terbesar (arah tren).
        return tuple(sorted(value, key=TIMEFRAME_MS.__getitem__))

    @field_validator("data_dir", "logs_dir")
    @classmethod
    def _resolve_dir(cls, value: Path) -> Path:
        # Path relatif selalu dihitung dari root proyek, bukan dari folder kerja.
        return value if value.is_absolute() else PROJECT_ROOT / value

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.weekly_loss_limit < self.daily_loss_limit:
            raise ValueError("WEEKLY_LOSS_LIMIT tidak boleh lebih kecil dari DAILY_LOSS_LIMIT")
        if self.retry_max_delay < self.retry_base_delay:
            raise ValueError("RETRY_MAX_DELAY tidak boleh lebih kecil dari RETRY_BASE_DELAY")
        if (self.binance_api_key is None) != (self.binance_api_secret is None):
            raise ValueError("BINANCE_API_KEY dan BINANCE_API_SECRET harus diisi keduanya atau dikosongkan keduanya")
        return self

    @property
    def has_api_credentials(self) -> bool:
        return self.binance_api_key is not None and self.binance_api_secret is not None

    @property
    def universe_file(self) -> Path:
        return self.data_dir / "universe.json"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def secret_values(self) -> list[str]:
        """Nilai rahasia mentah, hanya untuk disensor dari log."""
        secrets = (self.binance_api_key, self.binance_api_secret, self.telegram_bot_token)
        return [s.get_secret_value() for s in secrets if s is not None]

    def summary(self) -> str:
        """Ringkasan konfigurasi yang aman dicetak (tanpa rahasia)."""
        return (
            f"mode={self.trading_mode} testnet={self.binance_testnet} "
            f"api_key={'terisi' if self.has_api_credentials else 'kosong'} "
            f"universe={self.universe_size} min_volume=${self.min_quote_volume_usd:,.0f} "
            f"timeframes={','.join(self.timeframes)} risiko/trade={self.risk_per_trade:.2%}"
        )


def load_settings(
    env_file: str | os.PathLike[str] | None = DEFAULT_ENV_FILE,
    *,
    environ: Mapping[str, str] | None = None,
    **overrides: Any,
) -> Settings:
    """Bangun Settings dari file .env, environment variable, dan override.

    Nilai kosong (misal ``BINANCE_API_KEY=``) dianggap tidak diisi sehingga
    nilai default yang dipakai.
    """
    fields = Settings.model_fields
    raw: dict[str, Any] = {}

    if env_file is not None and Path(env_file).is_file():
        file_values = dotenv_values(env_file)
        unknown = sorted(key for key in file_values if key.lower() not in fields)
        if unknown:
            log.warning("Kunci tidak dikenal di %s diabaikan: %s", env_file, ", ".join(unknown))
        raw.update({key.lower(): value for key, value in file_values.items()})

    source_env = os.environ if environ is None else environ
    for name in fields:
        value = source_env.get(name.upper())
        if value is not None:
            raw[name] = value

    raw.update(overrides)
    cleaned = {key: value for key, value in raw.items() if key in fields and value is not None and value != ""}
    return Settings(**cleaned)


def ensure_live_allowed(settings: Settings, confirm_live: bool) -> None:
    """Pastikan semua syarat mode live terpenuhi, kalau tidak lempar LiveTradingNotAllowed.

    Syarat: TRADING_MODE=live, flag --confirm-live diberikan, dan API key terisi.
    Pemeriksaan izin API key (tanpa withdraw, wajib IP whitelist) dilakukan
    terhadap Binance saat bot live dijalankan.
    """
    problems = []
    if settings.trading_mode != "live":
        problems.append(f"TRADING_MODE harus 'live' (saat ini '{settings.trading_mode}')")
    if not confirm_live:
        problems.append("flag --confirm-live tidak diberikan")
    if not settings.has_api_credentials:
        problems.append("BINANCE_API_KEY dan BINANCE_API_SECRET belum diisi")
    if problems:
        raise LiveTradingNotAllowed("Mode live ditolak: " + "; ".join(problems))
