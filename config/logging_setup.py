"""Pengaturan logging: console, logs/bot.log, dan logs/learning.log.

Semua output log melewati RedactingFormatter yang menyensor nilai rahasia
(API key, secret, token Telegram) seandainya ikut tercetak, misalnya di
dalam pesan error dari library.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LEARNING_LOGGER = "learning"

_MANAGED_ATTR = "_bot_managed_handler"


class RedactingFormatter(logging.Formatter):
    """Formatter yang mengganti setiap nilai rahasia dengan ***."""

    def __init__(self, secrets: Iterable[str] = (), fmt: str = LOG_FORMAT, datefmt: str = DATE_FORMAT) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        # Rahasia yang terlalu pendek diabaikan agar tidak menyensor teks biasa.
        self._secrets = sorted({s for s in secrets if s and len(s) >= 6}, key=len, reverse=True)

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        for secret in self._secrets:
            text = text.replace(secret, "***")
        return text


def _add_handler(logger: logging.Logger, handler: logging.Handler, formatter: logging.Formatter) -> None:
    handler.setFormatter(formatter)
    setattr(handler, _MANAGED_ATTR, True)
    logger.addHandler(handler)


def setup_logging(level: str = "INFO", logs_dir: Path | None = None, secrets: Iterable[str] = ()) -> None:
    """Pasang handler logging. Aman dipanggil berulang kali (handler lama diganti)."""
    formatter = RedactingFormatter(secrets)
    root = logging.getLogger()
    learning = logging.getLogger(LEARNING_LOGGER)

    for logger in (root, learning):
        for handler in list(logger.handlers):
            if getattr(handler, _MANAGED_ATTR, False):
                logger.removeHandler(handler)
                handler.close()

    root.setLevel(level)
    _add_handler(root, logging.StreamHandler(sys.stdout), formatter)

    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        _add_handler(
            root,
            RotatingFileHandler(logs_dir / "bot.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"),
            formatter,
        )
        # Log modul learning juga ditulis terpisah agar perubahan bobot mudah diaudit.
        _add_handler(
            learning,
            RotatingFileHandler(logs_dir / "learning.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"),
            formatter,
        )

    for noisy in ("ccxt", "asyncio", "aiohttp", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
