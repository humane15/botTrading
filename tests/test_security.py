"""Test pengaman: tidak ada API key di kode, .env tidak ter-commit, log tersensor, jaringan diblokir."""

from __future__ import annotations

import logging
import re
import socket

import pytest

from config.logging_setup import LEARNING_LOGGER, RedactingFormatter, setup_logging
from config.settings import PROJECT_ROOT

# API key dan secret Binance berupa 64 karakter alfanumerik.
KEY_PATTERN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9]{64}(?![A-Za-z0-9])")
SKIP_DIRS = {".venv", "venv", ".git", "__pycache__", ".pytest_cache", "data", "logs"}


def project_files():
    for path in PROJECT_ROOT.rglob("*"):
        if path.is_file() and not SKIP_DIRS.intersection(path.relative_to(PROJECT_ROOT).parts):
            if path.suffix in {".py", ".md", ".txt", ".ini", ".toml", ".json", ".yml", ".yaml"} or path.name == ".env.example":
                yield path


def test_tidak_ada_api_key_tertulis_di_kode():
    offenders = [str(p) for p in project_files() if KEY_PATTERN.search(p.read_text(encoding="utf-8", errors="ignore"))]
    assert offenders == []


def test_file_env_diabaikan_git():
    lines = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in lines
    assert "!.env.example" in lines


def test_formatter_menyensor_rahasia():
    secret = "S" * 64
    formatter = RedactingFormatter([secret, "abc"])  # rahasia pendek diabaikan
    record = logging.LogRecord("x", logging.ERROR, __file__, 1, "gagal dengan key %s abc", (secret,), None)
    text = formatter.format(record)
    assert secret not in text
    assert "***" in text and "abc" in text


def test_setup_logging_idempoten_dan_menulis_file(tmp_path):
    setup_logging("INFO", tmp_path, secrets=["rahasia-panjang-sekali"])
    setup_logging("INFO", tmp_path, secrets=["rahasia-panjang-sekali"])
    root = logging.getLogger()
    managed = [h for h in root.handlers if getattr(h, "_bot_managed_handler", False)]
    assert len(managed) == 2  # console + bot.log, tidak dobel

    logging.getLogger("core.test").info("token rahasia-panjang-sekali bocor?")
    logging.getLogger(LEARNING_LOGGER).info("Penalti DEKAT_RESISTANCE dinaikkan 0.05")
    for handler in managed:
        handler.flush()
    for handler in logging.getLogger(LEARNING_LOGGER).handlers:
        handler.flush()

    bot_log = (tmp_path / "bot.log").read_text(encoding="utf-8")
    assert "rahasia-panjang-sekali" not in bot_log and "token ***" in bot_log
    assert "DEKAT_RESISTANCE" in (tmp_path / "learning.log").read_text(encoding="utf-8")


def test_guard_jaringan_memblokir_akses_ke_binance():
    with pytest.raises(RuntimeError, match="jaringan|DNS"):
        socket.create_connection(("api.binance.com", 443), timeout=1)
