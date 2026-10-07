"""Fixture bersama untuk seluruh test.

Aturan: test TIDAK BOLEH menembak API Binance sungguhan. Fixture
`block_network` (autouse) memblokir koneksi socket dan DNS ke luar
localhost, sehingga panggilan jaringan yang lolos dari mock langsung gagal.
Semua respons ccxt disimulasikan dengan unittest.mock / pytest-mock.
"""

from __future__ import annotations

import logging
import socket
from typing import Any

import pytest
from helpers import ASYNC_METHODS, FakeTime, make_settings

from config.logging_setup import _MANAGED_ATTR
from config.settings import Settings
from core.exchange import ExchangeClient

_LOCAL_HOSTS = {None, "localhost", "127.0.0.1", "::1", b"localhost"}
_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex
_REAL_GETADDRINFO = socket.getaddrinfo


def _is_local(sock: socket.socket, address: Any) -> bool:
    if sock.family == getattr(socket, "AF_UNIX", object()):
        return True
    host = address[0] if isinstance(address, tuple) else address
    return host in _LOCAL_HOSTS


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def guarded_connect(self: socket.socket, address: Any) -> Any:
        if _is_local(self, address):
            return _REAL_CONNECT(self, address)
        raise RuntimeError(f"Test mencoba mengakses jaringan ({address}). Gunakan mock ccxt.")

    def guarded_connect_ex(self: socket.socket, address: Any) -> Any:
        if _is_local(self, address):
            return _REAL_CONNECT_EX(self, address)
        raise RuntimeError(f"Test mencoba mengakses jaringan ({address}). Gunakan mock ccxt.")

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host in _LOCAL_HOSTS:
            return _REAL_GETADDRINFO(host, *args, **kwargs)
        raise RuntimeError(f"Test mencoba resolve DNS ({host}). Gunakan mock ccxt.")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)


@pytest.fixture(autouse=True)
def reset_logging():
    """Lepas handler logging yang dipasang setup_logging() setelah tiap test."""
    yield
    for name in ("", "learning"):
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            if getattr(handler, _MANAGED_ATTR, False):
                logger.removeHandler(handler)
                handler.close()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def mock_exchange(mocker):
    """Pengganti ccxt.async_support.binance: semua method async berupa AsyncMock."""
    exchange = mocker.MagicMock(name="ccxt.binance")
    exchange.id = "binance"
    exchange.markets = {}
    exchange.last_response_headers = {}
    for name in ASYNC_METHODS:
        setattr(exchange, name, mocker.AsyncMock(name=name))
    return exchange


@pytest.fixture
def fake_time() -> FakeTime:
    return FakeTime()


@pytest.fixture
def client(settings, mock_exchange, fake_time) -> ExchangeClient:
    return ExchangeClient(settings, mock_exchange, sleep=fake_time.sleep, monotonic=fake_time.monotonic, jitter=0)
