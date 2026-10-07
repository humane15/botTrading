"""Data historis untuk backtest: unduh dari Binance, simpan di data/history, baca offline.

Keputusan desain:

* Setiap timeframe (5m, 15m, 30m, 1h) diunduh langsung dari Binance, bukan hasil
  resample, sehingga candle yang dinilai backtest identik dengan yang dilihat bot live.
* Ditambah 1000 candle pemanasan sebelum tanggal mulai (sama dengan jendela data
  bot live), jadi EMA 200 dan persentil ATR sudah stabil sejak candle pertama.
* Disimpan sebagai file .npz per simbol dan timeframe (numpy murni, tanpa pickle).
  Unduhan berikutnya hanya mengambil bagian yang belum ada.
* Aturan pair (step size, tick size, min notional) dan daftar kandidat coin ikut
  disimpan agar backtest bisa diulang tanpa koneksi (--offline).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import TIMEFRAME_MS, Settings
from core.data_feed import OHLCV_COLUMNS, DataFeed, empty_ohlcv_frame, frame_open_ms
from core.exchange import ExchangeClient
from core.orders import SymbolRules

log = logging.getLogger(__name__)

WARMUP_CANDLES = 1000
RULES_FILE = "markets.json"
CANDIDATES_FILE = "candidates.json"


def history_dir(settings: Settings) -> Path:
    return settings.data_dir / "history"


def rules_to_dict(rules: SymbolRules) -> dict[str, Any]:
    return {
        "symbol": rules.symbol,
        "base": rules.base,
        "quote": rules.quote,
        "step_size": rules.step_size,
        "tick_size": rules.tick_size,
        "min_qty": rules.min_qty,
        "max_qty": None if math.isinf(rules.max_qty) else rules.max_qty,
        "min_notional": rules.min_notional,
        "order_types": sorted(rules.order_types),
        "oco_allowed": rules.oco_allowed,
    }


def rules_from_dict(data: Mapping[str, Any]) -> SymbolRules:
    return SymbolRules(
        symbol=str(data["symbol"]),
        base=str(data["base"]),
        quote=str(data["quote"]),
        step_size=float(data["step_size"]),
        tick_size=float(data["tick_size"]),
        min_qty=float(data.get("min_qty") or 0.0),
        max_qty=math.inf if data.get("max_qty") is None else float(data["max_qty"]),
        min_notional=float(data.get("min_notional") or 0.0),
        order_types=frozenset(data.get("order_types") or ()),
        oco_allowed=bool(data.get("oco_allowed", True)),
    )


class HistoryStore:
    """Penyimpanan candle historis per (simbol, timeframe) di folder data/history."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, symbol: str, timeframe: str) -> Path:
        return self.root / f"{symbol.replace('/', '_')}_{timeframe}.npz"

    def has(self, symbol: str, timeframe: str) -> bool:
        return self.path(symbol, timeframe).exists()

    def load(self, symbol: str, timeframe: str) -> pd.DataFrame:
        frame, _ = self._load_with_meta(symbol, timeframe)
        return frame

    def _load_with_meta(self, symbol: str, timeframe: str) -> tuple[pd.DataFrame, int | None]:
        path = self.path(symbol, timeframe)
        if not path.exists():
            return empty_ohlcv_frame(), None
        with np.load(path, allow_pickle=False) as data:
            index = pd.DatetimeIndex(pd.to_datetime(data["open_time"], unit="ms", utc=True), name="timestamp")
            frame = pd.DataFrame({col: data[col].astype("float64") for col in OHLCV_COLUMNS}, index=index)
            requested = int(data["requested_from"]) if "requested_from" in data.files else None
        return frame, requested

    def save(self, symbol: str, timeframe: str, frame: pd.DataFrame, requested_from: int | None = None) -> None:
        """Simpan frame (atomik: tulis file sementara lalu ganti nama)."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path(symbol, timeframe)
        tmp = path.with_name(path.stem + ".tmp.npz")
        arrays = {col: frame[col].to_numpy(dtype="float64") for col in OHLCV_COLUMNS}
        extra = {} if requested_from is None else {"requested_from": np.int64(requested_from)}
        np.savez_compressed(tmp, open_time=frame_open_ms(frame).astype("int64"), **arrays, **extra)
        os.replace(tmp, path)

    def load_symbol(self, symbol: str, timeframes: Iterable[str]) -> dict[str, pd.DataFrame]:
        return {tf: self.load(symbol, tf) for tf in timeframes}

    # ------------------------------------------------------------------
    # Aturan pair dan daftar kandidat
    # ------------------------------------------------------------------
    def save_rules(self, rules: Mapping[str, SymbolRules]) -> None:
        merged = {**self.load_rules(), **rules}
        self._write_json(RULES_FILE, {symbol: rules_to_dict(r) for symbol, r in sorted(merged.items())})

    def load_rules(self) -> dict[str, SymbolRules]:
        data = self._read_json(RULES_FILE) or {}
        return {symbol: rules_from_dict(item) for symbol, item in data.items()}

    def save_candidates(self, symbols: Sequence[str], source: str) -> None:
        self._write_json(CANDIDATES_FILE, {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": source,
            "symbols": list(symbols),
        })

    def load_candidates(self) -> list[str]:
        data = self._read_json(CANDIDATES_FILE) or {}
        return [str(s) for s in data.get("symbols") or []]

    def _write_json(self, name: str, payload: Any) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / name
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def _read_json(self, name: str) -> Any:
        path = self.root / name
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class DownloadReport:
    symbols: int = 0
    series_updated: int = 0
    series_complete: int = 0
    new_candles: int = 0
    api_calls: int = 0
    failed: dict[str, str] = field(default_factory=dict)

    def text(self) -> str:
        line = (
            f"Data historis: {self.symbols} coin | {self.series_updated} seri diperbarui, "
            f"{self.series_complete} sudah lengkap | {self.new_candles:,} candle baru | {self.api_calls} request API"
        )
        if self.failed:
            line += f" | gagal: {', '.join(sorted(self.failed))}"
        return line


def warmup_start_ms(start_ms: int, timeframe: str, warmup: int = WARMUP_CANDLES) -> int:
    return start_ms - warmup * TIMEFRAME_MS[timeframe]


async def _update_series(
    feed: DataFeed, store: HistoryStore, symbol: str, timeframe: str, need_start: int, end_ms: int, report: DownloadReport
) -> None:
    tf_ms = TIMEFRAME_MS[timeframe]
    existing, requested_from = store._load_with_meta(symbol, timeframe)
    opens = frame_open_ms(existing)
    parts = [existing]
    head_known = requested_from is not None and requested_from <= need_start
    if existing.empty:
        parts.append(await feed.fetch_history(symbol, timeframe, need_start, end_ms))
    else:
        if need_start < int(opens[0]) and not head_known:
            parts.append(await feed.fetch_history(symbol, timeframe, need_start, int(opens[0])))
        if int(opens[-1]) + tf_ms < end_ms:
            parts.append(await feed.fetch_history(symbol, timeframe, int(opens[-1]) + tf_ms, end_ms))
    fetched = [p for p in parts[1:] if not p.empty]
    new_requested = min(need_start, requested_from) if requested_from is not None else need_start
    if not fetched and requested_from == new_requested:
        report.series_complete += 1
        return
    merged = pd.concat([p for p in parts if not p.empty]) if any(not p.empty for p in parts) else empty_ohlcv_frame()
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    report.new_candles += len(merged) - len(existing)
    report.series_updated += 1
    store.save(symbol, timeframe, merged, requested_from=new_requested)


async def download_history(
    client: ExchangeClient,
    settings: Settings,
    store: HistoryStore,
    symbols: Sequence[str],
    start_ms: int,
    end_ms: int,
    *,
    warmup: int = WARMUP_CANDLES,
    progress: Callable[[str], None] | None = None,
) -> DownloadReport:
    """Lengkapi data [start - pemanasan, end) untuk semua simbol dan timeframe di settings."""
    report = DownloadReport(symbols=len(symbols))
    markets = await client.load_markets()
    store.save_rules({s: SymbolRules.from_market(markets[s]) for s in symbols if s in markets})
    for symbol in symbols:
        if symbol not in markets:
            report.failed[symbol] = "tidak ada di Binance spot"
    feed = DataFeed.from_settings(client, settings)
    valid = [s for s in symbols if s in markets]
    done = 0

    async def one(symbol: str) -> None:
        nonlocal done
        try:
            for timeframe in settings.timeframes:
                await _update_series(feed, store, symbol, timeframe, warmup_start_ms(start_ms, timeframe, warmup), end_ms, report)
        except Exception as exc:  # noqa: BLE001  (satu coin gagal tidak menghentikan unduhan coin lain)
            report.failed[symbol] = str(exc)
            log.warning("Unduh data %s gagal: %s", symbol, exc)
        done += 1
        if progress is not None and (done % 10 == 0 or done == len(valid)):
            progress(f"  data historis {done}/{len(valid)} coin")

    await asyncio.gather(*(one(symbol) for symbol in valid))
    report.api_calls = feed.stats.api_calls
    return report
