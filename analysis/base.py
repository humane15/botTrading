"""Tipe dasar hasil analisis yang dipakai semua modul di folder analysis/."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    """Batasi nilai ke [low, high]. NaN dianggap netral (0 dibatasi ke rentang)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        value = 0.0
    return max(low, min(high, float(value)))


def is_valid(*values: Any) -> bool:
    """True jika semua nilai berupa angka yang bukan NaN."""
    for value in values:
        if value is None:
            return False
        try:
            if math.isnan(float(value)):
                return False
        except (TypeError, ValueError):
            return False
    return True


@dataclass(frozen=True)
class SignalScore:
    """Skor satu modul analisis pada satu timeframe.

    * score: -1 (sangat bearish) sampai +1 (sangat bullish), otomatis dibatasi.
    * reason: penjelasan lengkap dalam Bahasa Indonesia.
    * label: teks pendek untuk ringkasan alasan sinyal, kosong jika tidak ada
      kejadian penting (misal "Fib 61.8%" atau "MACD cross").
    * tags: label mesin untuk memori pola Fase 5 (misal "fib_618"). Tag pertama
      selalu yang sesuai dengan `label` dan menjadi kunci pola.
    * details: angka mentah untuk jurnal trade dan fitur machine learning.
    """

    name: str
    score: float
    reason: str
    label: str = ""
    tags: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", round(clamp(self.score), 6))
        object.__setattr__(self, "tags", tuple(self.tags))

    @classmethod
    def neutral(cls, name: str, reason: str, **details: Any) -> SignalScore:
        return cls(name=name, score=0.0, reason=reason, details=details)


def has_columns(frame: pd.DataFrame, *columns: str) -> bool:
    return all(column in frame.columns for column in columns)


def column_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Isi satu kolom sebagai array numpy read only, tanpa membuat Series pandas.

    Fungsi analisis dipanggil puluhan kali per coin per candle (dan jutaan kali
    di backtest), sedangkan membuat Series pandas makan waktu sekitar 10 mikrodetik.
    `_get_column_array` adalah jalur internal pandas yang membaca array kolom
    langsung; jika suatu saat tidak ada, dipakai jalur publik yang lebih lambat.
    """
    loc = frame.columns.get_loc(column)
    getter = getattr(frame, "_get_column_array", None)
    values = np.asarray(getter(loc) if getter is not None else frame.iloc[:, loc].to_numpy(), dtype="float64")
    view = values.view()
    view.flags.writeable = False
    return view


def last(frame: pd.DataFrame, column: str, offset: int = 1) -> float:
    """Nilai `column` pada candle ke-`offset` dari belakang (1 = candle terakhir).

    Mengembalikan NaN jika kolom tidak ada atau data kurang.
    """
    if column not in frame.columns or len(frame) < offset:
        return float("nan")
    return float(column_values(frame, column)[-offset])


def tail(frame: pd.DataFrame, column: str, count: int) -> np.ndarray:
    """`count` nilai terakhir dari `column` sebagai array numpy (read only)."""
    return column_values(frame, column)[-count:]
