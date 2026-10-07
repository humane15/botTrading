"""Ukuran posisi berbasis risiko dan jarak stop loss.

Rumus utama (fixed fractional):
    jumlah coin = (modal x risiko per trade) / rugi per coin jika stop tersentuh

Rugi per coin sudah memasukkan fee beli, fee jual, dan slippage, sehingga
rugi nyata saat stop tersentuh sama dengan risiko yang direncanakan.

Batasan tambahan karena bot spot tanpa leverage:
* Porsi modal per posisi = modal / jumlah slot. Jumlah slot (maksimal 5)
  otomatis dikurangi jika modal kecil, karena setiap posisi harus cukup besar
  untuk dibagi dua (take profit bertahap) dan tiap bagian tetap di atas min
  notional Binance.
* Tidak boleh melebihi saldo USDT yang tersedia.
* Anti martingale: ukuran selalu dihitung dari modal SEKARANG dengan persen
  risiko tetap. Pengali regime hanya boleh mengecilkan (high volatility x0.5),
  tidak pernah membesarkan, jadi posisi tidak pernah dinaikkan setelah rugi.

Karena porsi per slot dibatasi, risiko aktual bisa lebih kecil dari
RISK_PER_TRADE saat stop sangat dekat (misal stop 1% pada posisi 20% modal
hanya merisikokan 0.2% modal). Ini disengaja: lebih aman daripada memusatkan
seluruh modal ke satu coin.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.orders import SymbolRules


@dataclass(frozen=True)
class SizingResult:
    qty: float              # jumlah coin, sudah dibulatkan ke step size
    notional: float         # nilai posisi dalam USDT di harga entry
    risk_amount: float      # rugi (USDT) jika stop tersentuh, termasuk fee dan slippage
    risk_pct: float         # risk_amount / modal
    limited_by: str         # "risiko", "porsi_slot", atau "saldo"
    max_positions: int      # jumlah slot posisi efektif untuk modal ini
    reason: str = ""        # alasan ditolak (jika qty 0)

    @property
    def ok(self) -> bool:
        return self.qty > 0 and not self.reason


def loss_per_unit(entry: float, stop: float, fee_rate: float, slippage: float) -> float:
    """Rugi per coin jika beli di `entry` lalu keluar lewat stop market di `stop`."""
    buy_cost = entry * (1 + slippage) * (1 + fee_rate)
    sell_value = stop * (1 - slippage) * (1 - fee_rate)
    return buy_cost - sell_value


def min_trade_notional(min_notional: float, buffer: float, split_exit: bool = True) -> float:
    """Nilai posisi terkecil yang masih bisa dikelola (dua bagian jika TP bertahap)."""
    return min_notional * buffer * (2 if split_exit else 1)


def effective_max_positions(equity: float, max_positions: int, min_trade: float) -> int:
    """Jumlah slot posisi: maksimal `max_positions`, dikurangi jika modal kecil."""
    if equity <= 0:
        return 0
    if min_trade <= 0:
        return max_positions
    return max(0, min(max_positions, int(equity // min_trade)))


def _rejected(reason: str, max_positions: int, limited_by: str = "") -> SizingResult:
    return SizingResult(0.0, 0.0, 0.0, 0.0, limited_by, max_positions, reason)


def calculate_position_size(
    *,
    equity: float,
    free_quote: float,
    entry: float,
    stop: float,
    rules: SymbolRules,
    risk_per_trade: float,
    max_positions: int,
    size_multiplier: float = 1.0,
    fee_rate: float = 0.001,
    slippage: float = 0.0005,
    entry_buffer: float = 0.003,
    min_notional_buffer: float = 1.1,
    split_exit: bool = True,
) -> SizingResult:
    """Hitung jumlah coin untuk satu entry.

    entry_buffer: batas harga order IOC di atas harga entry (dana yang dikunci Binance).
    split_exit: posisi harus bisa dibagi dua di atas min notional (TP bertahap).
    """
    min_trade = min_trade_notional(rules.min_notional, min_notional_buffer, split_exit)
    slots = effective_max_positions(equity, max_positions, min_trade)
    if equity <= 0:
        return _rejected("modal kosong", 0)
    if not 0 < stop < entry:
        return _rejected("stop loss harus di bawah harga entry", slots)
    if slots == 0:
        return _rejected(f"modal ${equity:.2f} terlalu kecil untuk min notional Binance (${min_trade:.2f} per posisi)", 0)

    # Pengali hanya boleh mengecilkan ukuran (anti martingale).
    multiplier = min(max(size_multiplier, 0.0), 1.0)
    risk_budget = equity * risk_per_trade * multiplier
    per_unit = loss_per_unit(entry, stop, fee_rate, slippage)
    order_price = entry * (1 + entry_buffer)

    candidates = {
        "risiko": risk_budget / per_unit,
        "porsi_slot": (equity / slots) / order_price,
        "saldo": max(free_quote, 0.0) * 0.998 / order_price,
    }
    limited_by = min(candidates, key=candidates.get)
    qty = rules.qty_down(candidates[limited_by])

    # Setiap bagian exit (dihitung di harga stop, harga terendah yang direncanakan)
    # harus tetap di atas min notional dan min qty Binance.
    parts = 2 if split_exit else 1
    part_qty = qty / parts
    if not rules.is_sellable(part_qty, stop, min_notional_buffer):
        return _rejected(
            f"ukuran {qty:g} {rules.base} terlalu kecil: tiap bagian exit harus >= "
            f"{rules.min_notional * min_notional_buffer:.2f} {rules.quote} (dibatasi {limited_by})",
            slots,
            limited_by,
        )
    risk_amount = qty * per_unit
    return SizingResult(
        qty=qty,
        notional=qty * entry,
        risk_amount=risk_amount,
        risk_pct=risk_amount / equity,
        limited_by=limited_by,
        max_positions=slots,
    )
