"""State recovery: cocokkan database posisi dengan kondisi riil di bursa saat bot start.

Wajib dijalankan SEBELUM scanning pertama setelah restart. Langkahnya:
1. Setiap posisi aktif di database disinkronkan dengan bursa (PositionManager.sync):
   * entry yang belum tercatat (bot mati setelah order dikirim) diselesaikan
     lewat client order id;
   * take profit atau stop loss yang terisi saat bot mati dicatat (PnL ikut dihitung);
   * proteksi yang hilang (stop dibatalkan/kedaluwarsa) dipasang ulang.
2. Jika coin posisi ternyata sudah tidak ada di akun (dijual manual di luar
   bot), posisi ditutup dengan catatan "ditutup di luar bot" (dideteksi oleh sync).
   Coin yang masih ada tetapi terkunci order lain TIDAK dianggap terjual: order
   bot yang tidak tercatat dibatalkan lalu stop dipasang ulang, sedangkan order
   manual dibiarkan dan dilaporkan sebagai error.
3. Saldo coin lain yang bukan milik posisi bot dan bernilai di atas min notional
   dilaporkan dan coin itu diblokir dari entry baru (agar eksposur tidak dobel).
4. Order milik bot (client id berawalan "tb-") yang tidak terhubung ke posisi
   aktif dibatalkan agar tidak tereksekusi tanpa pengawasan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import ccxt

from core.orders import OrderError, is_bot_order
from risk.position_manager import PositionManager
from risk.positions import CLOSED, FAILED, PENDING, Position
from risk.risk_manager import RiskManager

log = logging.getLogger(__name__)


@dataclass
class RecoveryReport:
    checked: int = 0
    resumed: list[str] = field(default_factory=list)
    entries_completed: list[str] = field(default_factory=list)
    filled_while_offline: list[str] = field(default_factory=list)
    closed_while_offline: list[str] = field(default_factory=list)
    failed_entries: list[str] = field(default_factory=list)
    reprotected: list[str] = field(default_factory=list)
    closed_externally: list[str] = field(default_factory=list)
    untracked: dict[str, float] = field(default_factory=dict)
    orphan_orders_canceled: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = [f"[RECOVERY] {self.checked} posisi aktif diperiksa"]
        groups = (
            ("dilanjutkan", self.resumed),
            ("entry diselesaikan", self.entries_completed),
            ("terisi saat bot mati", self.filled_while_offline),
            ("tertutup saat bot mati", self.closed_while_offline),
            ("entry gagal", self.failed_entries),
            ("proteksi dipasang ulang", self.reprotected),
            ("ditutup di luar bot", self.closed_externally),
            ("order yatim dibatalkan", self.orphan_orders_canceled),
            ("error", self.errors),
        )
        out += [f"  {label}: {', '.join(items)}" for label, items in groups if items]
        if self.untracked:
            text = ", ".join(f"{symbol} (${value:,.2f})" for symbol, value in sorted(self.untracked.items()))
            out.append(f"  saldo di luar bot (coin diblokir dari entry baru): {text}")
        return out


def _client_ids(position: Position) -> set[str]:
    ids = {position.entry_client_id, position.tp_client_id, position.stop_a_client_id, position.stop_b_client_id}
    return {cid for cid in ids if cid}


async def recover_state(manager: PositionManager, quote: str, risk: RiskManager | None = None) -> RecoveryReport:
    report = RecoveryReport()
    executor = manager.executor
    positions = manager.store.active(manager.mode)
    report.checked = len(positions)

    for position in positions:
        before_status, before_qty = position.status, position.qty
        events_before = len(manager.events)
        try:
            position = await manager.sync(position)
        except (OrderError, ccxt.BaseError) as exc:
            report.errors.append(f"{position.symbol}: {exc}")
            continue
        label = f"{position.symbol}#{position.id}"
        new_events = manager.events[events_before:]
        if any(text.startswith("[DI LUAR BOT]") for text in new_events):
            report.closed_externally.append(label)
        elif before_status == PENDING:
            (report.failed_entries if position.status == FAILED else report.entries_completed).append(label)
        elif position.status == CLOSED:
            report.closed_while_offline.append(label)
        elif position.qty < before_qty:
            report.filled_while_offline.append(label)
        else:
            report.resumed.append(label)
        if any(text.startswith("[PROTEKSI]") for text in new_events):
            report.reprotected.append(label)
        if any(text.startswith("[PROTEKSI TERTAHAN]") for text in new_events):
            report.errors.append(f"{label}: sebagian coin terkunci order di luar bot, stop belum lengkap")

    balances = await executor.balances()
    active = manager.store.active(manager.mode)
    tracked = {p.base for p in active}
    for asset, (_, total) in balances.items():
        if asset == quote or asset in tracked or total <= 0:
            continue
        symbol = f"{asset}/{quote}"
        try:
            rules = await executor.rules(symbol)
            bid, _ = await executor.best_prices(symbol)
        except (OrderError, ccxt.BaseError):
            continue  # bukan pair yang diperdagangkan bot
        value = total * bid
        if value >= rules.min_notional:
            report.untracked[symbol] = value
    if risk is not None and report.untracked:
        risk.block_symbols(report.untracked)

    known_ids = set().union(*(_client_ids(p) for p in active)) if active else set()
    for order in await executor.open_orders():
        if is_bot_order(order.client_id) and order.client_id not in known_ids:
            await executor.cancel_order(order.symbol, order.client_id)
            report.orphan_orders_canceled.append(order.client_id)

    for line in report.lines():
        log.info(line)
    return report

