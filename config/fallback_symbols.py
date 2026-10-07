"""Daftar cadangan 75 coin, dipakai hanya jika pemilihan dinamis gagal.

Daftar ini TIDAK dipakai mentah-mentah: setiap simbol tetap divalidasi
terhadap exchange.load_markets() (lihat core/universe.py) karena coin bisa
delisting atau berganti ticker. Simbol yang namanya sudah berubah otomatis
dipetakan lewat SYMBOL_RENAMES.
"""

from __future__ import annotations

# Disusun dari pair /USDT dengan likuiditas tinggi di Binance spot,
# kira kira urut dari volume terbesar.
FALLBACK_SYMBOLS: tuple[str, ...] = (
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "TRX/USDT", "AVAX/USDT", "LINK/USDT",
    "DOT/USDT", "TON/USDT", "SHIB/USDT", "LTC/USDT", "BCH/USDT",
    "NEAR/USDT", "UNI/USDT", "APT/USDT", "ICP/USDT", "ETC/USDT",
    "POL/USDT", "FIL/USDT", "ATOM/USDT", "ARB/USDT", "OP/USDT",
    "SUI/USDT", "INJ/USDT", "HBAR/USDT", "XLM/USDT", "RENDER/USDT",
    "FET/USDT", "PEPE/USDT", "WIF/USDT", "SEI/USDT", "TIA/USDT",
    "AAVE/USDT", "IMX/USDT", "STX/USDT", "GRT/USDT", "LDO/USDT",
    "RUNE/USDT", "ALGO/USDT", "VET/USDT", "S/USDT", "ENA/USDT",
    "JUP/USDT", "PYTH/USDT", "WLD/USDT", "ORDI/USDT", "BONK/USDT",
    "FLOKI/USDT", "GALA/USDT", "SAND/USDT", "MANA/USDT", "AXS/USDT",
    "CRV/USDT", "EGLD/USDT", "THETA/USDT", "TAO/USDT", "CAKE/USDT",
    "DYDX/USDT", "ZRO/USDT", "ENS/USDT", "NOT/USDT", "JTO/USDT",
    "PENDLE/USDT", "TRUMP/USDT", "ETHFI/USDT", "W/USDT", "STRK/USDT",
    "EIGEN/USDT", "NEIRO/USDT", "PNUT/USDT", "ARKM/USDT", "ZEC/USDT",
)

# Ticker lama -> ticker baru di Binance (rebrand, migrasi, atau merger token).
SYMBOL_RENAMES: dict[str, str] = {
    "MATIC": "POL",     # Polygon, September 2024
    "FTM": "S",         # Fantom menjadi Sonic, Januari 2025
    "RNDR": "RENDER",   # Render, Juli 2024
    "AGIX": "FET",      # merger Artificial Superintelligence Alliance, 2024
    "OCEAN": "FET",     # merger Artificial Superintelligence Alliance, 2024
    "GAL": "G",         # Galxe menjadi Gravity, 2024
    "EOS": "A",         # EOS menjadi Vaulta, 2025
    "TOMO": "VIC",      # TomoChain menjadi Viction, 2024
    "BTT": "BTTC",      # redenominasi BitTorrent, 2022
    "LEND": "AAVE",     # migrasi Aave, 2020
    "ERD": "EGLD",      # Elrond menjadi MultiversX, 2020
    "NPXS": "PUNDIX",   # redenominasi Pundi X, 2021
    "BCHABC": "BCH",
}
