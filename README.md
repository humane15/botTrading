# Bot Trading Spot Binance (Multi Timeframe + Learning)

Bot trading spot Binance berbasis Python yang menganalisis 75 coin pada
timeframe 5m, 15m, 30m, dan 1h, memakai analisis teknikal ala trader
(Support/Resistance, Moving Average, RSI, MACD, Bollinger Bands, Fibonacci),
dan belajar dari jurnal trade-nya sendiri.

> **Peringatan risiko.** Trading crypto berisiko tinggi dan Anda bisa
> kehilangan seluruh modal. Bot ini **tidak menjamin keuntungan**. Setiap
> klaim performa hanya boleh berdasarkan hasil backtest *out of sample*, dan
> hasil masa lalu tidak menjamin hasil masa depan.

## Status Pengerjaan

| Fase | Isi | Status |
| --- | --- | --- |
| 1 | Fondasi: config `.env`, koneksi exchange + retry, pemilihan 75 coin, data feed multi timeframe + cache | **Selesai** |
| 2 | Analisis: indikator, S&R, Fibonacci, regime, confluence engine | **Selesai** |
| 3 | Risiko dan order: sizing, risk manager, state recovery, partial fill | Belum |
| 4 | Backtest event driven + laporan | Belum |
| 5 | Learning: jurnal, mistake analyzer, pattern memory, model ML | Belum |
| 6 | Paper trading: loop utama, logging, notifikasi Telegram | Belum |
| 7 | Live (opsional, setelah paper trading 2 sampai 4 minggu positif) | Belum |

## Instalasi

Butuh Python 3.11 atau lebih baru.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env               # lalu sesuaikan isinya
```

## Keamanan API Key

* Buat API key dengan izin **Enable Reading** dan **Enable Spot & Margin Trading** saja.
* **Jangan pernah** mengaktifkan **Enable Withdrawals**.
* Wajib aktifkan **Restrict access to trusted IPs only** (IP whitelist).
* API key hanya ditulis di file `.env` (sudah masuk `.gitignore`), tidak pernah di kode.
* Mode paper tidak butuh API key karena hanya memakai data publik.
* `ExchangeClient.assert_api_key_safe()` memeriksa izin API key ke Binance
  (`/sapi/v1/account/apiRestrictions`) dan menolak key yang punya izin withdraw
  atau tidak dibatasi IP. Pemeriksaan ini akan diwajibkan sebelum mode live berjalan.
* Nilai rahasia otomatis disensor (`***`) jika sampai muncul di log.

## Mode Trading

* `paper` (default): harga live, saldo virtual.
* `live`: uang sungguhan. Hanya aktif jika **`.env` berisi `TRADING_MODE=live`**
  **dan** bot dijalankan dengan flag **`--confirm-live`**, serta API key terisi.
* `backtest`: simulasi pada data historis dengan kode sinyal yang sama persis.

## Perintah

```bash
python main.py universe               # tampilkan 75 coin (pakai cache 24 jam)
python main.py universe --refresh     # paksa pilih ulang dari Binance
python main.py universe --top 10      # tampilkan 10 teratas saja
python main.py fetch SOL/USDT         # ambil candle 5m, 15m, 30m, 1h dan tampilkan ringkasan
python main.py analyze SOL/USDT       # analisis lengkap satu coin: skor, alasan, regime, rencana SL/TP
python main.py scan --top 10          # analisis semua coin universe, tampilkan sinyal teratas
python main.py backtest               # Fase 4
python main.py train                  # Fase 5
python main.py report                 # Fase 4 dan 5
python main.py paper                  # Fase 6
python main.py live --confirm-live    # Fase 7
```

Gunakan `--env-file path/ke/.env` untuk memakai file konfigurasi lain.

## Cara Kerja Fase 1

### Pemilihan 75 coin (`core/universe.py`)

1. Ambil semua pair spot `/USDT` yang aktif (status `TRADING`).
2. Buang stablecoin (USDC, FDUSD, TUSD, DAI, USD1, dan lain lain, termasuk
   deteksi otomatis koin yang harganya terpaku di sekitar 1 USD), mata uang fiat,
   token leverage (UP/DOWN/BULL/BEAR), token pembungkus seperti WBTC dan WBETH
   (agar eksposur tidak dobel dengan BTC/ETH), coin ber-tag Monitoring, coin
   yang dijadwalkan delisting (butuh API key), dan simbol di `EXCLUDED_SYMBOLS`.
3. Urutkan berdasarkan `quoteVolume` 24 jam dengan minimum 5 juta USD.
4. Ambil 75 teratas dan simpan ke `data/universe.json` beserta tanggalnya.
   Daftar dipakai ulang selama 24 jam.

Jika API gagal, bot memakai `data/universe.json` terakhir. Jika file itu belum
ada, bot memakai daftar cadangan di `config/fallback_symbols.py` yang tetap
divalidasi terhadap `load_markets()`, termasuk pemetaan ticker lama ke ticker
baru (contoh MATIC menjadi POL, FTM menjadi S, RNDR menjadi RENDER).

### Koneksi exchange dan rate limit (`core/exchange.py`)

* Memakai `ccxt.async_support` dan hanya memuat market spot.
* Maksimal 5 request paralel (`asyncio.Semaphore`, atur lewat `MAX_CONCURRENT_REQUESTS`).
* Error jaringan dan HTTP 429 diulang dengan **exponential backoff** plus jitter:
  1, 2, 4, 8, 16 detik (maksimal `RETRY_MAX_DELAY`).
* Saat terkena rate limit, **semua** request ikut jeda (cooldown global) karena
  batas Binance dihitung per IP. Header `Retry-After` dihormati, dan HTTP 418
  (ban IP) membuat bot menunggu minimal 2 menit.
* Order **tidak** dikirim ulang saat timeout atau HTTP 5xx karena statusnya
  tidak pasti (mencegah order ganda). Penolakan 429 tetap aman diulang.
* Error timestamp `-1021` memicu sinkron ulang jam lalu dicoba sekali lagi.

### Data feed multi timeframe (`core/data_feed.py`)

* Hanya candle yang **sudah tertutup** yang dipakai, sehingga sinyal tidak
  berubah ubah dan perilaku live sama dengan backtest. Candle baru dianggap
  final 2 detik setelah tutup (`CANDLE_CLOSE_GRACE_MS`).
* Cache per (simbol, timeframe). Timeframe hanya diambil ulang saat candle
  barunya tutup: candle 1h diperbarui sekali per jam walau scan tiap 5 menit.
* Update inkremental: hanya candle baru yang diminta, lalu digabung dengan cache.
* 75 coin x 4 timeframe diambil paralel dengan `asyncio.gather`; kegagalan satu
  coin tidak mengganggu coin lain.
* `fetch_history()` mengambil data historis panjang dengan paginasi (untuk backtest).

## Cara Kerja Fase 2 (Analisis)

Semua modul ada di `analysis/` dan setiap modul skor mengembalikan skor
**-1 (sangat bearish) sampai +1 (sangat bullish)** beserta penjelasan teks dan
tag untuk memori pola (Fase 5).

### Indikator (`analysis/indicators.py`)

EMA 20/50/200, RSI 14, MACD 12/26/9, Bollinger 20/2 (bandwidth + persentil 100
candle), ATR 14 (+ persentil), ADX 14 (+DI/-DI), OBV, dan rasio volume.

* Ditulis dengan pandas/numpy murni sebagai satu sumber perhitungan, sehingga
  backtest dan live dijamin identik. pandas-ta saat ini hanya ada versi beta
  yang rawan gagal di-install, jadi tidak dipakai.
* Hasilnya **dicocokkan dengan TA-Lib** di unit test (selisih sekitar 1e-15).
* Semua indikator kausal (tanpa look ahead), dan ini diuji: nilai di candle t tidak
  berubah walau data setelah t ditambahkan.
* Data feed kini menyimpan 1000 candle per timeframe, sehingga EMA 200 live hanya
  selisih sekitar 0.0004% dari EMA 200 pada riwayat penuh di backtest.

### Modul skor

| Modul | Isi |
| --- | --- |
| `support_resistance.py` | Pivot 1h dan 15m digabung jadi zona (0.5 ATR). Kekuatan zona = sentuhan + volume + kebaruan. Pantulan support = positif, mendekati resistance = negatif. Saran stop loss (di bawah support, 1.5 sampai 2.5 ATR) dan rasio ruang ke resistance |
| `moving_average.py` | Susunan EMA 20 > 50 > 200, slope EMA 50 (satuan ATR), golden/death cross, pullback ke EMA 20/50 |
| `rsi.py` | RSI dibaca sesuai konteks tren (uptrend kuat boleh 60 sampai 80), bullish/bearish divergence |
| `macd.py` | Crossover, posisi terhadap nol, histogram. Cross di bawah nol saat tren 1h naik diberi skor lebih tinggi |
| `bollinger.py` | Squeeze (persentil bandwidth di bawah 20%), breakout dengan volume setelah squeeze, mean reversion di pasar ranging |
| `fibonacci.py` | Leg swing low ke swing high terakhir (minimal 3 ATR) pada 1h atau 15m, level 23.6% sampai 78.6%. Skor tertinggi di zona 50% sampai 61.8% yang bertepatan dengan support/EMA. Extension 127.2% dan 161.8% sebagai target |
| `volume.py`, `candles.py`, `structure.py` | Lonjakan volume + OBV, trigger candle 5m (engulfing, hammer, breakout candle), struktur higher high / higher low |
| `regime.py` | Trending Up / Ranging / Trending Down (ADX + susunan EMA), High Volatility (ATR persentil 90%, ukuran posisi x0.5), circuit breaker BTC (turun 3% dalam 1 jam, stop entry 2 jam) |

### Confluence engine (`analysis/confluence.py`)

* Peran timeframe: **1h** tren dan S&R besar, **30m** struktur, **15m** setup, **5m** timing entry.
* Skor timeframe = rata rata tertimbang skor modulnya; skor akhir
  `= 50 x (1 + rata rata tertimbang skor timeframe - penalti)`, hasilnya 0 sampai 100.
* **Bobot disimpan di SQLite** (`data/bot.db`, tabel `signal_weights`), bukan
  hardcode. Perubahan oleh modul learning dibatasi maksimal 10% per siklus dan
  dicatat di `logs/learning.log`.
* Entry hanya valid jika **semua** gerbang lolos:
  * minimal 3 dari 4 timeframe searah (bullish);
  * tren 1h tidak bearish dan regime coin bukan Trending Down;
  * circuit breaker BTC tidak aktif;
  * ada setup yang sesuai regime (trending: pullback Fibonacci/EMA; ranging: beli di support);
  * jarak ke resistance minimal 1.5 x jarak ke stop loss;
  * skor minimal `MIN_SIGNAL_SCORE` (default 65).
* Hasil berisi alasan (contoh `Pantulan Fib 61.8% + support 1h + MACD cross 5m`),
  rencana entry/SL/TP1/target Fibonacci, kondisi berisiko (untuk penalti
  learning), dan snapshot fitur untuk jurnal dan ML.
* Analisis 1h/30m/15m disimpan di cache dan baru dihitung ulang saat candle
  timeframe itu berganti (hasil identik, diuji), sehingga scan dan backtest cepat.

## Struktur Proyek

```
.
├── .env.example
├── config/
│   ├── settings.py          # Settings pydantic dari .env + gerbang mode live
│   ├── logging_setup.py     # log console, logs/bot.log, logs/learning.log (tersensor)
│   └── fallback_symbols.py  # 75 coin cadangan + peta rename ticker
├── core/
│   ├── exchange.py          # ccxt async, semaphore, retry, backoff 429, cek izin API key
│   ├── data_feed.py         # OHLCV multi timeframe + cache candle tertutup
│   ├── universe.py          # pemilihan 75 coin dinamis
│   └── database.py          # koneksi SQLite (data/bot.db)
├── analysis/
│   ├── indicators.py        # EMA, RSI, MACD, BB, ATR, ADX, OBV (pandas/numpy, cocok TA-Lib)
│   ├── pivots.py            # swing high/low
│   ├── support_resistance.py
│   ├── moving_average.py
│   ├── rsi.py
│   ├── macd.py
│   ├── bollinger.py
│   ├── fibonacci.py
│   ├── volume.py, candles.py, structure.py
│   ├── regime.py            # regime pasar + circuit breaker BTC
│   ├── weights.py           # bobot sinyal di SQLite
│   └── confluence.py        # skor 0..100 multi timeframe + gerbang entry
├── learning/                # Fase 5
├── risk/                    # Fase 3
├── backtest/                # Fase 4
├── main.py                  # CLI
├── tests/                   # unit test (semua respons Binance di-mock)
└── README.md
```

Folder `data/` (universe, database) dan `logs/` dibuat otomatis saat bot berjalan.

## Menjalankan Test

```bash
pytest
```

`requirements-dev.txt` juga memasang TA-Lib (paket siap pakai tersedia untuk
Windows, macOS, dan Linux) khusus untuk test pembanding indikator. Jika TA-Lib
tidak terpasang, test pembanding itu otomatis dilewati.

Semua unit test memakai mock `ccxt` (`unittest.mock` / `pytest-mock`) untuk
saldo, order, ticker, dan OHLCV. Test juga memasang penjaga jaringan: setiap
upaya koneksi ke luar localhost langsung gagal, jadi test dijamin tidak pernah
menembak API Binance sungguhan.
