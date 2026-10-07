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
| 3 | Risiko dan order: sizing, risk manager, state recovery, partial fill | **Selesai** |
| 4 | Backtest event driven + laporan | **Selesai** |
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
python main.py positions              # posisi aktif dan 10 posisi tertutup terakhir (dari database)
python main.py backtest               # backtest 6 bulan terakhir, universe 75 coin (unduh data otomatis)
python main.py backtest --offline     # ulangi backtest dengan data yang sudah diunduh
python main.py backtest --start 2026-01-01 --end 2026-07-01 --capital 1000
python main.py report                 # tampilkan laporan backtest terakhir
python main.py train                  # Fase 5
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

## Cara Kerja Fase 3 (Risiko dan Order)

Fase ini berisi semua yang terjadi setelah sinyal entry muncul: menghitung
ukuran posisi, memeriksa batas risiko, mengirim order, menjaga posisi sampai
ditutup, dan memulihkan keadaan setelah bot restart. Order sungguhan baru
dikirim di Fase 7; sampai saat itu semua dijalankan dengan simulasi bursa.

### Ukuran posisi (`risk/position_sizing.py`)

* Risiko per trade 1% modal (`RISK_PER_TRADE`, maksimal 2%). Jumlah coin =
  risiko dibagi rugi per coin jika stop loss tersentuh.
* Rugi per coin sudah termasuk fee 0.1% saat beli dan jual serta slippage,
  sehingga rugi nyata di stop tidak melebihi 1% modal (kecuali harga gap
  melewati stop).
* Ukuran juga dibatasi porsi satu slot (modal dibagi jumlah slot) dan saldo
  USDT bebas. Batas terkecil yang dipakai, dan alasannya dicatat.
* Dibulatkan ke bawah sesuai step size Binance memakai `Decimal`.
* Bagian TP1 dan bagian sisa (runner) masing masing harus tetap di atas min
  notional Binance x 1.1. Jika tidak, entry ditolak sebelum order dikirim.
* Slot posisi berkurang otomatis saat modal kecil: maksimal 5, atau modal dibagi
  (2 x min notional x 1.1) jika lebih kecil. Contoh: modal $30 hanya 2 slot,
  modal $10 tidak bisa entry.
* Anti martingale: pengali ukuran hanya bisa mengecilkan posisi (regime
  volatilitas tinggi x0.5), tidak pernah membesarkan. Setelah rugi, modal turun
  sehingga ukuran posisi ikut mengecil.

### Risk manager (`risk/risk_manager.py`)

Entry baru ditolak jika:

* file `STOP` ada di root proyek (kill switch);
* rugi hari ini mencapai 5% atau rugi minggu ini mencapai 10% dari nilai akun
  di awal hari atau minggu (UTC, posisi terbuka ikut dihitung). Entry dibuka
  lagi otomatis di periode berikutnya;
* slot posisi penuh atau modal terlalu kecil;
* coin itu sudah punya posisi, atau ada saldo coin itu di luar kendali bot;
* sudah ada 2 posisi pada coin yang korelasi return 1h terhadap BTC (7 hari
  terakhir) minimal 0.8, dan coin baru juga berkorelasi tinggi.

Posisi yang sudah terbuka tidak ditutup paksa: stop loss, TP, dan trailing
stop tetap bekerja.

```bash
touch STOP     # kill switch: hentikan entry baru (Windows: type nul > STOP)
rm STOP        # lanjutkan entry
```

### Siklus posisi (`risk/position_manager.py`)

1. **Entry**: order LIMIT IOC dengan harga paling mahal ask + 0.3%
   (`ENTRY_MAX_SLIPPAGE`). Bagian yang tidak terisi langsung batal, jadi tidak
   ada order beli yang menggantung. Client order id dicatat ke database
   **sebelum** order dikirim, sehingga order tetap bisa dilacak walau bot mati
   tepat setelah mengirim.
2. **Partial fill**: ukuran posisi, stop, dan TP dihitung dari jumlah yang
   benar benar terisi dikurangi fee (fee beli Binance dipotong dalam coin).
3. **Proteksi di bursa**: 50% posisi memakai OCO (TP1 berupa LIMIT_MAKER +
   STOP_LOSS market), 50% sisanya memakai STOP_LOSS market. Stop loss selalu
   ada di Binance, jadi posisi tetap terlindungi walau bot mati. Pair tanpa OCO:
   TP1 dipantau bot dan stop tetap di bursa. Pair tanpa STOP_LOSS market:
   memakai STOP_LOSS_LIMIT dengan batas harga 1% di bawah stop.
4. **TP1**: resistance terdekat (minimal 1.5R). Setelah terisi, stop sisa posisi
   pindah ke breakeven, yaitu harga entry ditambah fee beli dan jual.
5. **Trailing stop**: harga tertinggi dikurangi 2 x ATR 15m. Stop hanya naik
   (minimal 0.25 ATR per langkah agar hemat request) dan tidak pernah turun.
   Jika stop baru sudah di atas harga, sisa posisi langsung dijual market.
6. **Exit**: stop loss, breakeven, trailing stop, atau manual. PnL dan R multiple
   dicatat lengkap dengan fee.

Setiap sinkronisasi memeriksa ulang proteksi di bursa:

* stop yang hilang (misalnya dibatalkan manual di aplikasi) dipasang lagi;
* jika OCO dibatalkan manual, TP1 dipantau bot dan dijual market saat tersentuh;
* coin yang terkunci order lain **tidak** dianggap terjual. Order bot yang tidak
  tercatat (misalnya hasil kirimnya tidak pasti saat jaringan putus) dibatalkan
  lalu diganti stop baru, sedangkan order manual Anda tidak disentuh dan
  dilaporkan sebagai peringatan.

### Eksekusi order (`core/orders.py`)

* Filter Binance dibaca langsung (`LOT_SIZE`, `PRICE_FILTER`, `NOTIONAL` atau
  `MIN_NOTIONAL`). Jumlah dibulatkan ke bawah, harga TP ke atas, harga stop ke
  bawah, semuanya memakai `Decimal`.
* Order **tidak pernah dikirim ulang** saat timeout: bot mencari order itu
  berdasarkan client order id dulu, sehingga tidak ada order ganda.
* `LiveExecutor` menolak berjalan tanpa konfirmasi mode live.

### Simulasi bursa (`core/paper_exchange.py`)

Saldo virtual untuk paper trading dan backtest. Logika posisi di paper,
backtest, dan live adalah kode yang sama; hanya pelaksana ordernya yang
berbeda. Simulasi sengaja dibuat pesimis:

* entry terisi di ask + slippage, fee beli dipotong dari coin, fee jual dari USDT;
* jika TP dan stop tersentuh dalam satu candle, dianggap stop yang terisi dulu;
* jika harga dibuka gap di bawah stop, order terisi di harga open (rugi bisa
  lebih dari 1R);
* TP LIMIT_MAKER baru terisi jika harga melewati TP, bukan sekadar menyentuh.

### State recovery (`risk/state_recovery.py`)

Saat bot start, sebelum scan pertama, isi database dicocokkan dengan Binance:

* order entry yang terkirim sebelum bot mati diselesaikan: jika terisi,
  posisi dilanjutkan dan diberi proteksi; jika tidak ditemukan, ditandai gagal;
* TP1 atau stop yang terisi saat bot mati dicatat, dan stop dipindah ke
  breakeven jika TP1 sudah terisi;
* stop yang hilang dipasang ulang;
* coin yang dijual di luar bot dicatat sebagai `ditutup_di_luar_bot`;
* saldo coin di luar bot dilaporkan dan coin itu diblokir dari entry;
* order bot lama yang tidak terhubung ke posisi aktif dibatalkan.

### Database posisi (`risk/positions.py`)

Tabel `positions` (status, entry, stop, TP, PnL, snapshot sinyal dan fitur untuk
jurnal Fase 5), `orders` (audit semua order), dan `equity_snapshots` (nilai akun
di awal hari dan minggu). Lihat isinya dengan `python main.py positions`.

## Cara Kerja Fase 4 (Backtest)

```bash
python main.py backtest                       # 6 bulan terakhir, modal PAPER_START_BALANCE
python main.py backtest --capital 1000        # modal awal lain
python main.py backtest --months 3 --workers 2
python main.py backtest --symbols SOL/USDT,ETH/USDT --offline
python main.py report                         # baca ulang laporan terakhir
```

Opsi lain: `--start`/`--end` (tanggal UTC), `--candidates` (jumlah kandidat coin
yang diunduh, default 1.4 x UNIVERSE_SIZE), dan `--oos-fraction` (porsi akhir
periode untuk segmen B, default 1/3).

### Data historis (`backtest/data.py`)

* Candle 5m, 15m, 30m, dan 1h diunduh langsung dari Binance (bukan hasil resample),
  ditambah 1000 candle pemanasan per timeframe sebelum tanggal mulai, sama dengan
  jendela data bot live.
* Disimpan di `data/history/` (file `.npz` per coin dan timeframe) beserta aturan
  pair (step size, tick size, min notional). Unduhan berikutnya hanya mengambil
  bagian yang belum ada, dan `--offline` menjalankan ulang tanpa koneksi.
* Unduhan pertama 6 bulan untuk sekitar 105 kandidat butuh kira kira 9.000 request
  (dibatasi semaphore dan backoff rate limit dari Fase 1) dan sekitar 400 MB disk.

### Universe point in time (`backtest/universe.py`)

Memakai daftar 75 coin HARI INI untuk menguji 6 bulan ke belakang akan membuat
hasil terlalu bagus, karena coin yang sekarang ramai biasanya coin yang naik di
periode itu. Karena itu kandidat diunduh lebih banyak, lalu setiap hari dipilih 75
coin dari volume hari sebelumnya (minimal 5 juta USD), persis seperti bot live yang
memperbarui universe tiap 24 jam.

### Mesin backtest (`backtest/signals.py`, `backtest/engine.py`)

1. **Sinyal**: `ConfluenceEngine` yang sama dengan live dievaluasi di setiap candle
   5m tertutup, dengan jendela 1000 candle tertutup per timeframe. Indikator dihitung
   sekali lalu diiris (aman karena semua indikator kausal), sedangkan pivot, zona S&R,
   dan Fibonacci dihitung ulang dari jendela tiap langkah, persis seperti live.
   Regime BTC dan circuit breaker dihitung berurutan dengan kode live. Sinyal tiap
   coin tidak bergantung pada saldo, jadi dihitung paralel per coin (hasilnya
   identik dengan 1 proses, diuji).
2. **Simulasi portofolio (event driven)**, setiap candle 5m berurutan waktu:
   bursa simulasi mengeksekusi stop/TP pada candle yang baru tertutup, lalu
   `PositionManager` menjalankan TP1, breakeven, dan trailing stop; setelah itu sinyal
   baru dicoba lewat `RiskManager` (slot, batas rugi harian/mingguan, korelasi BTC)
   dan position sizing. Semua kode ini sama dengan mode live, dengan jam simulasi
   (waktu candle) untuk batas rugi harian dan catatan waktu posisi.
3. Entry terisi di harga penutupan candle sinyal + slippage; stop dan TP diperiksa
   mulai candle berikutnya. Posisi yang masih terbuka di akhir periode ditutup di
   harga terakhir.

Pengaman tanpa look ahead (semuanya diuji):

* jendela tiap timeframe hanya berisi candle yang sudah tertutup pada waktu keputusan;
* data setelah waktu T diubah drastis, sinyal sampai T tetap sama persis;
* sinyal backtest sama dengan evaluasi live yang menghitung indikator hanya dari
  data sampai T;
* universe hari D hanya memakai volume hari D-1.

Agar 75 coin x 52.000 candle 5m tetap cepat, engine punya **penolakan cepat**:
jika tren 1h, circuit breaker, timeframe 1h/30m/15m, atau ruang ke resistance sudah
PASTI menolak entry (termasuk batas atas skor yang masih mungkin), analisis 5m
dilewati. Keputusan entry dijamin identik dengan evaluasi penuh (diuji pada ribuan
candle acak). Simulasi juga mengecek slot penuh lebih dulu sebelum risk manager,
dengan hasil yang identik (diuji).

Ukuran kerja (diukur pada 89 kandidat x 182 hari, sekitar 3,9 juta evaluasi sinyal,
4 core): sekitar 15 menit, RAM sekitar 1,1 GB untuk proses utama ditambah sekitar
0,5 GB per proses paralel. Kurangi `--workers` jika RAM terbatas.

### Biaya dan dust

* Fee 0.1% per transaksi dan slippage 0.05% (bisa diatur di `.env`), min notional,
  step size, dan tick size per pair, urutan pesimis stop sebelum TP, dan gap.
* Fee beli Binance dipotong dalam coin, lalu jumlah yang bisa dijual dibulatkan ke
  bawah sesuai step size. Sisanya (dust) tetap di akun. Dust dicatat terpisah
  (`dust_qty`) sehingga PnL dan R hanya menghitung coin yang benar benar
  diperdagangkan, dan state recovery tidak menganggap dust milik bot sebagai saldo
  di luar bot.

### Laporan (`backtest/report.py`)

* Total return, CAGR, max drawdown dan lamanya, Sharpe dan Sortino (return harian x
  akar 365), Calmar, win rate, profit factor, rata rata dan median R, t-stat rata
  rata R, expectancy, exposure, total fee.
* Rincian per alasan exit, regime, pola setup, dan coin, return bulanan, serta
  pembanding buy and hold BTC.
* Periode dibagi menjadi segmen A dan segmen B (1/3 terakhir). Parameter strategi
  belum dioptimasi pada data ini, jadi seluruh periode adalah out of sample untuk
  strategi dasar. Di Fase 5, modul learning hanya boleh belajar dari segmen A, dan
  klaim performa memakai segmen B.
* Hasil disimpan di `data/backtests/<waktu>/`: `report.txt`, `report.json`,
  `trades.csv`, `equity.csv`, `events.log`, dan `journal.db` (jurnal trade untuk Fase 5).

Bias yang masih tersisa dan perlu diingat saat membaca hasil: coin yang sudah
delisting tidak bisa diunduh dari Binance, entry dianggap terisi di harga penutupan
candle (bot live mengirim order beberapa detik setelahnya), dan likuiditas order
book tidak disimulasikan (aman untuk modal kecil).

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
│   ├── database.py          # koneksi SQLite (data/bot.db)
│   ├── orders.py            # filter Binance, client order id, LiveExecutor (IOC, stop, OCO)
│   └── paper_exchange.py    # simulasi bursa untuk paper trading dan backtest
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
├── risk/
│   ├── position_sizing.py   # ukuran posisi dari jarak stop (fee + slippage)
│   ├── risk_manager.py      # kill switch, batas rugi, slot, korelasi BTC
│   ├── positions.py         # posisi, order, snapshot ekuitas di SQLite
│   ├── position_manager.py  # entry, OCO, TP1, breakeven, trailing, exit
│   └── state_recovery.py    # pencocokan database dengan Binance saat start
├── backtest/
│   ├── data.py              # unduh dan simpan data historis (data/history)
│   ├── universe.py          # universe 75 coin point in time per hari
│   ├── signals.py           # tahap 1: sinyal per coin, paralel, tanpa look ahead
│   ├── engine.py            # tahap 2: simulasi portofolio event driven
│   └── report.py            # metrik, rincian, pembanding, simpan hasil
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
