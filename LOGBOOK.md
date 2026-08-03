# Logbook Harian — Pengembangan Smart AI (Main Service & Agent Service)

Catatan kerja harian selama enam bulan: pemetaan sistem AutoML yang sudah berjalan, restrukturisasi
lapisan service, pembenahan logging, pengembangan chat assistant berbasis RAG, analitik clustering,
sampai konsistensi jalur realtime.

| | |
|---|---|
| Periode | Maret – Agustus 2026 |
| Hari kerja | 130 hari (26 minggu × 5 hari) |
| Repositori | `main-service` (backend ML), `agent-service` (chat assistant), `frontend-v1` (antarmuka) |

---

## Memahami sistem & menstabilkan dasar

### Minggu 1 — Pemetaan arsitektur
*2 Maret – 6 Maret 2026*

| Tanggal | Hari | Kegiatan | Keluaran |
|---|---|---|---|
| 02/03 | Senin | Menelusuri struktur `app/` dan memetakan alur routes → services → core | Diagram alur + catatan tanggung jawab tiap lapisan |
| 03/03 | Selasa | Membaca `app/core/` dan kontrak dataclass antar lapisan | Daftar contract per task type |
| 04/03 | Rabu | Menelusuri worker pm2 dan `WorkerManager` | Catatan: identitas worker dioper lewat argv |
| 05/03 | Kamis | Menelusuri `pull.py` dan integrasi API SmartLink | Catatan retry, timeout, dan pemetaan tagname→row_id |
| 06/03 | Jumat | Menelusuri InfluxDB dan skema penyimpanan hasil inference | Catatan: clustering belum menulis hasil ke Influx |

### Minggu 2 — Audit 12 kebutuhan
*9 Maret – 13 Maret 2026*

| Tanggal | Hari | Kegiatan | Keluaran |
|---|---|---|---|
| 09/03 | Senin | Menyusun daftar kebutuhan dan memeriksa poin 1–3 (dataset, entitas, task type) | Temuan: model anomaly tidak punya baris ModelML |
| 10/03 | Selasa | Memeriksa poin 4–6 (stream WebSocket, worker background, polling fitur) | Temuan: polling sekuensial, satu tag gagal membuang seluruh tick |
| 11/03 | Rabu | Memeriksa poin 7–9 (maintainability, efisiensi, logging) | Temuan: log per-dataset tidak pernah benar-benar terpisah |
| 12/03 | Kamis | Memeriksa poin 10–12 (worker ke client, retrain, forecast kedaluwarsa) | Temuan: `retrain_retention` belum ada; retrain hanya di worker supervised |
| 13/03 | Jumat | Merangkum hasil audit dan menyusun prioritas kerja | Dokumen temuan + urutan pengerjaan |

### Minggu 3 — Penyegaran forecast
*16 Maret – 20 Maret 2026*

| Tanggal | Hari | Kegiatan | Keluaran |
|---|---|---|---|
| 16/03 | Senin | Menelusuri `is_expired` dan menghitung ulang umur forecast | Rasio horizon terpakai sebagai dasar penyegaran |
| 17/03 | Selasa | Menyamakan gerbang `is_expired` dan `finetune` pada satu rasio | Menghilangkan finetune yang menolak jalan tapi terus dipanggil |
| 18/03 | Rabu | Menulis test `elapsed_horizon_ratio` dan batas horizon nol | 18 test kunci perilaku anchor forecast |
| 19/03 | Kamis | Menangani pull kosong pada `finetune` agar tidak diam-diam `return` | Kegagalan bisa dihitung pemanggil |
| 20/03 | Jumat | Meninjau ulang: mengganti pendekatan jadi degradasi bertahap | Keputusan: sajikan data terakhir, umumkan setelah 3 kegagalan |

### Minggu 4 — Degradasi gangguan SmartLink
*23 Maret – 27 Maret 2026*

| Tanggal | Hari | Kegiatan | Keluaran |
|---|---|---|---|
| 23/03 | Senin | Merancang kebijakan 3 kegagalan beruntun | Spesifikasi frame error + pemulihan |
| 24/03 | Selasa | Menulis `_on_poll_failed`, `_on_poll_ok`, dan backoff saat terdegradasi | Stream tidak lagi menembak SL tiap 10 detik saat gangguan |
| 25/03 | Rabu | Menyambungkan ke loop inference, siklus forecast, dan `get_actual` | Tiga jalur poll memakai kebijakan yang sama |
| 26/03 | Kamis | Memperbaiki `get_last_prediction` yang mengabaikan `max_age` | Client sekunder tidak lagi menerima prediksi basi tanpa batas |
| 27/03 | Jumat | Menulis test degradasi bertahap | 18 test: ambang, sekali umum, pemulihan, penanda basi |

## Restrukturisasi lapisan service

### Minggu 5 — Persiapan refactor
*30 Maret – 3 April 2026*

| Tanggal | Hari | Kegiatan | Keluaran |
|---|---|---|---|
| 30/03 | Senin | Menginventaris ukuran file dan peta ketergantungan antar modul | streamer.py 1103 baris, timeseries.py 439 baris |
| 31/03 | Selasa | Memutuskan struktur target: satu file per task type | Keputusan arsitektur + alasan |
| 01/04 | Rabu | Memutuskan penanganan worker pm2 saat file berpindah | Keputusan: pindah bersih, task dibuat ulang |
| 02/04 | Kamis | Menyusun rencana lima langkah beserta risikonya | Rencana + daftar test yang akan terdampak |
| 03/04 | Jumat | Memastikan seluruh test hijau sebagai jaring pengaman | Basis pembanding sebelum perubahan |

### Minggu 6 — Memindahkan streamer
*6 April – 10 April 2026*

| Tanggal | Hari | Kegiatan | Keluaran |
|---|---|---|---|
| 06/04 | Senin | Memisahkan `StreamState` dan pondasi `BaseStreamer` | state.py, base.py |
| 07/04 | Selasa | Memisahkan loop inference dan loop forecast | inference.py, forecast.py |
| 08/04 | Rabu | Memisahkan stream log berbasis watchdog | logs.py |
| 09/04 | Kamis | Menyusun manager lifecycle dan titik rakit paket | manager.py, __init__.py |
| 10/04 | Jumat | Memindahkan test dan menghapus `routes/streamer.py` | 7 file, perilaku tidak berubah |

### Minggu 7 — Memecah inference
*13 April – 17 April 2026*

| Tanggal | Hari | Kegiatan | Keluaran |
|---|---|---|---|
| 13/04 | Senin | Memisahkan jalur supervised | supervised.py + entry worker |
| 14/04 | Selasa | Memisahkan jalur anomaly dan memindahkan `auto_anomaly` | anomaly.py; model.py kembali murni lifecycle |
| 15/04 | Rabu | Membuat dispatcher lintas task | dispatch.py |
| 16/04 | Kamis | Mengangkat kerangka loop worker yang dipakai bersama | worker.py: `influx_write_loop`, `run_from_argv` |
| 17/04 | Jumat | Memecah test dan mengarahkan `get_script_path` | `_MAP_FUNC_AUTO` dan gerbang task type tidak diperlukan lagi |

### Minggu 8 — Memecah timeseries & clustering
*20 April – 24 April 2026*

| Tanggal | Hari | Kegiatan | Keluaran |
|---|---|---|---|
| 20/04 | Senin | Memisahkan perhitungan umur forecast dan read path | horizon.py, forecast.py |
| 21/04 | Selasa | Memisahkan back-fill nilai aktual ke InfluxDB | actuals.py |
| 22/04 | Rabu | Memisahkan `retrain` dan `finetune` | retrain.py |
| 23/04 | Kamis | Menyusun entry worker dan facade paket | worker.py, __init__.py |
| 24/04 | Jumat | Melebur `helpers.py` ke `clustering.py` | Menghapus lapisan re-export yang menyesatkan |

### Minggu 9 — Entry worker
*27 April – 1 Mei 2026*

| Tanggal | Hari | Kegiatan | Keluaran |
|---|---|---|---|
| 27/04 | Senin | Mengarahkan `get_script_path` ke entry per task | Empat entry, masing-masing berdiri sendiri |
| 28/04 | Selasa | Menghapus argv `task_type` yang tidak diperlukan lagi | Identitas worker = skrip yang dijalankan |
| 29/04 | Rabu | Menulis ulang `test_workers` mengikuti kontrak baru | Termasuk test bahwa tiap entry benar-benar ada |
| 30/04 | Kamis | Menguji keempat entry dari direktori kerja lain | Bootstrap `sys.path` terbukti benar |
| 01/05 | Jumat | Menyusun prosedur pembuatan ulang task pm2 | Langkah deploy + urutan aman |

## Logging, observability, dan pembersihan

### Minggu 10 — Diagnosis logging
*4 Mei – 8 Mei 2026*

| Tanggal | Hari | Kegiatan | Keluaran |
|---|---|---|---|
| 04/05 | Senin | Menelusuri perilaku `Logger()` dan lokasi file log | Log tersebar di tiga tempat berbeda |
| 05/05 | Selasa | Membuktikan `logger.remove()` loguru bersifat global | Uji: file dataset A kosong, isinya masuk ke file B |
| 06/05 | Rabu | Merancang tata letak folder per task dan per dataset | logs/<Task>/<dataset>/{main,worker}.log |
| 07/05 | Kamis | Memutuskan pemetaan Verbose ke sink dan sumber retensi | SILENT/NORMAL/DEBUG + `LOG_RETENTION_DAYS` |
| 08/05 | Jumat | Memutuskan penempatan anomaly dan perilaku SILENT | Anomaly ikut folder induk; ERROR tetap tercatat |

### Minggu 11 — Membangun LogManager
*11 Mei – 15 Mei 2026*

| Tanggal | Hari | Kegiatan | Keluaran |
|---|---|---|---|
| 11/05 | Senin | Menulis `LogManager` dengan sink berbasis filter | Sink didaftarkan sekali, tidak pernah dicabut |
| 12/05 | Selasa | Membuat scope relatif dan proxy logger global | Logger tetap hidup walau sink didaftarkan ulang |
| 13/05 | Rabu | Menulis test isolasi sink | Mengunci bug loguru yang lama |
| 14/05 | Kamis | Menulis test mode Verbose dan sumber retensi | 17 test |
| 15/05 | Jumat | Menambah `delay=True` dan fixture agar test tidak mencemari repo | Tidak ada folder log kosong di mode SILENT |

### Minggu 12 — Menyambungkan logging
*18 Mei – 22 Mei 2026*

| Tanggal | Hari | Kegiatan | Keluaran |
|---|---|---|---|
| 18/05 | Senin | Mengalihkan worker ke channel `worker` | Riwayat proses pm2 punya file sendiri |
| 19/05 | Selasa | Menambahkan middleware pencatat request | Seluruh route tercatat otomatis, bukan hanya dua |
| 20/05 | Rabu | Mengarahkan `read_logs` dan stream log ke tata letak baru | Penulis dan pembaca memakai satu sumber path |
| 21/05 | Kamis | Menambah parameter `channel` pada WebSocket log | Client bisa menonton log worker |
| 22/05 | Jumat | Menyisipkan `[dataset]` ke format baris dan menyesuaikan parser | File log lama tetap terbaca |

### Minggu 13 — Frontend log viewer
*25 Mei – 29 Mei 2026*

| Tanggal | Hari | Kegiatan | Keluaran |
|---|---|---|---|
| 25/05 | Senin | Menambahkan pemilih channel Process/Worker | Proses background akhirnya terlihat dari UI |
| 26/05 | Selasa | Memperbaiki tombol Pause yang menutup koneksi | Baris selama dijeda tidak lagi hilang |
| 27/05 | Rabu | Menambahkan reconnect backoff dan mengganti riwayat awal | Tidak ada lagi duplikasi ratusan baris |
| 28/05 | Kamis | Menampilkan field dataset dan menyiapkan halaman log global | Log sistem bisa dibaca terpisah |
| 29/05 | Jumat | Menguji jalur ujung-ke-ujung penulis → pembaca → penyaji | 12 test jalur log |

### Minggu 14 — Kegagalan permanen & pembersihan
*1 Juni – 5 Juni 2026*

| Tanggal | Hari | Kegiatan | Keluaran |
|---|---|---|---|
| 01/06 | Senin | Menambahkan frame fatal saat model anomaly tidak ada | Client tahu sebabnya, bukan sekadar terputus |
| 02/06 | Selasa | Menangani frame fatal di halaman anomaly | Berhenti menyambung ulang, tampilkan cara memperbaiki |
| 03/06 | Rabu | Menulis hook `useRealtimeStream` | Satu tempat untuk koneksi, reconnect, dan frame kontrol |
| 04/06 | Kamis | Menyeragamkan empat komponen realtime memakai hook | Menghapus empat salinan logika koneksi |
| 05/06 | Jumat | Menghapus modul mati dan memperbaiki tombol yang digerakkan data mock | Rebuild/Retrain memakai status dataset sungguhan |

## Chat Assistant & arsitektur RAG

### Minggu 15 — Persona LLM & streaming chat
*8 Juni – 12 Juni 2026*

| Tanggal | Hari | Kegiatan | Keluaran |
|---|---|---|---|
| 08/06 | Senin | Menelusuri struktur `agent-service`, terutama `llm_client.py` yang terhubung ke llama-server | Catatan kebutuhan: chatbot khusus domain BMS/SCADA, bukan asisten umum |
| 09/06 | Selasa | Menyusun system prompt peran BMS Engineer dan menambah `chat_stream()` berbasis SSE | Streaming response dari llama-server |
| 10/06 | Rabu | Menyambungkan `chat_stream` ke endpoint `/chat/stream` dan menguji manual | Bug parsing token `[DONE]` pada SSE ditemukan dan diperbaiki |
| 11/06 | Kamis | Membandingkan fine-tune dan RAG untuk pemahaman domain BMS/SCADA/DCIM | Keputusan: RAG dulu — dataset terbatas, model Qwen2.5-7B terkuantisasi |
| 12/06 | Jumat | Menyusun draf arsitektur RAG (retrieval + reranker + llama.cpp) | Kebutuhan dipecah jadi tiga fitur: Q&A SmartLink, BMS Support, Point Diagnosis |

### Minggu 16 — Arsitektur 3-lane & lazy init
*15 Juni – 19 Juni 2026*

| Tanggal | Hari | Kegiatan | Keluaran |
|---|---|---|---|
| 15/06 | Senin | Memisahkan tiga fitur menjadi lane masing-masing | `intent_router.py`: deteksi bahasa + klasifikasi intent |
| 16/06 | Selasa | Membangun orkestrator pemilih lane dan lane diagnosis | `pipeline.py`, `diagnostic.py` memakai ulang fungsi diagnose dari modscan |
| 17/06 | Rabu | Merombak `rag.py` ke pola lazy init agar model berat tidak dimuat saat startup | `embedding.py` terpisah + cache TTL in-memory di `cache.py` |
| 18/06 | Kamis | Merombak `incident_rag.py` untuk lane BMS Support | Tersambung ke data insiden yang sudah tersedia |
| 19/06 | Jumat | Menyiapkan kerangka pengujian pytest | `pytest.ini`, `conftest.py`, test intent router dan lane diagnosis |

### Minggu 17 — Pengujian & benchmark per fitur
*22 Juni – 26 Juni 2026*

| Tanggal | Hari | Kegiatan | Keluaran |
|---|---|---|---|
| 22/06 | Senin | Melengkapi test lane knowledge dan lane incident dengan mock | Test berjalan cepat tanpa memuat model asli |
| 23/06 | Selasa | Menyusun benchmark satu kali jalan per fitur | `test_benchmark.py`, opt-in agar tidak memakan sumber daya |
| 24/06 | Rabu | Menelusuri kegagalan CUDA out of memory saat benchmark | Penyebab: reranker dimuat dua kali, berebut VRAM dengan llama-server |
| 25/06 | Kamis | Menjadikan reranker satu instance bersama yang di-pin ke CPU | `reranker.py`; konflik versi transformers diselesaikan dengan downgrade |
| 26/06 | Jumat | Menjalankan ulang seluruh benchmark | Tiga fitur lolos; path berkas CSV insiden yang salah ikut diperbaiki |

### Minggu 18 — Skill bypass & optimasi latensi
*29 Juni – 3 Juli 2026*

| Tanggal | Hari | Kegiatan | Keluaran |
|---|---|---|---|
| 29/06 | Senin | Menambahkan field `skill` pada `ChatRequest` | Frontend bisa melewati klasifikasi intent otomatis |
| 30/06 | Selasa | Memperbaiki lane incident yang meminta API key OpenAI | Penyebab: `embed_model` hanya terisi bila lane knowledge jalan lebih dulu |
| 01/07 | Rabu | Menelusuri laporan respons lambat pada sintesis RAG | Mode `refine` (sampai 5 panggilan LLM) diganti `compact` (1–2 panggilan) |
| 02/07 | Kamis | Menambah dokumen ke knowledge base dan menelusuri mengapa tidak terbaca | Indeks vektor tidak ikut diperbarui; dibuat auto-rebuild lewat fingerprint berkas |
| 03/07 | Jumat | Menghapus praterjemahan Indonesia→Inggris, membetulkan `context_window`, dan ekstraksi nama point | Seluruh perubahan di-commit dan didorong ke repositori |

## Analitik clustering

### Minggu 19 — Diagnosis halaman riwayat
*6 Juli – 10 Juli 2026*

| Tanggal | Hari | Kegiatan | Keluaran |
|---|---|---|---|
| 06/07 | Senin | Mengukur payload halaman riwayat clustering | 605 KB untuk 2.665 baris |
| 07/07 | Selasa | Menelusuri penyebab lag di sisi peramban | Tabel merender ±88.000 sel DOM |
| 08/07 | Rabu | Menemukan bahwa 'riwayat' sebenarnya data latih | Hasil inference clustering tidak tersimpan di mana pun |
| 09/07 | Kamis | Merancang endpoint ringkasan per rentang tanggal | Bentuk respons + batas tanggal tersedia |
| 10/07 | Jumat | Memutuskan sajian yang benar-benar informatif | Katalog metrik generik, tidak bergantung nama fitur |

### Minggu 20 — Komposisi & episode
*13 Juli – 17 Juli 2026*

| Tanggal | Hari | Kegiatan | Keluaran |
|---|---|---|---|
| 13/07 | Senin | Menulis perhitungan komposisi dan keseimbangan cluster | Entropi ternormalisasi + penanda cluster mikro |
| 14/07 | Selasa | Menulis perhitungan episode dan ringkasannya | 2.665 baris menyusut jadi 47 episode |
| 15/07 | Rabu | Menambahkan deteksi kolom konstan dan interval data | 19 dari 30 fitur ternyata tidak pernah berubah |
| 16/07 | Kamis | Menulis test perhitungan | 19 test, termasuk rentang kosong |
| 17/07 | Jumat | Menambahkan endpoint `/clusters/report` | 11 KB, 34 ms — turun 55× |

### Minggu 21 — Geometri cluster
*20 Juli – 24 Juli 2026*

| Tanggal | Hari | Kegiatan | Keluaran |
|---|---|---|---|
| 20/07 | Senin | Menulis normalisasi fitur yang membuang kolom konstan | Prasyarat semua perhitungan jarak |
| 21/07 | Selasa | Menghitung fitur pembeda tiap cluster | Diurutkan dalam satuan standar deviasi |
| 22/07 | Rabu | Menghitung kekompakan tiap cluster | Menyingkap cluster yang sebenarnya kumpulan pencilan |
| 23/07 | Kamis | Menghitung keterpisahan antar pasangan cluster | Rasio tak bersatuan + penanda tumpang tindih |
| 24/07 | Jumat | Menulis test geometri | 19 test |

### Minggu 22 — Pencilan, mutu, dan pola waktu
*27 Juli – 31 Juli 2026*

| Tanggal | Hari | Kegiatan | Keluaran |
|---|---|---|---|
| 27/07 | Senin | Menghitung titik terjauh dari centroid beserta penyebabnya | Menemukan titik 21× radius yang berlabel normal |
| 28/07 | Selasa | Menghitung kesepakatan antar algoritma | ARI mengungkap dua model bercerita berbeda |
| 29/07 | Rabu | Menghitung pola jam dan matriks perpindahan | Pola harian terbaca tanpa menggambar titik |
| 30/07 | Kamis | Menghitung pergeseran komposisi dan menyertakan metrik mutu | Silhouette dibaca dari hasil training |
| 31/07 | Jumat | Mengoptimalkan perhitungan pencilan | 1650 ms → 80 ms, hasil identik |

### Minggu 23 — Frontend riwayat clustering
*3 Agustus – 7 Agustus 2026*

| Tanggal | Hari | Kegiatan | Keluaran |
|---|---|---|---|
| 03/08 | Senin | Memvalidasi palet warna untuk mode terang dan gelap | Lolos pemisahan buta warna dan kontras |
| 04/08 | Selasa | Menyusun kerangka halaman dan kontrol rentang tanggal | Filter dalam satu baris di atas panel |
| 05/08 | Rabu | Membangun kartu profil cluster dan strip timeline | Timeline digambar dari episode, bukan titik |
| 06/08 | Kamis | Membangun panel pola harian, perpindahan, dan tabel pencilan | Angka utama didahulukan |
| 07/08 | Jumat | Memverifikasi seluruh field respons dan build produksi | 40 field cocok, build lolos |

## Konsistensi realtime clustering

### Minggu 24 — Temuan label tidak stabil
*10 Agustus – 14 Agustus 2026*

| Tanggal | Hari | Kegiatan | Keluaran |
|---|---|---|---|
| 10/08 | Senin | Menelusuri jalur clustering realtime dari stream sampai core | Model dilatih ulang setiap tick |
| 11/08 | Selasa | Mengukur kestabilan label pada data yang identik | ARI 1.000 tapi label cocok 0% — penomoran diacak |
| 12/08 | Rabu | Menguji `predict_model` pada model tersimpan | kmeans stabil; spectral tidak punya predict |
| 13/08 | Kamis | Membuat prototipe penetapan lewat centroid terdekat | Bekerja di ruang pipeline model, 8 ms |
| 14/08 | Jumat | Merancang kontrak `ClusterAssignRequest` | Core tetap murni, acuan disuplai service |

### Minggu 25 — Penetapan stabil
*17 Agustus – 21 Agustus 2026*

| Tanggal | Hari | Kegiatan | Keluaran |
|---|---|---|---|
| 17/08 | Senin | Menulis `Unsupervised.assign` | Memakai model tersimpan, tanpa pelatihan ulang |
| 18/08 | Selasa | Menulis jalur cadangan untuk algoritma tanpa predict | Didokumentasikan sebagai pendekatan |
| 19/08 | Rabu | Menyambungkan service dan menghapus live buffer | Satu penulisan disk per tick hilang |
| 20/08 | Kamis | Menulis test penetapan dan jalur cadangan | 16 test, termasuk larangan melatih ulang |
| 21/08 | Jumat | Memverifikasi dengan model dan data sungguhan | 5 panggilan berturut-turut: label identik |

### Minggu 26 — Konteks realtime & UI
*24 Agustus – 28 Agustus 2026*

| Tanggal | Hari | Kegiatan | Keluaran |
|---|---|---|---|
| 24/08 | Senin | Menghitung jarak titik ke centroid saat penetapan | Statistik cluster dihitung sekali per algoritma |
| 25/08 | Selasa | Menambahkan pangsa historis dan durasi episode berjalan | Durasi disertai penanda kepastian |
| 26/08 | Rabu | Menulis test konteks realtime | 12 test: ambang, pangsa, episode |
| 27/08 | Kamis | Menulis ulang tampilan realtime clustering | Keadaan sekarang didahulukan, sebaran jadi pelengkap |
| 28/08 | Jumat | Menyatukan palet warna dua halaman dan verifikasi akhir | Cluster yang sama sewarna di riwayat dan realtime |

---

## Ringkasan hasil

| Bidang | Sebelum | Sesudah |
|---|---|---|
| Berkas terbesar di lapisan service | `streamer.py` 1103 baris | terbesar 441 baris |
| Payload halaman riwayat clustering | 605 KB | 11–29 KB |
| Waktu penyusunan laporan clustering | — | 80 ms |
| Label cluster antar tick | penomoran diacak (cocok 0%) | tetap |
| Log per dataset | tercampur satu file | terpisah per task, dataset, dan channel |
| Test otomatis | 117 | 272 |
| Pipeline chat assistant | satu jalur, klasifikasi kata kunci | tiga lane + bypass lewat field `skill` |
| Panggilan LLM per jawaban RAG | sampai 5 (mode refine) | 1–2 (mode compact) |
| Indeks vektor saat dokumen berubah | tidak diperbarui | auto-rebuild lewat fingerprint berkas |

## Yang belum selesai

- `retrain_retention` pada Dataset dan penjadwal retrain terpusat (retrain masih hanya berjalan di worker supervised)
- Model anomaly belum punya baris `ModelML`, sehingga tidak punya metrik dan tidak muncul di daftar model
- Empat bug tercatat: `preprocessing` tak terdefinisi di `pulling()`, `miss_val_handling_df` dapat mengembalikan `None`,
- Polling fitur masih berurutan; cache realtime baru dipakai di endpoint utilitas
- SQLite belum memakai mode WAL padahal ditulis banyak proses worker
