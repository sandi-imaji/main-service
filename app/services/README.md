# `app/services` — Orchestration Layer

Lapisan **orkestrasi**. Semua hal yang **bukan** komputasi ML murni tinggal di
sini: menarik data (`app.pull`), membaca/menulis database, menulis prediksi ke
InfluxDB, mengelola status dataset/model, penamaan cluster, reduksi PCA, dan
loop worker (pm2).

## Posisi dalam arsitektur

```
app/routes  (HTTP / WebSocket)  ──►  app/services  ──►  app/core (murni)
                                          │  ▲
                                          ▼  │ contracts (request/result)
                       app.database.DB (Dataset/ModelML) · app.pull · historicalDB (Influx)
```

Pola kerjanya selalu sama:

1. Ambil `Dataset` dari DB, tarik data live / buka `data.csv`.
2. Bangun **contract** lewat helper di `Dataset` (`to_predict_request`,
   `to_forecast_request`, `to_cluster_request`, `to_train_request`, …).
3. Panggil core: `result = core.method(request, logger)`.
4. Bungkus hasil jadi *result schema* (`results.py`), lalu **persist**:
   simpan ke DB, tulis ke InfluxDB, atau serialisasi ke HTTP.

Core tidak pernah tahu langkah 1, 2b, dan 4 — itu sepenuhnya di lapisan ini.

---

## Peta file

| File | Peran | Dipakai oleh |
|------|-------|--------------|
| `train.py` | Orkestrasi **training** & **finetune** untuk 4 tipe (via contracts). | routes/models, tasks |
| `inference.py` | Dispatcher **auto-inference** + worker entry Supervised/Anomaly. | routes/models, routes/streamer, worker pm2 |
| `timeseries.py` | Orkestrasi inferensi **TimeSeries** + worker entry. | routes/models, streamer, tasks, worker pm2 |
| `clustering.py` | Orkestrasi inferensi **Clustering** + worker entry. | routes/models, streamer, worker pm2 |
| `model.py` | Lifecycle model: integrity check, inferensi eksplisit, ganti top-model, cleanup, stats dashboard. | routes/models, routes/utils |
| `dataset.py` | Operasi tipis di atas `Dataset`: create/delete, sample, describe, PCA. | routes/datasets |
| `results.py` | **Result DTO** (output layer): schema hasil + `write_to_influx`. | semua service inferensi |
| `helpers.py` | Helper clustering storage-facing: naming, PCA transform, baca `clusters.csv`. | clustering, routes/utils, routes/datasets |
| `validators.py` | `split_data` (train/test split + secure path). | routes |
| `tasks.py` | **Background jobs** (FastAPI `BackgroundTasks`): tiap job buka sesi DB sendiri. | routes/models |

---

## Alur per tipe task

### Training — `train.py`

```
find_top_models(dataset, n_top, db, logger)
  └─ core.compare_models(dataset.to_train_request(n_top), logger)
       └─ _persist_trained_model(...)  → ModelML row + set top_model + status
```

- `build_train_request(task_type, **kwargs)` memilih subclass request yang tepat.
- `train_model(...)` untuk latih satu algoritma.
- `finetune(dataset, logger)` untuk refit model existing pada data baru. Hanya
  core yang punya `retrain_models` (Supervised/TimeSeries) yang didukung; core
  lain menjalankan finetune di worker-nya sendiri. Assembly window data baru ada
  di `_assemble_finetune_dataframe`.

### Inference dispatcher — `inference.py`

```
auto_inference(dataset, logger)                       # SATU pintu masuk
  ├─ TimeSeries  → timeseries_service.auto_inference
  ├─ Clustering  → clustering_service.auto_inference
  ├─ Anomaly     → auto_anomaly_detection   (lokal)
  └─ Supervised  → auto_predictions          (lokal)
```

`auto_inference_write_loop` = loop worker Supervised/Anomaly: (finetune bila
jatuh tempo) → inferensi → tulis Influx → tidur `interval` menit.
`main()` = entry pm2 (dijalankan langsung, nama dataset lewat argv).

### TimeSeries inferensi — `timeseries.py`

| Fungsi | Guna |
|--------|------|
| `is_expired(dataset, logger)` | Apakah forecast terakhir sudah lewat. |
| `forecast(dataset, logger)` | Turunkan timestamp masa depan + `TimeSeries.forecast` (core). |
| `auto_inference(dataset, logger)` | Bila expired: back-fill actual → retrain → forecast. |
| `get_actual_save` / `get_actual_save_all` | Isi nilai aktual dari Influx ke record inferensi lama. |
| `retrain(dataset, logger)` | Tarik data baru + `TimeSeries.retrain_models` (core). Sesi DB sendiri. |
| `finetune(dataset_name)` | Gabung data teraktualisasi, refit semua model. Sync & blocking → dipanggil via threadpool (BackgroundTasks) / `asyncio.to_thread` (streamer). |
| `auto_inference_write_loop` / `main` | Worker loop + entry pm2. |

### Clustering inferensi — `clustering.py`

```
inference(dataset, features, logger)
  ├─ context = data.csv (training STATIS, read-only) + live_buffer (rolling, berbatas)
  ├─ Unsupervised.predict(dataset.to_cluster_request(context), logger)   # core, label mentah
  ├─ terapkan naming  (get_naming_clusters → helpers.py)
  ├─ reduksi PCA 2D bila >2 fitur  (transform_pca → helpers.py)
  └─ tulis results/live_buffer.csv saja (data.csv & clusters.csv tak disentuh)
```

> **Pemisahan training vs buffer live.** `data.csv` (data training) & `clusters.csv`
> (scatter hasil train) bersifat **statis** — inference hanya membacanya. Titik live
> masuk ke `results/live_buffer.csv`, sebuah rolling window **berbatas**
> (`LIVE_BUFFER_MAXLEN`) di file terpisah. Jadi baseline training tak pernah
> tergerus, tulis-ulang per-tick jadi murah (hanya buffer), dan bila buffer hilang
> (mis. worker restart) inference tetap valid — training set sendiri sudah cukup
> sebagai referensi. Adaptasi drift, bila diinginkan, lewat retrain terjadwal.

`auto_inference` menarik fitur live lalu memanggil `inference`.
`auto_inference_write_loop` / `main` = worker loop + entry pm2.

---

## `results.py` — Result DTO (output layer)

Schema pydantic yang membungkus hasil core lalu punya method persist. Sengaja di
luar `app/core` supaya core bersih dari urusan output.

| Schema | Factory | `write_to_influx` |
|--------|---------|-------------------|
| `SupervisedResultSchema` | `from_result`, `failed` | ✔ (results + features + actual) |
| `AnomalyResultSchema` | `from_result`, `invalid` | ✔ (is_anomaly + score) |
| `TimeSeriesResultSchema` | `from_forecast`, `invalid` | ✔ (satu titik per timestamp) |
| `UnsupervisedResultSchema` | — | ✘ (clustering persist via CSV) |

Semua punya flag `is_valid` — dipakai worker loop supaya prediksi kosong tidak
ikut ditulis ke Influx.

---

## `helpers.py` — clustering storage-facing

Bagian clustering yang menyentuh storage (bukan komputasi), jadi terpisah dari
core:

- `transform_pca(dataset_name, data, logger)` — reduksi 2D pakai `pca.joblib`
  yang disimpan saat training.
- `get_clusters` / `get_cluster_unique` — baca `results/clusters.csv`.
- `get_naming_clusters` / `get_naming_cluster` / `save_naming_clusters` /
  `update_naming_clusters` — mapping `{algoritma: {label_mentah: nama}}` di
  `results/naming_clusters.json`.
- `naming_clusters` (validasi + map) / `rename_clusters` (map kolom DataFrame).

---

## `model.py` & `dataset.py`

**`model.py`** — lifecycle & inferensi eksplisit:
- `inference(payload, db)` — endpoint payload eksplisit. Clustering → delegasi ke
  `clustering_service.inference`; TimeSeries ditolak (arahkan ke `/forecast`);
  Supervised → `core.predict`.
- `check_dataset_pretrained` / `check_integrity_model`, `change_top_model`,
  `clean_models`, `clean_results`, `auto_anomaly`.
- `get_stats` / `_build_stats` — statistik dashboard.

**`dataset.py`** — `create_dataset`, `delete_dataset(s)`, `sample_dataset`,
`describe_dataset`, `reduce_dimensions`.

---

## `tasks.py` — background jobs

> FastAPI menutup sesi DB request begitu response terkirim, jadi job
> `BackgroundTasks` **tidak boleh** memakai ulang sesi itu.

Tiap job membuka sesinya sendiri (`get_db_session()`) dan mengambil ulang
dataset **by name**: `initialize_dataset`, `find_top_models`, `train_one_model`,
`run_auto_anomaly`, `refresh_forecast_actuals`, `pull_dataset`.

---

## Worker entry (pm2)

`WorkerManager.get_script_path(task_type)` mengarahkan tiap worker ke entry
**service-layer**-nya (core sekarang murni, tanpa `main()`):

| Task type | Script worker |
|-----------|---------------|
| Supervised (Classification/Regression) | `services/inference.py` |
| Anomaly | `services/inference.py` |
| TimeSeries | `services/timeseries.py` |
| Clustering | `services/clustering.py` |

Tiap file worker punya blok bootstrap `sys.path` di dalam
`if __name__ == "__main__":` **sebelum** impor `app.*`, karena pm2 menjalankan
file itu langsung dengan nama dataset sebagai argv.

---

## Konvensi & jebakan (penting)

- **`dataset.meta` itu OBJEK `MetaDataset`, bukan dict.** Akses via atribut
  (`dataset.meta.current_dt`), jangan `.get()` / `["..."]`.
- **Kolom JSON (`meta`) tidak melacak mutasi in-place.** Untuk menyimpan
  perubahan, reassign objek baru: `dataset.meta = dataset.meta.model_copy(update={...})`.
- Field ukuran model adalah `sizeof` (bukan `size_of` — yang lama diam-diam
  di-drop pydantic).
- Jangan panggil core dengan `Dataset` — selalu lewat contract
  (`dataset.to_*_request(...)`).

## Testing

```bash
.venv/bin/python -m pytest tests/services -q
```

Memakai `FakeDataset` + monkeypatch (core/Influx/pm2 semua di-mock). Lihat
`tests/services/test_*.py` — mencakup dispatcher `auto_inference`, orkestrasi
clustering, result DTO, dan stats.
