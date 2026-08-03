# `app/core` — Pure ML Compute

Lapisan **komputasi ML murni**. Semua yang berhubungan dengan PyCaret (setup,
train, predict, forecast, cluster) tinggal di sini — dan **hanya** itu.

## Prinsip utama

> Core tidak tahu apa-apa soal `Dataset`, database, InfluxDB, HTTP, atau storage.

Core hanya bicara lewat **contracts** (dataclass biasa di `contracts.py`).
Ia menerima sebuah *request object*, menjalankan komputasi, lalu mengembalikan
sebuah *result object*. Siapa yang memanggil, dari mana datanya, dan ke mana
hasilnya disimpan — itu urusan `app/services`.

Konsekuensinya:

- Bisa di-`import` dan diuji terisolasi (tidak menarik DB/FastAPI).
- Mudah di-mock saat testing (cukup mock modul PyCaret-nya).
- Tidak ada efek samping tersembunyi selain menyimpan artefak model ke `out_dir`
  yang **sudah ditentukan pemanggil** (bukan diturunkan dari `Dataset`).

```
app/routes  (HTTP)
     │
     ▼
app/services  (orkestrasi: pull data, persist DB, tulis Influx, worker loop)
     │   membangun request  ──►  core.method(request, logger)  ──►  result
     ▼
app/core  (komputasi murni)  ◄── modul ini
```

---

## Isi folder

| File | Isi | Keterangan |
|------|-----|-----------|
| `contracts.py` | Request/Result dataclasses | Jembatan data core ⇄ services. Tanpa impor `app.database`/`fastapi`. |
| `base.py` | `BaseMLCore`, `ALGO_LIST` | Fasilitas bersama: model cache (LRU) + `ALGO_LIST` dari `algorithm_list.json`. Tanpa impor `app.database`/`fastapi`. |
| `supervised.py` | `Supervised` | Classification & Regression. |
| `unsupervised.py` | `Unsupervised` | Clustering (transductive — lihat di bawah). |
| `time_series.py` | `TimeSeries` | Forecasting. |
| `anomaly.py` | `Anomaly` | Anomaly detection (satu model per dataset). |
| `__init__.py` | Re-export 4 kelas core | `from app.core import Supervised, Unsupervised, TimeSeries, Anomaly`. |

---

## Contracts (`contracts.py`)

Semua contract adalah `@dataclass(frozen=True)` (immutable, tanpa validasi
pydantic → nol overhead). Request training memakai `kw_only=True` agar pewarisan
frozen-dataclass tetap bersih.

### Training

```
TrainRequest (base, kw_only)
├── df: DataFrame            preprocessing: dict     out_dir: Path
├── task: str               n_top: int = 1
│
├── SupervisedTrainRequest   + target: str
├── ClusteringTrainRequest   + n_clusters: int
├── AnomalyTrainRequest      + fraction: float, algorithm: str
└── TimeSeriesTrainRequest   + fh: int   (forecast horizon)
```

`out_dir` = direktori tempat artefak model disimpan (biasanya
`<storage>/top_model`). Core hanya menulis file model + CSV ke situ, tidak tahu
itu "dataset yang mana".

### Output training

```
TrainedModel(algorithm, path, evaluation, size)
  .from_saved(algorithm, path, evaluation)   # size dihitung dari `{path}.pkl`
```

> **Konvensi penting:** PyCaret menyimpan `save_model(m, path)` sebagai
> `path.pkl`. `TrainedModel.from_saved` memusatkan konvensi `+.pkl` itu.

### Inference

| Request | Field | Result |
|---------|-------|--------|
| `PredictRequest` | `features`, `models=[(algo, path)]`, `task` | `PredictResult(features, predictions)` |
| `ForecastRequest` | `models`, `fh`, `task` | `ForecastResult(forecast)` |
| `ClusterRequest` | `df`, `algorithms`, `n_clusters`, `preprocessing`, `task` | `ClusterResult(clusters, assignments)` |
| (anomaly pakai `PredictRequest`) | | `AnomalyResult(features, is_anomaly, anomaly_score)` |

`PredictRequest` dipakai bersama oleh Supervised **dan** Anomaly (anomaly cuma
mengisi satu model di path `<storage>/anomaly`).

---

## Anatomi tiap core

### `Supervised` (Classification / Regression)

| Method | Contract in → out | Deskripsi |
|--------|-------------------|-----------|
| `predict(req, logger)` | `PredictRequest` → `PredictResult` | Inferensi tiap model (via cache), baca `prediction_label`. |
| `compare_models(req, logger)` | `SupervisedTrainRequest` → `List[TrainedModel]` | `compare_models(n_select=n_top)`, simpan top-N. |
| `train_one(req, algorithm, logger)` | `SupervisedTrainRequest` → `TrainedModel` | Latih satu algoritma. |
| `retrain_models(req, algorithms, logger)` | → `List[TrainedModel]` | Refit sekumpulan algoritma dalam satu setup (dipakai finetune). |

`module(task_type)` memilih `pycaret.classification` / `pycaret.regression`.

### `Unsupervised` (Clustering)

Clustering di sini **transductive**: PyCaret clustering tidak punya "predict
titik baru". Jadi saat inferensi, titik baru **ditambahkan** ke data training,
seluruhnya di-cluster ulang, lalu label baris terakhir dibaca.

| Method | Contract in → out | Deskripsi |
|--------|-------------------|-----------|
| `predict(req, logger)` | `ClusterRequest` → `ClusterResult` | `req.df` = training + titik baru. Balikan: label baris terakhir per algoritma + assignment penuh. |
| `compare_models(req, logger)` | `ClusteringTrainRequest` → `List[TrainedModel]` | Latih semua algoritma, ranking (Silhouette ↓, Calinski-Harabasz ↓, Davies-Bouldin ↑), simpan `pca.joblib` + `clusters.csv`. |
| `train_one(req, algorithm, logger)` | → `TrainedModel` | Latih satu algoritma clustering. |
| `is_same_cluster(clusters, name=None)` | | Util: apakah semua algoritma sepakat satu cluster. |

> Label **mentah** (angka). Penamaan manusiawi ("cold"/"hot"), reduksi PCA 2D,
> dan penulisan CSV hasil inferensi adalah tugas `services/clustering.py` +
> `services/helpers.py`.

### `TimeSeries` (Forecasting)

| Method | Contract in → out | Deskripsi |
|--------|-------------------|-----------|
| `forecast(req, logger)` | `ForecastRequest` → `ForecastResult` | Proyeksi tiap model `fh` langkah. **Tanpa timestamp** (itu ditempel service). |
| `compare_models(req, logger)` | `TimeSeriesTrainRequest` → `List[TrainedModel]` | `compare_models(include=TOP_ALGO)`. |
| `train_one` / `retrain_models` | | Sama pola dengan Supervised. |
| `get_cache_stats()` | | Statistik cache model TS. |

`TOP_ALGO` = daftar algoritma forecasting yang dibandingkan.

### `Anomaly`

| Method | Contract in → out | Deskripsi |
|--------|-------------------|-----------|
| `predict(req, logger)` | `PredictRequest` → `AnomalyResult` | Satu model (`req.models[0]`), baca kolom `Anomaly` + `Anomaly_Score`. |
| `train_one(req, algorithm, logger)` | `AnomalyTrainRequest` → `TrainedModel` | Satu model disimpan di `<storage>/anomaly`, + `anomaly.csv`. |

Anomaly menyimpan **satu** model per dataset (bukan per-algoritma di
`top_model/`), makanya path diturunkan dari `req.out_dir.parent`.

---

## `base.py` — fasilitas bersama

- **`BaseMLCore`** (ABC): `get_cache()` (wajib di-override), `get_cache_stats()`,
  `clear_cache()`, `_load_model_cached(mod, path, logger)` (LRU + thread-safe).
  Hanya plumbing cache — semua compute ada di subclass task.
- **`ALGO_LIST`**: dibaca sekali dari `algorithm_list.json` (path absolut
  `Config.dir`) saat import.

Tiap core punya cache singleton per-proses:
`get_supervised_cache()` / `get_unsupervised_cache()` / `get_timeseries_cache()` /
`get_anomaly_cache()` (di `app/utils/model_cache.py`).

---

## Aturan saat menambah / mengubah core

1. **Jangan** `import` apa pun dari `app.database`, `app.pull`,
   `app.database.historicalDB`, atau `fastapi` di file core.
2. Input & output **selalu** lewat contract. Kalau butuh data baru, tambahkan
   field ke contract-nya, jangan terima `Dataset`.
3. Menyimpan artefak? Tulis ke `req.out_dir` (atau `.parent`) — jangan susun
   sendiri path dari nama dataset.
4. Method harus `staticmethod`/`classmethod` yang bisa dipanggil tanpa state.
5. Tambahkan test di `tests/core/` dengan modul PyCaret di-mock.

## Testing

```bash
.venv/bin/python -m pytest tests/core -q
```

PyCaret di-mock (`monkeypatch.setattr("app.core.<mod>.mod", FakeMod)` atau
`...module`), jadi test jalan cepat tanpa training sungguhan. Lihat
`tests/core/test_*.py`.
