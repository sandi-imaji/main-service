"""
Clustering — orkestrasi lengkap satu task.

Komputasi ML-nya murni di `app.core.unsupervised` (`Unsupervised.assign`);
lapisan ini yang Dataset/storage-aware: menyiapkan acuan centroid, memberi nama
manusiawi pada cluster, reduksi PCA untuk visual 2D, ringkasan per rentang
tanggal, dan loop worker. `data.csv` / `results/clusters.csv` diperlakukan
read-only di sini — keduanya referensi statis hasil training.

Sekaligus entry point worker: pm2 menjalankan file ini langsung dengan nama
dataset sebagai argv — lihat `main()` di bawah.
"""
# Script-entry bootstrap: when pm2 runs this file directly the project root is
# not yet importable, so put it on sys.path BEFORE the `app.*` imports below.
if __name__ == "__main__":
  import sys
  from pathlib import Path
  _root = Path(__file__).resolve().parents[2]
  if str(_root) not in sys.path: sys.path.insert(0, str(_root))

import json
import os
import time
import traceback
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

from app.config import Config
from app.core.unsupervised import Unsupervised
from app.database.DB import Dataset
from app.helpers import DTEncoder
from app.pull import pull_realtime
from app.services import cluster_report
from app.services import worker
from app.services.results import UnsupervisedResultSchema

__all__ = [
    "inference", "auto_inference", "auto_inference_write_loop", "main",
    "get_clusters", "get_cluster_unique", "get_naming_clusters",
    "get_naming_cluster", "save_naming_clusters", "update_naming_clusters",
    "naming_clusters", "rename_clusters", "transform_pca", "report",
]


# --- Reduksi PCA -----------------------------------------------------------

def transform_pca(dataset_name: str, data: pd.DataFrame, logger) -> Optional[pd.DataFrame]:
  """Reduce features to 2D using the PCA object saved at train time."""
  fpath = Config.dir / "storages" / dataset_name / "pca.joblib"
  logger.info("transform features to n-2 dimensionality")
  if not fpath.exists():
    logger.warning(f"pca object is not found : {fpath}")
    return None
  pca_obj = joblib.load(fpath)
  return pd.DataFrame(pca_obj.transform(data), columns=["X1", "X2"])

# --- Baca hasil cluster & penamaannya (storage) ----------------------------

def get_clusters(dataset: Dataset, logger) -> Dict:
  """Read the persisted cluster assignments, apply naming, and add 2D coords."""
  if not dataset.models:
    raise ValueError(f"Dataset {dataset.name} has no trained models yet")

  fpath = Config.dir / "storages" / dataset.name / "results/clusters.csv"
  if not os.path.exists(fpath):
    raise FileNotFoundError(
        f"clusters.csv not found for dataset {dataset.name}. Please ensure training is complete.")

  algorithms = [a.algorithm for a in dataset.models]
  feature_names = dataset.features
  result = pd.read_csv(fpath)

  naming = get_naming_clusters(dataset.name)
  if naming:
    logger.info("renaming clusters ...")
    result = rename_clusters(result, naming)

  result_dict = result.to_dict(orient="list")
  result_dict["feature_names"] = feature_names
  result_dict["algorithms"] = algorithms

  if len(feature_names) > 2:
    feature_reduce = transform_pca(dataset.name, result[feature_names], logger=logger)
    if feature_reduce is None:
      return {}
    result_dict["X1"] = feature_reduce["X1"].values.tolist()
    result_dict["X2"] = feature_reduce["X2"].values.tolist()
  else:
    result_dict["X1"] = result_dict[feature_names[0]]
    result_dict["X2"] = result_dict[feature_names[1]]

  return result_dict

def report(dataset: Dataset, start=None, end=None) -> Dict:
  """Ringkasan hasil clustering untuk satu rentang tanggal.

  Bacanya dari `clusters.csv` yang sama dengan `get_clusters`, bedanya yang
  dikirim ringkasan — bukan 2.665 baris × 33 kolom nilai mentah. Rentang di luar
  data yang tersimpan tidak ditarik ulang dari SL; `available` memberi tahu
  frontend batas yang memang ada isinya.
  """
  fpath = Config.dir / "storages" / dataset.name / "results" / "clusters.csv"
  if not fpath.exists():
    raise FileNotFoundError(
        f"clusters.csv not found for dataset {dataset.name}. Please ensure training is complete.")

  df = pd.read_csv(fpath)
  df["dt"] = pd.to_datetime(df["dt"], utc=True)
  available = {"start": df["dt"].min().isoformat(), "end": df["dt"].max().isoformat()}

  window = df
  if start is not None:
    window = window[window["dt"] >= pd.to_datetime(start, utc=True)]
  if end is not None:
    window = window[window["dt"] <= pd.to_datetime(end, utc=True)]

  algorithms = [m.algorithm for m in dataset.models]
  features = [c for c in df.columns if c != "dt" and c not in algorithms]

  data = cluster_report.build_report(window, algorithms, features,
                                     naming=get_naming_clusters(dataset.name))
  data["dataset"] = dataset.name
  data["algorithms"] = algorithms
  data["available"] = available
  # Metrik mutu model sudah dihitung saat training dan tersimpan di ModelML —
  # dibaca dari sana, bukan dihitung ulang (nilainya milik data latih penuh,
  # bukan milik rentang yang sedang dilihat).
  data["quality"] = {
      m.algorithm: {k: v for k, v in (m.evaluation or {}).items()
                    if k in ("Silhouette", "Calinski-Harabasz", "Davies-Bouldin")}
      for m in dataset.models
  }
  return data


def get_cluster_unique(dataset: Dataset) -> Dict:
  """The distinct raw cluster labels per algorithm (from clusters.csv)."""
  fpath = Config.dir / "storages" / dataset.name / "results" / "clusters.csv"
  if not fpath.exists():
    raise ValueError(f"clusters.csv is not found in : {fpath}")
  df = pd.read_csv(fpath)
  algorithms = [a.algorithm for a in dataset.models]
  data = df[algorithms].to_dict(orient="list")
  return {key: list(set(data[key])) for key in data}


# Penamaan cluster tinggal di modul terpisah (bebas PyCaret) dan di-re-export
# di sini supaya permukaan `clustering_service.*` tidak berubah.
from app.services.cluster_naming import (  # noqa: E402
  get_naming_cluster,
  get_naming_clusters,
  naming_clusters,
  rename_clusters,
  save_naming_clusters,
  update_naming_clusters,
)


# --- Konteks hasil inference realtime --------------------------------------

# Titik dianggap tidak biasa kalau jaraknya melebihi ambang ini kali radius
# rata-rata cluster-nya. 2× dipilih karena radius adalah RATA-RATA: separuh
# anggota normal sudah berada di atas 1×, jadi ambang 1 akan menandai terlalu
# banyak. Pada data uji, pencilan sungguhan mencapai 21×.
UNUSUAL_RATIO = 2.0


# Baris latih + kolom label per algoritma, dipakai hanya oleh algoritma yang
# tidak punya `predict` (spectral clustering) untuk mencari centroid terdekat.
# Dibaca dari clusters.csv yang sama dengan laporan, jadi tidak ada artefak baru.
def _reference_frame(dataset: Dataset) -> Optional[pd.DataFrame]:
  fpath = Config.dir / "storages" / dataset.name / "results" / "clusters.csv"
  if not fpath.exists():
    return None
  return pd.read_csv(fpath)


# Episode yang sedang berjalan, per proses. Tidak dipersistensi: kalau service
# restart, hitungannya mulai dari nol — karena itu `exact` menandai apakah kita
# benar-benar MENYAKSIKAN perpindahannya atau label itu sudah aktif sejak kita
# mulai memantau. Tanpa penanda itu, "baru 2 menit" bisa menyesatkan.
_EPISODE_STATE: Dict[str, Dict[str, dict]] = {}


def _track_episode(dataset_name: str, algorithm: str, label: str, now) -> dict:
  """Catat sejak kapan cluster ini aktif, dan apakah awalnya teramati."""
  per_dataset = _EPISODE_STATE.setdefault(dataset_name, {})
  previous = per_dataset.get(algorithm)
  if not previous or previous["label"] != label:
    per_dataset[algorithm] = {
        "label": label, "since": now,
        "exact": bool(previous),        # False = sudah aktif sebelum kita memantau
    }
  return per_dataset[algorithm]


def reset_episode_state(dataset_name: Optional[str] = None) -> None:
  """Lupakan episode yang tercatat (dipakai test dan saat model dilatih ulang)."""
  if dataset_name is None:
    _EPISODE_STATE.clear()
  else:
    _EPISODE_STATE.pop(dataset_name, None)


def _build_context(dataset: Dataset, result, reference, now) -> Dict[str, dict]:
  """Konteks tiap algoritma: seberapa tidak biasa titik ini, seberapa umum
  cluster-nya, dan sudah berapa lama bertahan.

  Tanpa ini angka jarak tidak bisa dibaca: "1,2" tidak berarti apa-apa sampai
  dibandingkan dengan radius cluster dan seberapa sering kondisi itu muncul.
  """
  context: Dict[str, dict] = {}
  for algorithm, label in result.clusters.items():
    entry = dict(result.distances.get(algorithm) or {})
    # Label MENTAH ikut dikirim: `schema.clusters` sudah memakai nama pemberian
    # user kalau ada, sedangkan warna di frontend harus mengacu pada label yang
    # sama dengan halaman riwayat — kalau tidak, cluster yang sama tampil
    # berbeda warna di dua halaman.
    entry["label"] = str(label)
    entry["unusual"] = bool(entry.get("ratio") is not None and entry["ratio"] > UNUSUAL_RATIO)

    if reference is not None and algorithm in reference.columns:
      shares = reference[algorithm].astype(str).value_counts(normalize=True)
      entry["share"] = round(float(shares.get(str(label), 0.0)), 4)

    episode = _track_episode(dataset.name, algorithm, str(label), now)
    entry["since"] = episode["since"].isoformat()
    entry["duration_minutes"] = round((now - episode["since"]).total_seconds() / 60, 2)
    entry["duration_exact"] = episode["exact"]
    context[algorithm] = entry
  return context


# --- Read/write path -------------------------------------------------------

def inference(dataset: Dataset, features: Dict, logger) -> UnsupervisedResultSchema:
  """Tetapkan cluster untuk satu titik live.

  Memakai model yang sudah dilatih (`Unsupervised.assign`), bukan melatih ulang
  tiap panggilan. Versi sebelumnya transduktif: titik baru ditempelkan ke data
  latih lalu seluruhnya di-cluster ulang. Pengelompokannya konsisten (ARI 1.000
  antar panggilan), tapi PENOMORANNYA diacak setiap kali — pada data yang
  identik, label yang cocok 0%. Akibatnya warna di grafik realtime tidak bisa
  dibandingkan antar waktu dan nama cluster yang diberi user menempel ke
  kelompok yang salah.
  """
  feats = pd.DataFrame.from_dict(features)

  reference = _reference_frame(dataset)          # dipakai untuk centroid & pangsa
  result = Unsupervised.assign(
      dataset.to_cluster_assign_request(features, reference=reference), logger)

  naming = get_naming_clusters(dataset.name)
  clusters = {algo: (naming[algo][raw] if naming and raw in naming.get(algo, {}) else raw)
              for algo, raw in result.clusters.items()}

  now = DTEncoder.now()
  schema = UnsupervisedResultSchema(
      features=feats.iloc[0].to_dict(), dataset_name=dataset.name,
      clusters=clusters, timestamp=now, is_valid=True,
      context=_build_context(dataset, result, reference, now))
  if len(dataset.features) > 2:
    reduced = transform_pca(dataset.name, feats, logger)
    if reduced is not None:
      schema.features_reduce = reduced.iloc[0].to_dict()

  return schema


def auto_inference(dataset: Dataset, logger) -> Optional[UnsupervisedResultSchema]:
  """Fetch live features and run one clustering inference."""
  logger.info("auto inference start ...")
  try:
    X = pull_realtime(dataset.features, logger)
    features = {col: [X[i]] for i, col in enumerate(dataset.features)}
    return inference(dataset, features, logger)
  except Exception as e:
    logger.error(str(e))
    return None


# --- Worker loop -----------------------------------------------------------

def auto_inference_write_loop(dataset: Dataset, logger) -> None:
  """Worker loop: run clustering inference every interval, forever."""
  logger.info("start auto inference write loop ...")
  n = 1
  try:
    while True:
      logger.info(f"auto inference - {n}")
      auto_inference(dataset, logger)
      n += 1
      time.sleep(dataset.interval * 60)
  except KeyboardInterrupt:
    return
  except Exception as e:
    logger.error(str(e))
    logger.error(traceback.format_exc())


def main() -> None:
  """Worker entry point (pm2 runs this file with the dataset name as argv)."""
  worker.run_from_argv(auto_inference_write_loop)


if __name__ == "__main__":
  main()
