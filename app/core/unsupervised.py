"""
Unsupervised (Clustering) core — pure ML compute.

Dua jalur penetapan cluster:
  `assign`  — memakai model yang sudah dilatih. Penomoran cluster STABIL, jadi
              inilah yang dipakai jalur realtime dan inference.
  `predict` — transduktif: melatih ulang lalu membaca label baris terakhir.
              Penomorannya diacak tiap panggilan, jadi hasilnya tidak bisa
              dibandingkan antar waktu. Disimpan untuk eksplorasi satu kali jalan.

No Dataset / DB / storage / HTTP knowledge: everything is expressed through the
contracts in `app.core.contracts`. The Dataset/storage-aware orchestration
(inference read/write path, cluster naming, PCA reduction, CSV persistence,
worker loop) lives in `app.services.clustering`.
"""
import os,time,numpy as np,pandas as pd,joblib
from typing import List

from app.config import Config
from app.core.base import BaseMLCore, ALGO_LIST
from app.core.contracts import (ClusterAssignRequest, ClusterRequest, ClusterResult,
                                ClusteringTrainRequest, TrainedModel)
from app.helpers import pca
from app.utils.model_cache import get_unsupervised_cache

from pycaret import clustering as mod


class Unsupervised(BaseMLCore):
  """Clustering operations with shared model caching."""
  _task_type = "unsupervised"

  @classmethod
  def get_cache(cls):
    """Get the model cache for unsupervised learning."""
    return get_unsupervised_cache()

  @staticmethod
  def is_same_cluster(clusters: dict, name: str | None = None) -> bool:
    """True when every algorithm agrees on a single cluster (optionally `name`)."""
    values = set(clusters.values())
    result = len(values) == 1
    if not name: return result
    return result and name == list(values)[0]

  @staticmethod
  def _cluster_stats(pipeline, reference, algorithm: str, point):
    """Centroid dan radius rata-rata tiap cluster, di ruang milik model sendiri.

    Baris latih dilewatkan pipeline preprocessing model (imputasi → normalisasi
    → PCA) supaya jaraknya dihitung di ruang yang sama dengan tempat model
    mengelompokkan — bukan di ruang fitur mentah yang skalanya campur aduk.

    Return: (stats per label, koordinat titik baru), atau (None, None) kalau
    acuannya tidak tersedia.
    """
    if reference is None or algorithm not in reference.columns:
      return None, None

    # kolom algoritma LAIN juga harus dibuang — irisan dengan kolom titik baru
    # menyisakan tepat kolom fiturnya.
    features = [c for c in reference.columns
                if c not in ("dt", algorithm) and c in point.columns]
    transform = pipeline[:-1].transform            # semua langkah kecuali modelnya

    reference_z = pd.DataFrame(transform(reference[features]))
    reference_z.index = reference.index

    stats = {}
    for label, rows in reference_z.groupby(reference[algorithm].astype(str)):
      centroid = rows.mean().to_numpy()
      radius = float(np.linalg.norm(rows.to_numpy() - centroid, axis=1).mean())
      stats[str(label)] = {"centroid": centroid, "radius": radius}

    return stats, np.asarray(transform(point[features]))[0]

  @staticmethod
  def assign(req: ClusterAssignRequest, logger) -> ClusterResult:
    """Tetapkan cluster titik baru memakai model yang sudah dilatih.

    Komputasi murni: model dimuat lewat cache bersama, tidak ada pelatihan ulang.
    Karena berasal dari satu model yang tetap, penomoran cluster stabil sepanjang
    waktu — syarat agar grafik realtime dan penamaan cluster bermakna.

    Selain label, ikut dihitung jarak titik ke centroid cluster-nya dan
    perbandingannya terhadap radius cluster itu. Label sendiri tidak pernah
    menunjukkan bahwa sebuah titik duduk jauh dari pusat kelompoknya.
    """
    features_df = pd.DataFrame.from_dict(req.features)

    clusters, distances = {}, {}
    for algorithm, path in req.models:
      model = Unsupervised._load_model_cached(mod, path, logger)
      stats, point_z = Unsupervised._cluster_stats(model, req.reference, algorithm, features_df)

      try:
        result = mod.predict_model(model, data=features_df)
        clusters[algorithm] = str(result["Cluster"].iloc[0])
      except TypeError:
        # pycaret melempar TypeError untuk model tanpa metode predict (mis.
        # spectral clustering, yang transduktif secara matematis). Cadangannya
        # centroid terdekat — sebuah pendekatan, tapi konsisten antar waktu.
        if stats is None:
          raise ValueError(
              f"'{algorithm}' tidak punya predict dan tidak ada data acuan untuk centroid")
        nearest = min(stats, key=lambda lbl: np.linalg.norm(point_z - stats[lbl]["centroid"]))
        clusters[algorithm] = nearest
        logger.debug(f"{algorithm}: tanpa predict → centroid terdekat = {nearest}")

      label = clusters[algorithm]
      if stats and label in stats:
        distance = float(np.linalg.norm(point_z - stats[label]["centroid"]))
        radius = stats[label]["radius"]
        distances[algorithm] = {
            "distance": round(distance, 3),
            "radius": round(radius, 3),
            "ratio": round(distance / radius, 2) if radius else None,
        }
      logger.info(f"Assign cluster with {algorithm}: {label}")

    return ClusterResult(clusters=clusters, assignments={}, distances=distances)

  @staticmethod
  def predict(req: ClusterRequest, logger) -> ClusterResult:
    """
    Assign clusters for `req.df` (training rows + the new point appended).

    Pure compute: PyCaret clustering is transductive here — for each algorithm we
    set up on the combined frame, fit, and read the `Cluster` column. Returns the
    raw label of the last row per algorithm plus the full per-row assignment.
    Naming, PCA, and CSV persistence are the service's job.
    """
    if req.preprocessing: mod.setup(req.df, ignore_features=["dt"], **req.preprocessing)
    else: mod.setup(req.df, ignore_features=["dt"])

    logger.info("Inference models : ")
    clusters, assignments = {}, {}
    for algorithm in req.algorithms:
      model = mod.create_model(algorithm, num_clusters=req.n_clusters, verbose=Config.verbose.pycaret)
      result = mod.assign_model(model, verbose=Config.verbose.pycaret)
      assignments[algorithm] = result["Cluster"].tolist()
      clusters[algorithm] = result.tail(1)["Cluster"].item()
      logger.info(f"Inference with {algorithm}")
    return ClusterResult(clusters=clusters, assignments=assignments)


  @staticmethod
  def compare_models(req: ClusteringTrainRequest, logger) -> List[TrainedModel]:
    """
    Train every clustering algorithm, rank them, and persist the top-N.

    Pure compute (no Dataset/DB): model artifacts go into `req.out_dir`, and the
    two clustering side-artifacts (PCA object + cluster-assignment CSV) into the
    dataset storage root (`req.out_dir.parent`). Cluster labels are written raw;
    human naming is applied later at read time (get_clusters). Returns the top-N
    descriptors, best first.
    """
    df = req.df
    features = [c for c in df.columns if c != "dt"]
    storage_root = req.out_dir.parent          # out_dir == <storage>/top_model

    # PCA (only when >2 features) — used later for 2D viz / inference.
    if len(features) > 2:
      pca_obj = pca(df[features].copy(), n_components=2, only_obj=True)
      joblib.dump(pca_obj, storage_root / "pca.joblib")
      logger.info("Saved PCA Object...")

    mod.setup(df, verbose=Config.verbose.pycaret, ignore_features=["dt"], **req.preprocessing)
    logger.info(f"Features : {mod.get_config('X').columns.tolist()}")

    # Train each algorithm, capturing its metric row and wall-time.
    trained = {}                               # algo -> (model, metric_row, train_time)
    list_algorithm = ALGO_LIST[req.task]
    for algo, desc in list_algorithm.items():
      tic = time.monotonic()
      model = mod.create_model(algo, num_clusters=req.n_clusters, verbose=Config.verbose.pycaret)
      row = mod.pull().copy().iloc[0].to_dict()
      row["Algorithm"] = algo
      row["Description"] = desc
      trained[algo] = (model, row, time.monotonic() - tic)
      logger.info(f"{algo} successfully trained!")

    # Rank: Silhouette desc, Calinski-Harabasz desc, Davies-Bouldin asc.
    def _score(row):
      return (
        -(row.get("Silhouette", -1e9) if row.get("Silhouette") is not None else -1e9),
        -(row.get("Calinski-Harabasz", -1e9) if row.get("Calinski-Harabasz") is not None else -1e9),
        (row.get("Davies-Bouldin", 1e9) if row.get("Davies-Bouldin") is not None else 1e9),
      )
    ranked = sorted((row for _, row, _ in trained.values()), key=_score)

    os.makedirs(req.out_dir, exist_ok=True)
    clusters_df = df.copy()
    out: List[TrainedModel] = []
    for row in ranked[:req.n_top]:
      algo = row["Algorithm"]
      model, _, train_time = trained[algo]
      path = req.out_dir / algo
      mod.save_model(model, str(path))
      clusters_df[algo] = mod.assign_model(model)["Cluster"]

      metric = {k: v for k, v in row.items() if k not in ("Algorithm", "Description")}
      metric["TT (Sec)"] = train_time
      out.append(TrainedModel.from_saved(algo, path, metric))
      logger.info(f"Save Model at {path}")

    results_dir = storage_root / "results"
    os.makedirs(results_dir, exist_ok=True)
    clusters_df.to_csv(results_dir / "clusters.csv", index=False)

    get_unsupervised_cache().clear()
    return out

  @staticmethod
  def train_one(req: ClusteringTrainRequest, algorithm: str, logger) -> TrainedModel:
    """Train a single clustering algorithm. Pure compute — returns its descriptor."""
    mod.setup(req.df, verbose=Config.verbose.pycaret, ignore_features=["dt"], **req.preprocessing)

    model = mod.create_model(algorithm, num_clusters=req.n_clusters, verbose=Config.verbose.pycaret)
    metric = mod.pull().iloc[0].to_dict()

    os.makedirs(req.out_dir, exist_ok=True)
    path = req.out_dir / algorithm
    mod.save_model(model, str(path))
    logger.info(f"Save Model at {path}")

    get_unsupervised_cache().clear()
    return TrainedModel.from_saved(algorithm, path, metric)


