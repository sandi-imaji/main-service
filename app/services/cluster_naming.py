"""Penamaan cluster — baca/tulis nama manusiawi untuk label mentah.

Dipisahkan dari `app.services.clustering` karena isinya murni operasi berkas
JSON dan penggantian string: tidak butuh PyCaret sama sekali. Sebelumnya
memanggil `get_naming_clusters` saja menarik `Unsupervised` → `pycaret.clustering`,
yang menambah ~2 detik pada permintaan pertama endpoint tabel dataset.

`app.services.clustering` me-re-export semuanya, jadi pemanggil lama tidak
perlu berubah.
"""
import json
from typing import Dict, Optional

import numpy as np
import pandas as pd

from app.config import Config


def get_naming_clusters(dataset_name: str) -> Optional[Dict]:
  """Load the saved {algorithm: {raw_label: name}} mapping, or None."""
  fpath = Config.dir / "storages" / dataset_name / "results" / "naming_clusters.json"
  if not fpath.exists():
    return None
  with open(fpath, "r") as f:
    return json.load(f)


def get_naming_cluster(dataset_name: str, algorithm: str, cluster_raw: str) -> Optional[str]:
  data = get_naming_clusters(dataset_name)
  if not data:
    return None
  return data[algorithm][cluster_raw]


def save_naming_clusters(dataset, naming: Dict[str, Dict[str, str]]) -> None:
  fpath = Config.dir / "storages" / dataset.name / "results" / "naming_clusters.json"
  with open(fpath, "w") as f:
    json.dump(naming, f)


def update_naming_clusters(dataset, update_naming: Dict[str, Dict[str, str]]):
  fpath = Config.dir / "storages" / dataset.name / "results" / "naming_clusters.json"
  if not fpath.exists():
    save_naming_clusters(dataset, update_naming)
    return
  naming = get_naming_clusters(dataset.name)
  for model, clusters in naming.items():
    mapper = update_naming.get(model, {})
    for cluster_name, value in clusters.items():
      if value in mapper:
        clusters[cluster_name] = mapper[value]
  save_naming_clusters(dataset, naming)
  return naming


def naming_clusters(clusters: pd.DataFrame, naming: Dict[str, str], logger):
  """Validate that every observed cluster has a name, then map them."""
  if not all(clust in naming.keys() for clust in np.unique(clusters)):
    message = f"{np.unique(clusters).tolist()} not in naming : {list(naming.keys())}"
    logger.error(message)
    raise ValueError(message)
  return [naming[clust] for clust in clusters]


def rename_clusters(df: pd.DataFrame, naming_clusters: dict, inplace: bool = False) -> pd.DataFrame:
  """Rename cluster labels per model using {col: {raw_label: name}}.

  Memakai `.replace`, bukan `.map`: `.map` mengubah label yang TIDAK ada di
  pemetaan menjadi NaN, sehingga penamaan yang belum lengkap diam-diam
  menghapus data (mis. user baru menamai 2 dari 3 cluster). `.replace`
  membiarkan label yang belum dinamai apa adanya.
  """
  target = df if inplace else df.copy()
  for col, mapping in naming_clusters.items():
    if col not in target.columns:
      continue
    target[col] = target[col].replace(mapping)
  return target
