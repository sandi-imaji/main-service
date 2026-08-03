"""
Core compute contracts.

Plain data structures exchanged between the service layer and the ML core.
Intentionally free of any Dataset / DB / HTTP knowledge, so the core can be
imported and tested in isolation (this module imports nothing from
`app.database` or `fastapi`).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True, kw_only=True)
class TrainRequest:
  """Training inputs shared by every task type — no Dataset, no DB session.

  Task-specific inputs live on the per-task subclasses below, so an invalid
  combination (e.g. a clustering request carrying a supervised target) simply
  cannot be constructed. `kw_only` keeps the frozen-dataclass inheritance clean
  (subclasses may add required fields after the base's defaulted `n_top`).
  """
  df: pd.DataFrame
  preprocessing: dict          # already-resolved pycaret kwargs
  out_dir: Path                # absolute directory to save model artifacts into
  task: str                    # task-type name, e.g. "Regression"/"Clustering"
  n_top: int = 1

@dataclass(frozen=True, kw_only=True)
class SupervisedTrainRequest(TrainRequest):
  """Classification / Regression."""
  target: str                  # label column


@dataclass(frozen=True, kw_only=True)
class ClusteringTrainRequest(TrainRequest):
  """Clustering (unsupervised)."""
  n_clusters: int              # number of clusters to fit


@dataclass(frozen=True, kw_only=True)
class AnomalyTrainRequest(TrainRequest):
  """Anomaly detection."""
  fraction: float              # expected outlier fraction
  algorithm: str


@dataclass(frozen=True, kw_only=True)
class TimeSeriesTrainRequest(TrainRequest):
  """Forecasting."""
  fh: int                      # forecast horizon (number of steps ahead)

# pycaret cross-validation bookkeeping columns, not model-quality metrics.
# `cutoff` marks each fold's split point; on the "Mean" row it comes out as NaN.
_NON_METRIC_COLUMNS = {"cutoff", "Model", "Object", "index"}


def fold_spread(pulled) -> dict:
  """Extract the between-fold standard deviation from pycaret's `pull()` table.

  `pull()` after `create_model` returns one row per fold plus a "Mean" and a
  "Std" row. The code kept only "Mean", which threw away the model's STABILITY:
  MAPE 0.05 ± 0.01 and MAPE 0.05 ± 0.20 were stored as the exact same number,
  even though the second means performance depends heavily on which slice of the
  data happens to be used.

  Returned with a `Std ` prefix (e.g. `"Std MAPE"`) so it sits alongside the
  metric it belongs to without changing the shape of the JSON column.

  Returns an empty dict when the table has no "Std" row — the `compare_models`
  path uses the comparison table, which does not carry one.
  """
  try:
    if pulled is None or "Std" not in pulled.index:
      return {}
    return {f"Std {k}": v for k, v in pulled.loc["Std"].to_dict().items()}
  except (AttributeError, KeyError, TypeError):
    # `pull()` may return an unexpected shape; stability is a bonus, not worth
    # failing a training run that already completed.
    return {}


def clean_metrics(evaluation: dict) -> dict:
  """Tidy pycaret's metric table before it goes into the JSON column.

  Two things happen, and the second one prevents a real failure:

  1. Bookkeeping columns are dropped — not metrics, of no use to anyone.
  2. Non-finite values (NaN/Inf) become `None`. THIS IS THE IMPORTANT PART:
     FastAPI serialises responses with `json.dumps(..., allow_nan=False)` and
     raises `ValueError: Out of range float values are not JSON compliant` the
     moment it meets a NaN. `TimeSeries.retrain_models` reads
     `pull().loc["Mean"]`, which ALWAYS carries `cutoff = NaN` — so without this
     cleanup the first forecast auto-retrain would make every endpoint that
     returns that model answer HTTP 500.

  Values become `None` rather than disappearing so a genuinely undefined metric
  (e.g. R² on a constant fold) still shows up as "not available" instead of
  vanishing without trace.
  """
  cleaned = {}
  for key, value in (evaluation or {}).items():
    if key in _NON_METRIC_COLUMNS:
      continue
    if isinstance(value, float) and not math.isfinite(value):
      value = None
    elif hasattr(value, "item"):        # numpy scalar → native Python type
      value = value.item()
      if isinstance(value, float) and not math.isfinite(value):
        value = None
    cleaned[key] = value
  return cleaned


@dataclass(frozen=True)
class TrainedModel:
  """One trained artifact produced by the core."""
  algorithm: str
  path: str                    # absolute path of the saved model (without extension)
  evaluation: dict
  size: int                    # artifact size in bytes

  @classmethod
  def from_saved(cls, algorithm: str, path, evaluation: dict) -> "TrainedModel":
    """Describe a model just saved at `path`. It is a pycaret artifact, so the
    real file on disk is `{path}.pkl` — this centralises that convention.

    Every core funnels through here, so this is where metric cleaning lives:
    one place, covering all four task types.
    """
    return cls(algorithm=algorithm, path=str(path), evaluation=clean_metrics(evaluation),
               size=os.path.getsize(f"{path}.pkl"))


@dataclass(frozen=True)
class PredictRequest:
  """Everything the core needs to run inference — no Dataset, no DB."""
  features: dict                       # {"x1": [value], "x2": [value], ...}
  models: List[Tuple[str, str]]        # [(algorithm, absolute_model_path), ...]
  task: str                            # task-type name, e.g. "Regression"


@dataclass(frozen=True)
class PredictResult:
  """Predictions produced by the core."""
  features: dict                       # flattened input features {name: value}
  predictions: dict                    # {algorithm: predicted_value}


@dataclass(frozen=True)
class AnomalyResult:
  """One anomaly-detection inference produced by the core."""
  features: dict                       # flattened input features {name: value}
  is_anomaly: bool
  anomaly_score: float


@dataclass(frozen=True)
class ForecastRequest:
  """Everything the core needs to forecast — no Dataset, no timestamps.

  Forecasting has no input features; it just projects each model `fh` steps
  ahead. Timestamps for those steps are an output-layer concern (they depend on
  the dataset interval / current time), so the service attaches them.
  """
  models: List[Tuple[str, str]]        # [(algorithm, absolute_model_path), ...]
  fh: int                              # forecast horizon (number of steps ahead)
  task: str                            # task-type name, e.g. "TimeSeries"


@dataclass(frozen=True)
class ForecastResult:
  """Forecasts produced by the core."""
  forecast: dict                       # {algorithm: [predicted values]}


@dataclass(frozen=True)
class ClusterAssignRequest:
  """Assign clusters for new points using an ALREADY-trained model.

  Replaces the transductive approach (`ClusterRequest`), which refit the model on
  every call. Refitting shuffles the cluster numbering on every tick — measured
  on identical data: the grouping is exactly the same (ARI 1.000) but 0% of the
  labels match. The consequence is that colours on the realtime chart cannot be
  compared over time, and user-given cluster names stick to the wrong group.

  `reference` is only used by algorithms without a `predict` method (e.g.
  spectral clustering, which is transductive by construction): the training rows
  and their label column, used to find the nearest centroid in the same space as
  the model.
  """
  features: dict                       # {"x1": [value], ...}
  models: List[Tuple[str, str]]        # [(algorithm, absolute_model_path), ...]
  task: str
  reference: Optional[pd.DataFrame] = None


@dataclass(frozen=True)
class ClusterRequest:
  """Everything the core needs to assign clusters — no Dataset, no storage.

  Clustering is transductive here: the new point(s) are appended to the training
  rows and the whole set is re-clustered, so `df` carries both. A `dt` column may
  be present; the core ignores it. Naming / PCA / CSV persistence are output-layer
  concerns handled by the service.
  """
  df: pd.DataFrame                     # training rows + the new point(s), 'dt' ignored
  algorithms: List[str]                # algorithms to assign with
  n_clusters: int
  preprocessing: dict                  # already-resolved pycaret kwargs
  task: str                            # task-type name, e.g. "Clustering"


@dataclass(frozen=True)
class ClusterResult:
  """Cluster assignments produced by the core (raw labels — no human naming)."""
  clusters: dict                       # {algorithm: raw label of the last/new row}
  assignments: dict                    # {algorithm: [raw label per row of df]}
  # {algorithm: {"distance": float, "radius": float, "ratio": float}} — the
  # point's distance to its own cluster centroid, against that cluster's mean
  # radius. A label alone never reveals that a point sits far from the centre of
  # its group; this ratio is what surfaces it.
  distances: dict = field(default_factory=dict)
