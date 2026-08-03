"""Running model performance, computed from predictions vs realised actuals.

The metrics stored on `ModelML.evaluation` are a snapshot taken at TRAINING time
— cross-validation means over the training data, never updated afterwards. A
model with 2% MAPE three months ago may be far off today because the pattern
shifted, and nothing in the system would show it.

The raw material already exists: `services/timeseries/actuals.py` writes the
realised value onto each inference record in InfluxDB once its timestamp passes,
alongside every model's prediction. This module is what finally uses it —
pairing the two and computing the error that actually occurred.

Only TimeSeries has actuals: supervised predictions are made for points whose
true value the system never gets to learn.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from app.database.DB import Dataset
from app.database.historicalDB import get_influx_storage
from app.exceptions import ValidationException

# Below this, the ratio is not reported. MAPE divides by the actual value, so an
# actual near zero explodes the percentage — a single such point can produce a
# 4000% MAPE and drown the whole summary.
_MIN_DENOMINATOR = 1e-9

# Warning threshold: running error this many times the training error counts as
# genuine drift rather than ordinary noise.
DRIFT_WARN_RATIO = 2.0


def _finite(value) -> Optional[float]:
  """None for anything that cannot be sent as JSON."""
  if value is None:
    return None
  try:
    value = float(value)
  except (TypeError, ValueError):
    return None
  return value if math.isfinite(value) else None


def _pairs(records: List[dict]) -> Dict[str, List[tuple]]:
  """Collect (prediction, actual) pairs per algorithm.

  Records without an `actual` are skipped — those are predictions for a time
  that has not arrived yet, not failures.
  """
  per_model: Dict[str, List[tuple]] = {}
  for row in records:
    actual = _finite(row.get("actual"))
    if actual is None:
      continue
    for algorithm, predicted in (row.get("results") or {}).items():
      predicted = _finite(predicted)
      if predicted is None:
        continue
      per_model.setdefault(algorithm, []).append((predicted, actual))
  return per_model


def _metrics(pairs: List[tuple]) -> dict:
  """MAE / RMSE / MAPE from prediction-actual pairs."""
  n = len(pairs)
  errors = [p - a for p, a in pairs]
  mae = sum(abs(e) for e in errors) / n
  rmse = math.sqrt(sum(e * e for e in errors) / n)

  # MAPE covers only the points with a usable denominator; how many were used is
  # reported too, so the number is not mistaken for the whole window.
  ratios = [abs(p - a) / abs(a) for p, a in pairs if abs(a) > _MIN_DENOMINATOR]
  mape = sum(ratios) / len(ratios) if ratios else None

  # Bias: the mean SIGNED error. MAE hides it — a model that always guesses 5
  # too high and one that misses randomly by ±5 share a MAE, yet only the first
  # one can be corrected for.
  bias = sum(errors) / n

  return {
      "MAE": round(mae, 6),
      "RMSE": round(rmse, 6),
      "MAPE": round(mape, 6) if mape is not None else None,
      "Bias": round(bias, 6),
      "n": n,
      "n_mape": len(ratios),
  }


def _drift(live_mape: Optional[float], trained_mape: Optional[float]) -> dict:
  """Compare the running error against the error seen during training."""
  if live_mape is None or not trained_mape or trained_mape <= 0:
    return {"ratio": None, "status": "unknown"}
  ratio = live_mape / trained_mape
  return {
      "ratio": round(ratio, 3),
      "status": "degraded" if ratio >= DRIFT_WARN_RATIO else "ok",
  }


def running_metrics(dataset: Dataset, start: Optional[str] = None,
                    end: Optional[str] = None, limit: int = 1000) -> Dict[str, Any]:
  """Each model's real performance over a time range.

  Returns, per algorithm: the running error, the training error, and the ratio
  between them — so "is this model still good?" can be answered rather than
  guessed at.
  """
  if not dataset.task_type.is_timeseries():
    raise ValidationException(
        f"Running performance is only available for TimeSeries; "
        f"'{dataset.name}' is {dataset.task_type}. Other tasks never learn the "
        f"true value of the points they predicted.")

  influx = get_influx_storage()
  records = influx.query_inference(dataset_name=dataset.name, start=start, end=end,
                                   limit=limit)
  per_model = _pairs(records)

  trained = {m.algorithm: (m.evaluation or {}).get("MAPE") for m in dataset.models}

  models = {}
  for algorithm, pairs in sorted(per_model.items()):
    result = _metrics(pairs)
    trained_mape = _finite(trained.get(algorithm))
    result["trained_MAPE"] = trained_mape
    result["drift"] = _drift(result["MAPE"], trained_mape)
    models[algorithm] = result

  matched = sum(len(p) for p in per_model.values())
  return {
      "dataset": dataset.name,
      "total_records": len(records),
      # How many records already have their actual filled in. Low coverage means
      # the numbers above rest on few points — or that the actuals back-fill is
      # lagging behind.
      "records_with_actual": sum(1 for r in records if _finite(r.get("actual")) is not None),
      "matched_points": matched,
      "models": models,
  }
