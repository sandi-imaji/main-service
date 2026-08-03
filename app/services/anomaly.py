"""Anomaly detection — training, realtime inference, and worker orchestration.

Anomaly is an ADD-ON task: it attaches to a dataset whose primary task may be
Regression/Clustering/TimeSeries. Hence its model is kept as a single artifact at
`<storage>/anomaly.pkl` (not per-algorithm under `top_model/`) and its worker
runs separately under the name `<dataset>-Anomaly`.

This file doubles as the worker entry point: pm2 runs it directly with the
dataset name as argv — see `main()` below.
"""
# Script-entry bootstrap: when pm2 runs this file directly the project root is
# not yet importable, so put it on sys.path BEFORE the `app.*` imports below.
if __name__ == "__main__":
  import sys
  from pathlib import Path
  _root = Path(__file__).resolve().parents[2]
  if str(_root) not in sys.path: sys.path.insert(0, str(_root))

import json

from app.core.anomaly import Anomaly
from app.database.schemas import AnomalyRequestSchema, TaskType
from app.logger import Logger
from app.pull import pull_realtime
from app.services import worker
from app.services.results import AnomalyResultSchema
from app.workers.manager import WorkerManager


def auto_anomaly_detection(dataset, logger) -> AnomalyResultSchema:
  """Pull the latest features, run the pure anomaly predict, and wrap the result."""
  logger.info("Auto Inference Start ...")
  try:
    columns = list(dataset.meta.columns) if dataset.meta.columns else \
              dataset.open_dataframe().drop(columns="dt").columns.tolist()
    columns = [c for c in columns if c != "dt"]

    pulled = pull_realtime(columns, logger)
    if not any(v is not None for v in pulled):  # nothing live -> non-valid result
      return AnomalyResultSchema.invalid(dataset.name)
    pulled = [v if v is not None else 0.0 for v in pulled]
    features = {col: [pulled[i]] for i, col in enumerate(columns)}

    result = Anomaly.predict(dataset.to_anomaly_predict_request(features), logger)
    logger.info("Auto Inference Finished!")
    return AnomalyResultSchema.from_result(result, dataset.name)
  except Exception as e:
    logger.warning(f"Anomaly auto inference error: {e}")
    return AnomalyResultSchema.invalid(dataset.name)


METRICS_FILE = "anomaly_metrics.json"


def metrics_path(dataset):
  return dataset.path / METRICS_FILE


def read_metrics(dataset) -> dict:
  """Detection summary from the last anomaly training, or an empty dict.

  Anomaly has NO `ModelML` row — it attaches to another task's dataset and keeps
  a single model at `<storage>/anomaly.pkl`. Its metrics therefore cannot ride
  along in `model.evaluation` like the other tasks, and are stored next to the
  artifact instead.
  """
  fpath = metrics_path(dataset)
  if not fpath.exists():
    return {}
  try:
    return json.loads(fpath.read_text(encoding="utf-8"))
  except (json.JSONDecodeError, OSError):
    return {}


def _save_metrics(dataset, evaluation: dict, logger) -> None:
  """Write the detection summary next to `anomaly.pkl`."""
  try:
    metrics_path(dataset).write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    logger.info(f"Saved anomaly metrics: {evaluation}")
  except OSError as e:
    # The model is already saved and usable; the summary is a supplement, so a
    # write failure here must not invalidate the training run.
    logger.warning(f"Failed to save anomaly metrics: {e}")


def auto_anomaly(payload: AnomalyRequestSchema, dataset):
  """Train an anomaly model for a dataset, then start its worker."""
  task_type = TaskType.Anomaly
  logger = Logger(payload.dataset_name)
  logger.info("Run Auto Anomaly ...")
  trained = Anomaly.train_one(
      dataset.to_anomaly_train_request(payload.fraction, payload.algorithm), logger)
  # This return value used to be discarded outright, so the detection summary the
  # core computed was never visible to anyone. Checked before use: the model is
  # already saved at this point, so an unexpectedly empty summary must not fail a
  # training run that actually succeeded.
  evaluation = getattr(trained, "evaluation", None)
  if evaluation:
    _save_metrics(dataset, evaluation, logger)

  task_name = f"{payload.dataset_name}-{task_type}"
  if WorkerManager.is_active(task_name): return {"detail": f"{task_name} is still active"}
  WorkerManager.create(payload.dataset_name, task_type)


# --- Worker ------------------------------------------------------------------

def auto_inference_write_loop(dataset, logger) -> None:
  """Anomaly worker loop: detect → write to InfluxDB.

  No finetune check here: the anomaly core has no `retrain_models`, and this
  worker rides on a dataset owned by another task — triggering a finetune here
  would retrain the primary task's models.
  """
  worker.influx_write_loop(dataset, logger, auto_anomaly_detection)


def main():
  """Worker entry point (pm2 runs this file with the dataset name as argv)."""
  worker.run_from_argv(auto_inference_write_loop)


if __name__ == "__main__":
  main()
