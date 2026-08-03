"""Supervised (Regression / Classification) — orkestrasi inference realtime.

Menarik data live, menjalankan `predict` di core yang murni, lalu membungkus
hasilnya. Lapisan ini Dataset/DB-aware; komputasi ML-nya tinggal di app.core.

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

from app.config import Config
from app.helpers import DTEncoder
from app.pull import pull_realtime
from app.services import worker
from app.services.results import SupervisedResultSchema


def auto_predictions(dataset, logger):
  """Pull the latest features + actual, predict, and wrap the result."""
  logger.info("Auto Inference Start ...")
  try:
    features_names = dataset.features
    columns = features_names + [dataset.target]

    if Config.debug_mode:
      logger.info("Use Sampling from dataframe ...")
      pulled = dataset.open_dataframe().sample(1).drop(columns="dt")
      pulled_list = pulled[columns].values.flatten().tolist()
    else:
      pulled_list = pull_realtime(columns, logger)

    if sum(pulled_list) == 0:                  # no live data → non-valid result
      algorithms = [m.algorithm for m in dataset.models]
      return SupervisedResultSchema(
          features={k: 0.0 for k in dataset.features},
          predictions={k: 0.0 for k in algorithms},
          timestamp=DTEncoder.now(), dataset_name=dataset.name, actual=0)

    X, actual = pulled_list[:-1], pulled_list[-1]
    features = {name: [X[i]] for i, name in enumerate(features_names)}

    result = dataset.task_type.core().predict(dataset.to_predict_request(features), logger)
    logger.info("Auto Inference Finished!")
    return SupervisedResultSchema.from_result(result, dataset.name, actual=actual)

  except Exception as e:
    logger.error(f"Auto inference error: {e}")
    return None


# --- Finetune terjadwal (hanya supervised) ----------------------------------

def _sync_finetune_marker(dataset, logger) -> None:
  """finetune commits a new current_dt in its OWN DB session. This worker loop
  holds a long-lived dataset object, so refresh its in-memory current_dt from
  the DB — otherwise the stale value keeps `is_time_to_finetune()` True and
  finetune re-fires every tick."""
  from app.database.base import get_db_session
  from app.database.DB import Dataset
  try:
    with get_db_session() as db:
      fresh = Dataset.get_by_name(dataset.name, db)
      if fresh is not None:
        # in-place update on the JSON-column pydantic object is NOT ORM-tracked,
        # so it won't dirty/flush this loop's own session.
        dataset.meta.current_dt = fresh.meta.current_dt
        dataset.meta.last_update = fresh.meta.last_update
  except Exception as e:
    logger.warning(f"sync finetune marker failed: {e}")


def finetune_if_due(dataset, logger) -> None:
  """Refit kalau jadwalnya sudah tiba. Dijalankan tiap tick sebelum inference."""
  if not dataset.is_time_to_finetune():
    return
  logger.info("it's Time to finetune Dataset :)")
  from app.services import train as train_service
  train_service.finetune(dataset, logger)
  _sync_finetune_marker(dataset, logger)


# --- Worker ------------------------------------------------------------------

def auto_inference_write_loop(dataset, logger) -> None:
  """Worker loop supervised: cek finetune → prediksi → tulis ke InfluxDB."""
  worker.influx_write_loop(dataset, logger, auto_predictions,
                           before_tick=finetune_if_due)


def main():
  """Worker entry point (pm2 runs this file with the dataset name as argv)."""
  worker.run_from_argv(auto_inference_write_loop)


if __name__ == "__main__":
  main()
