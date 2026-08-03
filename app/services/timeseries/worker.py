"""Worker time-series (entry pm2).

Beda dari worker task lain: alih-alih memprediksi tiap interval, worker ini
menyusuri horizon forecast — menunggu tiap timestamp tiba lalu menuliskan nilai
aktualnya berdampingan dengan prediksi yang sudah dibuat.
"""
# Script-entry bootstrap: when pm2 runs this file directly the project root is
# not yet importable, so put it on sys.path BEFORE the `app.*` imports below.
if __name__ == "__main__":
  import sys
  from pathlib import Path
  _root = Path(__file__).resolve().parents[3]
  if str(_root) not in sys.path: sys.path.insert(0, str(_root))

import time
import traceback

from app.database.DB import Dataset
from app.database.historicalDB import InfluxDBStorage
from app.helpers import DTEncoder
from app.pull import get_realtime
from app.services import worker as service_worker
from app.services.timeseries.forecast import forecast
from app.services.timeseries.horizon import is_expired
from app.services.timeseries.retrain import retrain


def auto_inference_write_loop(dataset: Dataset, logger) -> None:
  """Worker walk-through of the forecast horizon: refit if expired, forecast,
  then for each future timestamp sleep until it arrives and write the realised
  actual alongside the prediction."""
  logger.info("start auto inference write loop ...")
  try:
    if is_expired(dataset, logger): retrain(dataset, logger)
    results = forecast(dataset, logger)
    feature = dataset.features[0]
    model_names = list(results.forecast.keys())
    with InfluxDBStorage() as writer:
      results.write_to_influx(writer, logger)
      for idx, ts in enumerate(results.timestamps):
        logger.info(f"auto inference - {idx + 1}")
        now = DTEncoder.now()
        if now > ts: continue
        time.sleep((ts - now).total_seconds())
        value = get_realtime(feature, logger=logger)
        if value is None:
          logger.warning("Actual value is None")
          continue
        write_result = writer.write_inference(
            dataset_name=dataset.name, task_type=str(dataset.task_type),
            timestamp=DTEncoder.to_utc(ts),
            results={col: results.forecast[col][idx] for col in model_names},
            actual=value)
        logger.info(f"Write Result : {write_result}")
  except KeyboardInterrupt:
    return
  except Exception as e:
    logger.error(str(e))
    logger.error(traceback.format_exc())


def main() -> None:
  """Worker entry point (pm2 runs this file with the dataset name as argv)."""
  service_worker.run_from_argv(auto_inference_write_loop)


if __name__ == "__main__":
  main()
