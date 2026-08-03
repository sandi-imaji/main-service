"""Back-fill nilai aktual ke record inference di InfluxDB.

Prediksi ditulis lebih dulu (untuk timestamp masa depan); nilai nyatanya baru
ada belakangan. Modul ini yang menjodohkan keduanya berdasarkan timestamp.
"""
import datetime

import pandas as pd

from app.database.DB import Dataset
from app.database.historicalDB import get_influx_storage
from app.helpers import DTEncoder
from app.pull import PullDate, get_history
from app.services.timeseries.horizon import _current_dt


def get_actual_save(dataset: Dataset, logger) -> None:
  """Fetch the realised values for the current forecast window and write them
  back onto the matching inference records in InfluxDB."""
  influx = get_influx_storage()
  interval = int(dataset.interval)
  n = int(dataset.target)
  current_dt = _current_dt(dataset)

  end_dt = DTEncoder.get_end_datetime(last_dt=current_dt, interval_minutes=interval, n=n)
  curr_pull_dt = PullDate.from_dt(current_dt + datetime.timedelta(minutes=5))
  end_time = f"{end_dt.hour}:{end_dt.minute}:{end_dt.second}"

  data_actual = get_history(tagname=dataset.features[0],
                            current_date=curr_pull_dt.current_date,
                            time_start=curr_pull_dt.time_start, time_end=end_time,
                            logger=logger, to_dataframe=True)

  if not isinstance(data_actual, pd.DataFrame):
    logger.warning("Data Actual is empty !")
    return
  logger.info(f"Total Data Actual Time Series : {len(data_actual)}")

  data_influx = list(reversed(influx.query_inference(dataset_name=dataset.name)))
  logger.info(f"Total Data Time Series from InfluxDB: {len(data_influx)}")

  value_name = dataset.features[0]
  for idx, d in enumerate(data_influx):
    actual_value = data_actual.iloc[idx][value_name]
    actual_timestamp = data_actual.iloc[idx]["dt"]
    if actual_timestamp.isoformat() == d["timestamp"]:
      status = influx.write_inference(dataset_name=dataset.name,
                                      task_type=str(dataset.task_type),
                                      timestamp=DTEncoder.to_utc(actual_timestamp),
                                      results=d["results"], actual=actual_value)
      if status: logger.info(f"write actual in data inference ... [{idx + 1}]")


def get_actual_save_all(dataset: Dataset, logger) -> None:
  """Back-fill actuals for every past inference record of this dataset."""
  influx = get_influx_storage()
  data_influx = influx.query_inference(dataset.name, task_type=str(dataset.task_type))
  feature_name = dataset.features[0]
  now = DTEncoder.now()
  timestamps = [d["timestamp"] for d in data_influx
                if datetime.datetime.fromisoformat(d["timestamp"]) < now]
  if not timestamps:
    logger.warning("No past inference records to back-fill")
    return
  end_dt = datetime.datetime.fromisoformat(timestamps[0])
  start_dt = datetime.datetime.fromisoformat(timestamps[-1])
  logger.info(f"Start Date : {start_dt}")
  logger.info(f"End Date : {end_dt}")

  curr_pull_dt = PullDate.from_dt(start_dt)
  end_time = f"{end_dt.hour}:{end_dt.minute}:{end_dt.second}"

  data_actual = get_history(tagname=feature_name, current_date=curr_pull_dt.current_date,
                            time_start=curr_pull_dt.time_start, time_end=end_time,
                            logger=logger, to_dataframe=True)

  if not isinstance(data_actual, pd.DataFrame) or data_actual.empty:
    logger.warning("Data Actual is empty !")
    return

  data_influx = list(reversed(data_influx))
  for idx, d in enumerate(data_influx):
    actual_value = data_actual.iloc[idx][feature_name]
    d_timestamp = datetime.datetime.fromisoformat(d["timestamp"])
    actual_timestamp = data_actual.iloc[idx]["dt"]
    if DTEncoder.compare(actual_timestamp, d_timestamp):
      status = influx.write_inference(dataset_name=dataset.name,
                                      task_type=str(dataset.task_type),
                                      timestamp=DTEncoder.to_utc(actual_timestamp),
                                      results=d["results"], actual=actual_value)
      if status: logger.info(f"write actual in data inference ... [{idx + 1}]")
      else: logger.info(f"write actual in data inference is failed [{idx + 1}]")
