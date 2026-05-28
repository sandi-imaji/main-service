"""
Time Series Forecasting Module
Optimized for performance with model caching and memory management.
Phase 3: Async support dengan unified cache
"""

import pandas as pd, os, datetime,time,traceback,sys
from typing import List, Optional, Dict, Any
from contextlib import closing
from pathlib import Path

if __name__ == "__main__":
  # Get the directory containing this file: app/core/
  current_dir = Path(__file__).parent
  # Go up to parent: app/
  app_dir = current_dir.parent
  # Go up again to root project
  root_dir = app_dir.parent
  # Add root to sys.path if not already there
  if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.database.schemas import (
    MetaDataset,
    StatusProcess,
    MetaModel,
)
from app.helpers import DTEncoder
from app.database.influx import InfluxDBStorage, get_influx_storage
from pydantic import BaseModel

from app.pull import (
    get_history,
    pull_history,
    pulling,PullDate,get_realtime
)
from app.utils.model_cache import get_timeseries_cache
from app.database.orm import Dataset, ModelML
from app.database.db import get_session
from pycaret import time_series as mod
from app.config import Config
from app.logger import Logger,LoggerNone
from app.core.base import BaseMLCore, InferenceMixin, ResultSchema
from uuid import uuid4


class TimeSeriesResultSchema(BaseModel, ResultSchema):
  timestamps: List[datetime.datetime]
  forecast: Dict[str, List[int | float]]
  dataset_name: str
  task_type:str = "TimeSeries"
  is_valid: bool = False

  @property
  def timestamps_(self) -> List[str]: return [str(i) for i in self.timestamps]

  def write_to_influx(self,influx:InfluxDBStorage,logger):
    logger.info(f"[Efficient] Got InfluxDB storage: {influx.bucket}")
    model_names = list(self.forecast.keys())
    for idx,timestamp in enumerate(self.timestamps):
      result = {col:self.forecast[col][idx] for col in model_names}
      influx.write_inference(
        dataset_name = self.dataset_name,
        task_type = "TimeSeries",
        timestamp=DTEncoder.to_utc(timestamp),
        results = result,
        features = {}
      )
      logger.info(f"[Efficient] Write result: {timestamp}")
    logger.info(f"Total Saved : {len(self.timestamps)}")


# =============================================================================
# Time Series Class
# =============================================================================


class TimeSeries(BaseMLCore, InferenceMixin):
  """
  Time series forecasting operations with optimizations:
  - Model caching for faster repeated forecasting
  - Memory-efficient data handling
  - Incremental model updates
  """
  _forecast : Optional[TimeSeriesResultSchema] = None
  TOP_ALGO: List[str] = [
      "catboost_cds_dt",
      "xgboost_cds_dt",
      "lightgbm_cds_dt",
      "gbr_cds_dt",
      "huber_cds_dt",
      "ets",
  ]
  EXP = 3
  _task_type = "timeseries"

  @classmethod
  def get_cache(cls):
    """Get the model cache for time series forecasting."""
    return get_timeseries_cache()

  @staticmethod
  def is_expired(dataset: Dataset, logger) -> bool:
    """
    Check if forecasts have expired and need update.

    Args:
        dataset: Dataset object
        logger: Logger instance

    Returns:
        True if forecasts are expired
    """
    now = DTEncoder.now()
    meta = dataset.meta
    current_dt = meta.get("current_dt", "")
    if not current_dt:
      df = dataset.open_dataframe()
      current_dt = df.iloc[-1]["dt"]
      current_dt = current_dt.to_pydatetime()
    else: current_dt = datetime.datetime.fromisoformat(current_dt)

    # last_preds_dt = current_dt + datetime.timedelta(
    #     minutes=dataset.interval * int(dataset.target)
    # )
    end_date = DTEncoder.get_end_datetime(current_dt,dataset.interval,int(dataset.target))

    logger.info(f"Now Timestamp : {now}")
    logger.info(f"Last Preds Timestamp : {end_date}")

    return now.timestamp() > end_date.timestamp()

  @staticmethod
  def inference(dataset: Dataset, logger) -> TimeSeriesResultSchema:
    """
    Run forecast inference (a.k.a. forecast).

    Args:
      dataset: Dataset object
      logger: Logger instance

    Returns: Dictionary with forecasted values
    """

    logger.info("Forecast models :")
    fh = int(dataset.target)
    current_dt = dataset.meta.get("current_dt", "")
    if not current_dt:
      df = dataset.open_dataframe()
      current_dt = df.iloc[-1]["dt"].to_pydatetime()
    else: current_dt = datetime.datetime.fromisoformat(current_dt)

    futures_date = DTEncoder.generate_dt(n=fh, interval_minutes=dataset.interval, last_dt=current_dt)

    result = {}
    try:
      for m in dataset.models:
        modelpath = Config.dir / m.path
        model = TimeSeries._load_model_cached(dataset.task_type.module(), str(modelpath), logger)

        res = mod.predict_model(model, fh, verbose=Config.verbose.pycaret)["y_pred"].values.tolist()
        result[m.algorithm] = res
        logger.info(f"Inference {m.algorithm} success!")
      result = TimeSeriesResultSchema(
          timestamps=futures_date,
          forecast=result,
          dataset_name=dataset.name,
          is_valid=True,
      )
      logger.info("Forecast was successful!")
      TimeSeries._forecast = result
      return result
    except:
      return TimeSeriesResultSchema(
        timestamps = futures_date,forecast={},dataset_name=dataset_name,is_valid=False
      )

  @staticmethod
  def auto_inference(dataset: Dataset, logger) -> TimeSeriesResultSchema:
    """
    Automated inference with actual value fetching.

    Args:
      dataset: Dataset object
      logger: Logger instance

    Returns: Dictionary with forecasts
    """
    if TimeSeries.is_expired(dataset,logger):
      TimeSeries.get_actual_save(dataset,logger)
      TimeSeries.retrain(dataset,logger)
      return TimeSeries.inference(dataset, logger)
    else: return TimeSeriesResultSchema(timestamps = [],forecast={},dataset_name=dataset_name,is_valid=False)

  @staticmethod
  def get_actual_save_all(dataset:Dataset,logger):
    import pprint
    influx = get_influx_storage()
    data_influx = influx.query_inference(dataset.name,task_type=str(dataset.task_type))
    feature_name = dataset.features[0]
    now = DTEncoder.now()
    timestamps = [d["timestamp"] for d in data_influx if datetime.datetime.fromisoformat(d['timestamp']) < now ]
    end_dt,start_dt = datetime.datetime.fromisoformat(timestamps[0]),datetime.datetime.fromisoformat(timestamps[-1])
    logger.info(f"Start Date : {start_dt}")
    logger.info(f"End Date : {end_dt}")

    curr_pull_dt = PullDate.from_dt(start_dt)
    end_time = f"{end_dt.hour}:{end_dt.minute}:{end_dt.second}"

    data_actual = get_history(tagname = dataset.features[0],current_date=curr_pull_dt.current_date,
                              time_start=curr_pull_dt.time_start,time_end=end_time,logger=logger,to_dataframe=True)

    if not isinstance(data_actual,pd.DataFrame):
      logger.warning("Data Actual is empty !")
      return 
    if data_actual.empty:
      return 

    # if len(data_actual) != len(timestamps):
    #   raise ValueError(f"length data actual [{len(data_actual)}] != length data from database [{len(timestamps)}]")
    pprint.pprint(timestamps)
    print(data_actual)

    data_influx = list(reversed(data_influx))
    for idx,d in enumerate(data_influx):
      actual_value = data_actual.iloc[idx][feature_name]
      d_timestamp = datetime.datetime.fromisoformat(d['timestamp'])
      actual_timestamp = data_actual.iloc[idx]["dt"]
      if DTEncoder.compare(actual_timestamp,d_timestamp):
      # if abs(actual_timestamp.timestamp() - d_timestamp.timestamp()) < 1.5:
        status = influx.write_inference(dataset_name=dataset.name,
                               task_type=str(dataset.task_type),
                               timestamp=DTEncoder.to_utc(actual_timestamp),
                               results=d['results'],
                               actual=actual_value)
        if status: logger.info(f"write actual in data inference ... [{idx+1}]")
        else: logger.info(f"write actual in data inference is failed [{idx+1}]")



  @staticmethod
  def get_actual_save(dataset:Dataset,logger) -> None:
    influx = get_influx_storage()
    current_dt = dataset.meta["current_dt"]
    interval = int(dataset.interval)
    n = int(dataset.target)
    if not current_dt: current_dt = dataset.open_dataframe().iloc[-1]["dt"]
    else: current_dt = datetime.datetime.fromisoformat(current_dt)

    end_dt = DTEncoder.get_end_datetime(last_dt=current_dt,interval_minutes=interval,
                                        n=n)
    # Konversi ke PullDate
    curr_pull_dt  = PullDate.from_dt(current_dt+datetime.timedelta(minutes=5))
    end_time = f"{end_dt.hour}:{end_dt.minute}:{end_dt.second}"

    data_actual = get_history(tagname=dataset.features[0],
                       current_date=curr_pull_dt.current_date,
                       time_start=curr_pull_dt.time_start,time_end=end_time,
                       logger=logger,to_dataframe=True)

    logger.info(f"Total Data Actual Time Series : {len(data_actual)}")

    if not isinstance(data_actual,pd.DataFrame):
      logger.warning("Data Actual is empty !")
      return 
    # Konversi ke UTC
    current_dt  = DTEncoder.to_utc(current_dt)
    end_dt = DTEncoder.to_utc(end_dt)+datetime.timedelta(minutes=interval)

    # get data from influx
    data_influx = influx.query_inference(dataset_name=dataset.name)
    data_influx = list(reversed(data_influx))
    logger.info(f"Total Data Time Series from InfluxDB: {len(data_influx)}")

    # Replace Data Inference
    value_name = dataset.features[0]
    dataset_name  = dataset.name
    task_type = dataset.task_type
    for idx,d in enumerate(data_influx):
      actual_value = data_actual.iloc[idx][value_name]
      actual_timestamp = data_actual.iloc[idx]["dt"]
      if actual_timestamp.isoformat() == d["timestamp"]:
        status = influx.write_inference(dataset_name=dataset_name,
                               task_type=str(task_type),
                               timestamp=DTEncoder.to_utc(actual_timestamp),
                               results=d['results'],
                               actual=actual_value)
        if status: logger.info(f"write actual in data inference ... [{idx+1}]")

  @staticmethod
  def auto_inference_loop(dataset:Dataset,logger) -> None:
    logger.info("start auto inference write loop ...")
    try:
      if TimeSeries.is_expired(dataset,logger): TimeSeries.retrain(dataset,logger)
      results = TimeSeries.inference(dataset,logger)
      feature = dataset.features[0]
      model_names = list(results.forecast.keys())
      with InfluxDBStorage() as writer:
        results.write_to_influx(writer,logger)
        for idx,ts in enumerate(results.timestamps):
          logger.info(f"auto inference - {idx+1}")
          now = DTEncoder.now()
          if now > ts:
            continue
          delay = (ts - now).total_seconds()
          time.sleep(delay)
          value = get_realtime(feature,logger=logger)
          if value is None: logger.warning("Actual value is None")
          else:
            write_result = writer.write_inference(
              dataset_name=dataset.name,
              task_type = str(dataset.task_type),
              timestamp = DTEncoder.to_utc(ts),
              results = {col:results.forecast[col][idx] for col in model_names},
              actual = value
            )
            logger.info(f"Write Result : {write_result}")
    except Exception as e:
      logger.error(str(e))
      logger.error(traceback.format_exc())
      return None
    except KeyboardInterrupt: return None

  @staticmethod
  def auto_inference_write_loop(dataset:Dataset,logger) -> None:
    logger.info("start auto inference write loop ...")
    try:
      if TimeSeries.is_expired(dataset,logger): TimeSeries.retrain(dataset,logger)
      results = TimeSeries.inference(dataset,logger)
      feature = dataset.features[0]
      model_names = list(results.forecast.keys())
      with InfluxDBStorage() as writer:
        results.write_to_influx(writer,logger)
        for idx,ts in enumerate(results.timestamps):
          logger.info(f"auto inference - {idx+1}")
          now = DTEncoder.now()
          if now > ts:
            continue
          delay = (ts - now).total_seconds()
          time.sleep(delay)
          value = get_realtime(feature,logger=logger)
          if value is not None:
            write_result = writer.write_inference(
              dataset_name=dataset.name,
              task_type = str(dataset.task_type),
              timestamp = DTEncoder.to_utc(ts),
              results = {col:results.forecast[col][idx] for col in model_names},
              actual = value
            )
            logger.info(f"Write Result : {write_result}")
          else: logger.info("Forecasted ...")
          time.sleep(dataset.interval * 60)
    except Exception as e:
      logger.error(str(e))
      logger.error(traceback.format_exc())
      return None
    except KeyboardInterrupt: return None

  @staticmethod
  def retrain(dataset: Dataset, logger):
    if not TimeSeries.is_expired(dataset, logger): return
    n_rows = dataset.meta.get("n_rows", 1000)
    interval = dataset.interval
    current_dt = dataset.meta.get("current_dt", "")
    if not current_dt:
      df = dataset.open_dataframe()
      current_dt = df.iloc[-1]["dt"]
      current_dt = current_dt.to_pydatetime()
    else:
      current_dt = datetime.datetime.fromisoformat(current_dt)

    start_dt = DTEncoder.get_start_datetime(DTEncoder.now(), interval, n_rows)
    start_dt = DTEncoder.dt_to_str(start_dt)
    end_dt = DTEncoder.now_str()
    dataset_name = dataset.name

    with closing(next(get_session())) as db:
      # Reload dataset dalam session baru
      dataset = Dataset.get_by_name(dataset_name, db)
      if not dataset:
        logger.error(f"Dataset {dataset_name} not found in database")
        return
      meta = dataset.meta
      meta["current_dt"] = end_dt
      dataset.start_date = start_dt
      dataset.end_date = end_dt
      dataset.meta = meta
      dataset.save(db)
      db.commit()
      pulling(dataset.name)
      df = dataset.open_dataframe()[dataset.features]
      mod.setup(df, fh=int(dataset.target), verbose=Config.verbose.pycaret)
      for m in dataset.models:
        tic = time.monotonic()
        model = mod.create_model(m.algorithm, verbose=Config.verbose.pycaret)
        metric = mod.pull().loc["Mean"].to_dict()
        model = mod.finalize_model(model)
        mod.save_model(model, str(Config.dir / m.path))
        toc = time.monotonic()

        meta = MetaModel(
            created_by="Anonymous",
            created_at=DTEncoder.now().isoformat(),
            train_time=toc - tic,
            size_of=os.path.getsize(f"{Config.dir}/{m.path}.pkl"),
            notes="",
        )
        m.meta = meta.model_dump()
        m.evaluation = metric
        m.save(db)
        db.commit()

        logger.info(f"Model : {m.algorithm} Successfully")
        logger.info(f"Train Model {m.algorithm} Finished!")

      dataset.status = StatusProcess.IDLE
      dataset.save(db)
      db.commit()

    # Clear cache
    get_timeseries_cache().clear()

  @staticmethod
  async def finetune(dataset_name: str):
    """
    Finetune models with new data.
    Args:
        dataset_name: Name of the dataset
    """
    with closing(next(get_session())) as db:
      dataset = Dataset.get_by_name(dataset_name, db)

      if not dataset: raise ValueError("Dataset is not found!")

      logger = Logger(dataset_name)
      logger.info(f"Finetune {dataset.name} start ...")

      # Get current datetime from meta
      meta = dataset.meta
      if isinstance(meta, dict):
        current_dt = meta.get("current_dt", "")
      else:
        current_dt = getattr(meta, "current_dt", "")

      df = dataset.open_dataframe()
      current_dt_df = df.iloc[-1]["dt"]

      if not current_dt: current_dt = current_dt_df
      else: current_dt = datetime.datetime.fromisoformat(str(current_dt))

      if str(current_dt) != str(current_dt_df):
        raise ValueError(
            "Current Date in Database != Current Date in Dataframe!"
        )

      # Clean up old results
      predPath = Config.dir / "storages" / dataset.name / "results/results.bin"
      actualPath = Config.dir / "storages" / dataset.name / "results/actuals.bin"

      if os.path.exists(predPath):
        os.remove(predPath)
      if os.path.exists(actualPath):
        os.remove(actualPath)

      now_dt = DTEncoder.now()
      preds_dt_first = current_dt + \
          datetime.timedelta(minutes=dataset.interval)
      preds_dt_last = current_dt + datetime.timedelta(
          minutes=dataset.interval * int(dataset.target)
      )

      logger.info(f"Current Date : {current_dt}")
      logger.info(f"Preds First Date : {preds_dt_first}")
      logger.info(f"Preds Last Date : {preds_dt_last}")

      diff = now_dt - preds_dt_last
      threshold = (int(dataset.target) / 2) * 60

      if diff.total_seconds() < 0:
        logger.info(
            f"Dataset below threshold, finetune only when above {threshold} seconds"
        )
        return

      logger.info("Pulling new data...")

      start_date = DTEncoder.dt_to_str(preds_dt_first)
      end_date = DTEncoder.dt_to_str(now_dt)
      time_start = f"{preds_dt_first.hour}:{preds_dt_first.minute}:00"
      time_end = f"{now_dt.hour}:{preds_dt_first.minute}:00"

      start_date_obj = PullDate(current_date=start_date, time_start=time_start)
      end_date_obj = PullDate(current_date=end_date, time_start="00:00:00")

      logger.info(f"Start Date : {start_date_obj} | End Date : {end_date_obj}")

      actuals = pull_history(
          columns=dataset.features,
          start_date=start_date_obj,
          end_date=end_date_obj,
          logger=logger,
      )

      if isinstance(actuals, pd.DataFrame):
        if actuals.empty:
          logger.error("actuals is empty")
          return

        actuals.columns = actuals.columns.astype(str)
        df.columns = df.columns.astype(str)
        actuals.reset_index(drop=True, inplace=True)

        df_new = pd.concat([df, actuals], axis=0)
        df_new.reset_index(drop=True, inplace=True)
        df_new = df_new.sort_values("dt", ascending=True)

        fpath = Config.dir / "storages" / dataset.name / "data.csv"
        current_dt = df_new.iloc[-1]["dt"]

        dataset.end_date = DTEncoder.dt_to_str(current_dt)

        # Update metadata
        if isinstance(dataset.meta, dict):
          meta_obj = MetaDataset(**dataset.meta)
        else:
          meta_obj = dataset.meta

        meta_obj.last_update = datetime.datetime.now().isoformat()
        meta_obj.current_dt = str(df_new.iloc[-1]["dt"])
        dataset.meta = meta_obj.model_dump()

        # Retrain with recent data
        fh = int(dataset.target)
        n_select = fh * 10
        df_new = df_new.tail(n_select)
        X = df_new.drop(columns="dt")

        mod.setup(X, fh=fh, verbose=Config.verbose.pycaret)

        cache = get_timeseries_cache()

        for m in dataset.models:
          model = mod.create_model(m.algorithm, verbose=Config.verbose.pycaret)
          metric = mod.pull().loc["Mean"].to_dict()
          model = mod.finalize_model(model)

          path_model = f"storages/{dataset.name}/top_model/{m.algorithm}"

          if os.path.exists(Config.dir / path_model):
            os.remove(Config.dir / path_model)
            logger.info(f"Delete Model Previous : {m.algorithm}")
            cache.invalidate(str(Config.dir / path_model))

          mod.save_model(model, str(Config.dir / path_model))
          logger.info(
              f"Save Model Update: {m.algorithm} at {Config.dir / path_model}"
          )

          model_meta = MetaModel(
              created_by="Anonymous",
              last_update=datetime.datetime.now().isoformat(),
              created_at=m.meta.get("created_at", "")
              if isinstance(m.meta, dict)
              else getattr(m.meta, "created_at", ""),
              size_of=os.path.getsize(f"{Config.dir}/{path_model}.pkl"),
              notes=f"Update {datetime.datetime.now().isoformat()}",
          )

          m.evaluation = metric
          m.meta = model_meta.model_dump()
          logger.info(f"Model Update : {m.algorithm} Successfully")

        db.commit()
        df_new.to_csv(str(fpath), index=False)
        logger.info(f"Dataset {dataset.name} Successfully Updated!")
      else: logger.error(f"actuals is not dataframe but {type(actuals)}")

  @staticmethod
  def find_top_model(dataset_name: str, n_top: int, logger):
    """
    Find and train top N time series models.

    Args:
      dataset_name: Name of the dataset
      n_top: Number of top models to select
      logger: Logger instance
    """
    with closing(next(get_session())) as db:
      tic = time.monotonic()
      dataset = Dataset.get_by_name(dataset_name, db)

      if not dataset: raise ValueError("Dataset is not found!")

      try:
        df = dataset.open_dataframe()[dataset.features]
        fh = int(dataset.target)

        mod.setup(df, fh=fh, verbose=Config.verbose.pycaret, ignore_features=["dt"])

        top_model = mod.compare_models(
            n_select=n_top, verbose=Config.verbose.pycaret, include=TimeSeries.TOP_ALGO
        )

        metrics = mod.pull().to_dict("index")
        keys = list(metrics.keys())

        for i in range(n_top):
          model_name = f"{keys[i]}-{str(uuid4())[:8]}"
          evaluation = metrics[keys[i]]
          del evaluation["Model"]

          path_model = f"storages/{dataset.name}/top_model/{keys[i]}"
          model = mod.finalize_model(top_model[i])
          mod.save_model(model, path_model)

          meta = MetaModel(
              created_by="Anonymous",
              created_at=datetime.datetime.now().isoformat(),
              size_of=os.path.getsize(f"{Config.dir / path_model}.pkl"),
              notes="",
          )

          q_model = ModelML(
              name=model_name,
              algorithm=keys[i],
              is_active=True,
              evaluation=evaluation,
              meta=meta.model_dump(),
              status=StatusProcess.SUCCESS_TRAIN,
              path=path_model,
          )

          dataset.models.append(q_model)
          logger.info(f"Model : {keys[i]} Successfully!")

        # Update metadata
        if isinstance(dataset.meta, dict):
          current_meta = MetaDataset(**dataset.meta)
        else:
          current_meta = dataset.meta

        current_meta.train_time = time.monotonic() - tic
        dataset.meta = current_meta.model_dump()
        dataset.top_model = dataset.models[0].name
        dataset.status = StatusProcess.IDLE

        db.commit()
        logger.info("Compare Models Finished!")

        # Clear cache after training
        get_timeseries_cache().clear()

      except Exception as e:
        logger.error(str(e))
        db.rollback()

  @staticmethod
  def train(dataset: Dataset, algorithm: str, logger):
    """
    Train a specific time series algorithm.

    Args:
        dataset: Dataset object
        algorithm: Algorithm identifier
        logger: Logger instance
    """
    df = dataset.open_dataframe()
    mod.setup(
        df[dataset.features],
        fh=int(dataset.target),
        verbose=Config.verbose.pycaret,
        ignore_features=["dt"],
    )

    model = mod.create_model(algorithm, verbose=Config.verbose.pycaret)
    metric = mod.pull().loc["Mean"].to_dict()
    model_name = f"{algorithm}-{str(uuid4())[:8]}"
    path_model = f"storages/{dataset.name}/top_model/{algorithm}"

    model = mod.finalize_model(model)
    mod.save_model(model, str(Config.dir / path_model))

    logger.info(f"Save Model at {Config.dir / path_model}")

    meta = MetaModel(
        created_by="Anonymous",
        created_at=datetime.datetime.now().isoformat(),
        train_time=0.0,
        size_of=os.path.getsize(f"{Config.dir}/{path_model}.pkl"),
        notes="",
    )

    q_model = ModelML(
        name=model_name,
        algorithm=algorithm,
        is_active=True,
        evaluation=metric,
        meta=meta.model_dump(),
        status=StatusProcess.SUCCESS_TRAIN,
        path=path_model,
    )

    dataset.models.append(q_model)
    logger.info(f"Model : {algorithm} Successfully")
    dataset.status = StatusProcess.IDLE
    logger.info(f"Train Model {algorithm} Finished!")

    # Clear cache
    get_timeseries_cache().clear()


  @staticmethod
  def get_cache_stats() -> Dict[str, Any]:
    """Get model cache statistics."""
    stats = get_timeseries_cache().get_stats()
    return {
        "hits": stats.hits,
        "misses": stats.misses,
        "size": stats.size,
        "max_size": stats.max_size,
        "hit_rate": stats.hit_rate,
    }


def main():
  if len(sys.argv) > 1:
    dataset_name = sys.argv[1]
    db = next(get_session())
    dataset = Dataset.get_by_name(dataset_name,db)
    if not dataset: raise ValueError("Dataset is not found!")
    logger = LoggerNone(dataset_name)
    TimeSeries.auto_inference_loop(dataset,logger=logger)
  else:
    print(sys.argv)
    print("Dataset name is None")


if __name__ == "__main__":
  # from app.logger import Logger
  # dataset_name = "TimeSeries-222aa49a"
  # db = next(get_session())
  # dataset = Dataset.get_by_name(dataset_name,db)
  # if not dataset: raise ValueError("Dataset is not found!")
  # TimeSeries.retrain(dataset,Logger("test"))
  # print(TimeSeries.is_expired(dataset,Logger("test")))
  # while True:
  #   result = TimeSeries.inference(dataset,Logger("test"))
  #   print(result)

  # influx = get_influx_storage()
  # with closing(next(get_session())) as db:
  #   interval = 5
  #   dataset_name = "TimeSeries-321a3acd"
  #   dataset = Dataset.get_by_name(dataset_name, db)
  #   logger = Logger("test")
  #   if not dataset: raise ValueError("Dataset is not found!")
  # result = TimeSeries.inference(dataset,Logger("test"))
  # with InfluxDBStorage() as writer:
  #   result.write_to_influx(writer,Logger("test"))
  # TimeSeries.get_actual_save_all(dataset,Logger('test'))
  
  # TimeSeries.retrain(dataset,Logger("test"))
  main()

