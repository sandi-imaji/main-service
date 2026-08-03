"""Refit model time-series pada data yang lebih baru.

Dua jalur yang berbeda asal-usul: `retrain` (dipakai worker, menarik ulang
jendela data dari awal) dan `finetune` (dipakai route + stream, menempelkan data
yang baru terealisasi ke data latih lalu melatih ulang).
"""
import datetime
import os

import pandas as pd

from app.config import Config
from app.core.contracts import TimeSeriesTrainRequest
from app.core.time_series import TimeSeries
from app.database.base import get_db_session
from app.database.DB import Dataset
from app.database.schemas import MetaModel, PreprocessingSchema, StatusProcess
from app.helpers import DTEncoder
from app.logger import Logger
from app.pull import PullDate, pull_history, pulling
from app.services.timeseries.horizon import elapsed_horizon_ratio, is_expired
from app.utils.model_cache import get_timeseries_cache


def retrain(dataset: Dataset, logger) -> None:
  """Pull recent data and refit every model of an expired dataset (in a fresh
  DB session, re-fetching by name — safe for background execution).

  Compute stays in the pure core (`TimeSeries.retrain_models`); this function
  owns pulling fresh data, dataset bookkeeping, and persistence.
  """
  if not is_expired(dataset, logger): return
  n_rows = dataset.meta.n_rows or 1000
  interval = dataset.interval

  start_dt = DTEncoder.dt_to_str(DTEncoder.get_start_datetime(DTEncoder.now(), interval, n_rows))
  end_dt = DTEncoder.now_str()
  dataset_name = dataset.name

  with get_db_session() as db:
    dataset = Dataset.get_by_name(dataset_name, db)
    if not dataset:
      logger.error(f"Dataset {dataset_name} not found in database")
      return
    # meta lives on a JSON column: reassign a fresh object so the change is tracked.
    dataset.meta = dataset.meta.model_copy(update={"current_dt": end_dt})
    dataset.start_date = start_dt
    dataset.end_date = end_dt
    dataset.save(db)
    db.commit()

    if not pulling(dataset.name, db):
      logger.error(f"Pull gagal untuk {dataset.name}, retrain dibatalkan")
      return

    by_algo = {m.algorithm: m for m in dataset.models}
    req = TimeSeriesTrainRequest(
        df=dataset.open_dataframe()[dataset.features], preprocessing={},
        out_dir=dataset.path / "top_model", task=dataset.task_type.name,
        fh=int(dataset.target))

    for tm in TimeSeries.retrain_models(req, list(by_algo.keys()), logger):
      m = by_algo[tm.algorithm]
      m.meta = MetaModel(
          created_by="Anonymous",
          created_at=DTEncoder.now().isoformat(),
          sizeof=tm.size,
          notes="",
      ).model_dump()
      m.evaluation = tm.evaluation
      m.save(db)
      db.commit()
      logger.info(f"Train Model {m.algorithm} Finished!")

    dataset.status = StatusProcess.IDLE
    dataset.save(db)
    db.commit()

  get_timeseries_cache().clear()


def finetune(dataset_name: str) -> None:
  """Append newly-realised data to the training set and refit every model.

  Triggered by the `/forecast` endpoint when the dataset is expired. Skips when
  the horizon is not yet `Config.forecast_refresh_ratio` consumed.

  Raises when the fresh pull yields nothing, so the caller counts it as an error
  instead of retrying every tick against a data source that is down.

  Synchronous & blocking (pull + pycaret). Callers must keep it off the event
  loop: FastAPI runs it in a threadpool via BackgroundTasks, and the forecast
  streamer via asyncio.to_thread.
  """
  with get_db_session() as db:
    dataset = Dataset.get_by_name(dataset_name, db)
    if not dataset: raise ValueError("Dataset is not found!")

    logger = Logger(dataset_name)
    logger.info(f"Finetune {dataset.name} start ...")

    df = dataset.open_dataframe()
    current_dt_df = df.iloc[-1]["dt"]
    current_dt = dataset.meta.current_dt
    current_dt = current_dt_df if not current_dt else datetime.datetime.fromisoformat(str(current_dt))

    if str(current_dt) != str(current_dt_df):
      raise ValueError("Current Date in Database != Current Date in Dataframe!")

    now_dt = DTEncoder.now()
    preds_dt_first = current_dt + datetime.timedelta(minutes=dataset.interval)
    preds_dt_last = current_dt + datetime.timedelta(minutes=dataset.interval * int(dataset.target))
    logger.info(f"Current Date : {current_dt}")
    logger.info(f"Preds First Date : {preds_dt_first}")
    logger.info(f"Preds Last Date : {preds_dt_last}")

    # Gate yang SAMA dengan is_expired: kalau keduanya berbeda, pemanggil bisa
    # melihat "expired" sementara finetune menolak jalan -> anchor tak pernah
    # maju dan finetune dipanggil ulang tiap tick.
    ratio = elapsed_horizon_ratio(dataset)
    if ratio < Config.forecast_refresh_ratio:
      logger.info(f"Horizon baru terpakai {ratio:.0%}, finetune menunggu "
                  f"{Config.forecast_refresh_ratio:.0%}")
      return

    logger.info("Pulling new data...")
    start_date = DTEncoder.dt_to_str(preds_dt_first)
    end_date = DTEncoder.dt_to_str(now_dt)
    time_start = f"{preds_dt_first.hour}:{preds_dt_first.minute}:00"
    start_date_obj = PullDate(current_date=start_date, time_start=time_start)
    end_date_obj = PullDate(current_date=end_date, time_start="00:00:00")
    logger.info(f"Start Date : {start_date_obj} | End Date : {end_date_obj}")

    # row_id sudah tersimpan di meta.tags -> skip lookup get_row_id (~7s/kolom).
    actuals = pull_history(columns=dataset.features, start_date=start_date_obj,
                           end_date=end_date_obj,
                           preprocessing=dataset.preprocessing or PreprocessingSchema(),
                           logger=logger, tags=dataset.meta.tags)

    # Pull kosong TIDAK boleh `return` diam-diam: anchor tidak maju, jadi
    # is_expired tetap True dan pemanggil (loop WS, tiap sleep_interval)
    # memanggil finetune lagi — tiap kali dengan pull penuh ke SL. Dilempar
    # supaya kena penghitung consecutive_errors pemanggil dan loop berhenti
    # kalau sumber datanya memang sedang mati.
    if not isinstance(actuals, pd.DataFrame):
      raise ValueError(f"Actuals pull failed: expected a DataFrame, got {type(actuals).__name__}")
    if actuals.empty:
      raise ValueError(f"Pull actual kosong untuk {preds_dt_first} → {now_dt} "
                       f"(kolom: {dataset.features})")

    actuals.columns = actuals.columns.astype(str)
    df.columns = df.columns.astype(str)
    actuals.reset_index(drop=True, inplace=True)

    df_new = pd.concat([df, actuals], axis=0)
    df_new.reset_index(drop=True, inplace=True)
    df_new = df_new.sort_values("dt", ascending=True)

    current_dt = df_new.iloc[-1]["dt"]
    dataset.end_date = DTEncoder.dt_to_str(current_dt)
    dataset.meta = dataset.meta.model_copy(update={
        "last_update": datetime.datetime.now().isoformat(),
        "current_dt": str(df_new.iloc[-1]["dt"]),
    })

    fh = int(dataset.target)
    df_new = df_new.tail(fh * 10)
    X = df_new.drop(columns="dt")

    from pycaret import time_series as mod
    mod.setup(X, fh=fh, verbose=Config.verbose.pycaret)
    cache = get_timeseries_cache()

    for m in dataset.models:
      model = mod.create_model(m.algorithm, verbose=Config.verbose.pycaret)
      metric = mod.pull().loc["Mean"].to_dict()
      model = mod.finalize_model(model)

      path_model = f"storages/{dataset.name}/top_model/{m.algorithm}"
      if os.path.exists(Config.dir / f"{path_model}.pkl"):
        os.remove(Config.dir / f"{path_model}.pkl")
        logger.info(f"Delete Model Previous : {m.algorithm}")
        cache.invalidate(str(Config.dir / path_model))

      mod.save_model(model, str(Config.dir / path_model))
      logger.info(f"Save Model Update: {m.algorithm} at {Config.dir / path_model}")

      created_at = m.meta.created_at if m.meta else ""
      m.meta = MetaModel(
          created_by="Anonymous",
          last_update=datetime.datetime.now().isoformat(),
          created_at=created_at,
          sizeof=os.path.getsize(f"{Config.dir}/{path_model}.pkl"),
          notes=f"Update {datetime.datetime.now().isoformat()}",
      ).model_dump()
      m.evaluation = metric
      logger.info(f"Model Update : {m.algorithm} Successfully")

    db.commit()
    df_new.to_csv(str(Config.dir / "storages" / dataset.name / "data.csv"), index=False)
    logger.info(f"Dataset {dataset.name} Successfully Updated!")
