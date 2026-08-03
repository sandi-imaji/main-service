"""
Training orchestration (service layer).

Owns everything around producing/updating models that is NOT pure ML compute:
pulling data, creating/updating ModelML rows, top_model / status bookkeeping,
and committing. The ML compute stays in the pure core (via contracts): every
task-type core exposes `compare_models` / `train_one` (and supervised also
`retrain_models` for finetune).
"""
from __future__ import annotations
import os, datetime, time, pandas as pd
from uuid import uuid4

from app.database.DB import Dataset, ModelML
from app.database.base import get_db_session
from app.database.schemas import MetaModelML, PreprocessingSchema, StatusProcess,TaskType
from app.exceptions import ValidationException
from app.core.contracts import AnomalyTrainRequest, ClusteringTrainRequest, SupervisedTrainRequest, TimeSeriesTrainRequest, TrainRequest, TrainedModel
from app.config import Config
from app.pull import pull_history, PullDate
from app.helpers import DTEncoder
from app.logger import Logger

def build_train_request(task_type:TaskType,**kwargs) -> TrainRequest :
  if task_type.is_supervised(): return SupervisedTrainRequest(**kwargs)
  elif task_type.is_clustering(): return ClusteringTrainRequest(**kwargs)
  elif task_type.is_timeseries(): return TimeSeriesTrainRequest(**kwargs)
  else: return AnomalyTrainRequest(**kwargs)

# ---------------------------------------------------------------------------
# Train / find-top orchestration
#
# The core computes (pure, via contracts) and returns TrainedModel descriptors;
# this layer owns all persistence: creating ModelML rows, top_model / status,
# and committing.
# ---------------------------------------------------------------------------

def _persist_trained_model(dataset: Dataset, tm: TrainedModel, db) -> ModelML:
  """Turn a core TrainedModel into a persisted ModelML row on the dataset."""
  model = ModelML(
      name=f"{tm.algorithm}-{uuid4().hex[:8]}",
      algorithm=tm.algorithm,
      is_active=True,
      evaluation=tm.evaluation,
      meta=MetaModelML(sizeof=tm.size),
      status=StatusProcess.SUCCESS_TRAIN,
      path=os.path.relpath(tm.path, Config.dir),
  )
  dataset.models.append(model)
  return model


def find_top_models(dataset: Dataset, n_top: int, db, logger):
  """Find & persist the top-N models for a dataset."""
  n_top = max(1, n_top)                         # never request 0 models
  logger.info(f"Find Top Model : {dataset.name} | n_top : {n_top}")
  try:
    core = dataset.task_type.core()
    tic = time.monotonic()
    results = core.compare_models(dataset.to_train_request(n_top), logger)
    # persist and use the newly-trained best (models list may already hold older
    # rows, so `dataset.models[0]` is not necessarily this run's best).
    persisted = [_persist_trained_model(dataset, tm, db) for tm in results]
    dataset.top_model = persisted[0].name
    dataset.status = StatusProcess.IDLE
    dataset.meta = dataset.meta.model_copy(update={"train_time": time.monotonic() - tic})
    db.commit()
    logger.info("Compare Models Finished!")
  except Exception as e:
    logger.error(str(e))
    dataset.status = StatusProcess.ERROR_TRAIN
    db.commit()


def train_model(dataset: Dataset, algorithm: str, db, use_this=False):
  dataset.check_integrity()
  core = dataset.task_type.core()
  logger = Logger(dataset.name)
  logger.info("Training Model ...")
  if algorithm not in dataset.task_type.algorithms().keys():
    raise ValidationException(f"Algorithm '{algorithm}' is not available for {dataset.task_type}")
  if not core: raise ValidationException(f"Task type '{dataset.task_type}' is not supported for training")

  tm = core.train_one(dataset.to_train_request(), algorithm, logger)
  model = _persist_trained_model(dataset, tm, db)
  if use_this or not dataset.top_model:
    dataset.top_model = model.name
  dataset.status = StatusProcess.IDLE
  db.commit()


# ---------------------------------------------------------------------------
# Finetune orchestration
#
# Owns everything that is NOT pure ML: pulling fresh data, assembling the new
# training window, persistence and dataset bookkeeping. The retraining compute
# stays in the pure core (`core.retrain_models`). Runs in its own session
# (worker context), so it re-fetches the dataset by name — not a request one.
# ---------------------------------------------------------------------------

def _assemble_finetune_dataframe(dataset: Dataset, logger):
  """Pull the fresh window, merge it with the stored data, and truncate.

  Returns (filtered_df, start_date, now). No DB writes here.
  """
  now = DTEncoder.now()
  # meta itu objek MetaDataset (bukan dict) -> akses via atribut.
  current_dt = datetime.datetime.fromisoformat(dataset.meta.current_dt) if dataset.meta.current_dt else now
  logger.info(f"Now :{now} | current_dt : {current_dt}")
  current_dt = current_dt + datetime.timedelta(minutes=dataset.interval)

  preprocessing = dataset.preprocessing or PreprocessingSchema()
  n_out = preprocessing.interval_finetune // 2

  df_before = dataset.open_dataframe()
  # copy supaya .remove tidak memutasi list kolom di objek meta.
  columns = [c for c in dataset.meta.columns if c != "dt"]

  start_dt_pull = PullDate.from_dt(current_dt)
  end_dt_pull = PullDate.from_dt(now)
  end_dt_pull.time_start = "00:00:00"

  # row_id sudah tersimpan di meta.tags -> skip lookup get_row_id (~7s/kolom).
  df_new = pull_history(columns=columns, start_date=start_dt_pull, end_date=end_dt_pull,
                        logger=logger, preprocessing=preprocessing, tags=dataset.meta.tags)
  all_data = pd.concat([df_before, df_new]).sort_values(by="dt").reset_index(drop=True)

  start_date = DTEncoder.str_to_dt(dataset.start_date) + datetime.timedelta(days=n_out)
  filtered_df = all_data[(all_data["dt"] >= pd.Timestamp(start_date.date(), tz="UTC"))
                         & (all_data["dt"] <= pd.Timestamp(now.date(), tz="UTC"))].drop_duplicates(subset=["dt"])
  return filtered_df, start_date, now


def finetune(dataset: Dataset, logger):
  """Retrain a dataset's existing models on freshly-pulled data.

  Only cores migrated to the pure-contract `retrain_models` are driven here (the
  service owns persistence). Cores not yet migrated run their own finetune inside
  their own worker, so there is deliberately no generic `core.finetune` fallback:
  `retrain_models` is a distinct name that legacy methods can't collide with.
  """
  core = dataset.task_type.core()
  if not hasattr(core, "retrain_models"):      # core not migrated to contract finetune
    logger.warning(f"service finetune not supported for {dataset.task_type} (core not migrated)")
    return

  logger.info("start finetune ...")
  try:
    with get_db_session() as db:
      ds = Dataset.get_by_name(dataset.name, db)
      status_before = ds.status
      ds.status = StatusProcess.RUNNING_FINETUNE
      db.commit()

      filtered_df, start_date, now = _assemble_finetune_dataframe(ds, logger)
      preprocessing = ds.preprocessing or PreprocessingSchema()
      req = SupervisedTrainRequest(
          df=filtered_df, target=ds.target,
          preprocessing=preprocessing.to_args_pycaret(),
          out_dir=ds.path / "top_model", task=ds.task_type.name)

      by_algo = {m.algorithm: m for m in ds.models}
      for tm in core.retrain_models(req, list(by_algo.keys()), logger):   # pure compute
        m = by_algo[tm.algorithm]
        m.evaluation = tm.evaluation
        m.meta = m.meta.model_copy(update={
            "last_update": DTEncoder.now().isoformat(),
            "sizeof": tm.size,
        })

      ds.start_date = DTEncoder.dt_to_str(start_date)
      ds.end_date = DTEncoder.dt_to_str(now)
      ds.meta = ds.meta.model_copy(update={
          "current_dt": now.isoformat(),
          "last_update": now.isoformat(),
      })
      ds.status = status_before
      db.commit()

      filtered_df.to_csv(ds.path / "data.csv", index=False)
      logger.info(f"Dataset {ds.name} Successfully Finetune!")
  except Exception as e:
    logger.error(f"Error Finetune : {e}")

if __name__ == "__main__":
  from app.database.base import get_session
  dataset_name = "Clustering-10de2b03"
  logger = Logger("test")
  dataset = Dataset.get_by_name(name=dataset_name,db=next(get_session()))
  print(_assemble_finetune_dataframe(dataset,logger=logger))
  

