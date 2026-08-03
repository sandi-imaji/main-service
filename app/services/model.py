from __future__ import annotations
import datetime
from typing import Optional
from app.database.DB import Dataset, ModelML
from collections import Counter
from app.workers.manager import WorkerManager
from app.database.schemas import (
    InferenceRequestSchema,
    StatusProcess,
    TaskType,
)
from app.exceptions import InvalidStateException, NotFoundException, ValidationException
from app.database.base import get_session, get_db_session
from app.services.results import SupervisedResultSchema
from sqlmodel import select
from app.config import Config
from app.helpers import DTEncoder
from app.logger import Logger
import os,shutil,statistics

"""
Model lifecycle & inference service: integrity checks, inference, top-model
selection, cleanup, and dashboard stats. Training/finetune orchestration lives
in app.services.train.
"""

def check_dataset_pretrained(dataset: Dataset):
  dataset.check_integrity()
  if not dataset.top_model and not dataset.models:
    raise NotFoundException("Trained model", dataset.name)
  for m in dataset.models: check_integrity_model(m)


def check_available_results(dataset: Dataset):
  paths = Config.dir / "storages" / dataset.name / "results"
  if not os.listdir(paths): raise NotFoundException("Results", dataset.name)


def check_integrity_model(m: Optional[ModelML]):
  if not m: raise NotFoundException("Model", "unknown")
  if not m.is_active:
    raise InvalidStateException("Model", "inactive", "active")
  if m.status != StatusProcess.SUCCESS_TRAIN:
    raise InvalidStateException("Model", str(m.status), str(StatusProcess.SUCCESS_TRAIN))
  pathfile = Config.dir / f"{m.path}.pkl"
  if not os.path.exists(pathfile):
    raise NotFoundException("Model file", str(m.path))

def change_top_model(dataset: Dataset, algorithm: str):
  logger = Logger(dataset.name)
  check_dataset_pretrained(dataset)
  before_top_model = dataset.top_model
  algorithm_list = [m.name for m in dataset.models if m.algorithm == algorithm]
  if not algorithm_list:
    raise NotFoundException("Algorithm", algorithm)
  dataset.top_model = algorithm_list[0]
  logger.info(
      f"Change top model dataset [{dataset.name}] from : {before_top_model} to : {dataset.top_model}"
  )


def inference(payload: InferenceRequestSchema, db):
  dataset = Dataset.get_by_name(payload.dataset_name, db)
  check_dataset_pretrained(dataset)
  logger = Logger(dataset.name)
  logger.info(f"Inference | X : {payload.X}")
  payload.check_integrity_payload(dataset.task_type)  # Check X data type

  if dataset.task_type.is_clustering():
    # Clustering is transductive (append point + refit + persist CSVs), not a
    # plain predict-with-saved-model — delegate to its own service.
    from app.services import clustering as clustering_service
    return clustering_service.inference(dataset, payload.X, logger)

  if dataset.task_type.is_timeseries():
    raise ValidationException(
        f"TimeSeries is not supported on this endpoint, use /models/{dataset.name}/forecast instead")

  core = dataset.task_type.core()
  if not core: raise ValidationException(f"Task type '{dataset.task_type}' is not supported for inference")
  result = core.predict(dataset.to_predict_request(payload.X), logger)
  return SupervisedResultSchema.from_result(result, dataset.name)


def clean_models(dataset: Dataset, db):
  check_dataset_pretrained(dataset)
  logger = Logger(dataset.name)
  dataset.top_model = ""
  dataset.status = StatusProcess.SUCCESS_PULL
  ix = 0
  for m in dataset.models:
    pathfile = (
      Config.dir / "storages" / dataset.name /
      "top_model" / f"{m.algorithm}.pkl"
    )
    resultsfile = Config.dir / "storages" / dataset.name / "results"
    db.delete(m)
    db.commit()
    if os.path.exists(pathfile):
      os.remove(pathfile)
    if os.path.exists(resultsfile):
      shutil.rmtree(resultsfile)
      os.makedirs(resultsfile, exist_ok=True)
    ix += 1
  logger.info("Cleaned Models Succesfully!")
  return ix

def clean_results(dataset: Dataset):
  core = dataset.task_type.core()
  if not core: raise ValidationException(f"Task type '{dataset.task_type}' is not supported")
  fpath = Config.dir / "storages" / dataset.name / "results"
  if os.path.exists(fpath):
    shutil.rmtree(fpath)
    os.makedirs(fpath, exist_ok=True)


def get_stats():
  with get_db_session() as db:
    return _build_stats(db.exec(select(Dataset)).all())


def _as_datetime(value):
  """`meta.created_at` disimpan sebagai STRING ISO di kolom JSON.

  Seluruh fungsi di bawah memperlakukannya sebagai datetime (`.date()`,
  `time_ago`), sehingga tanpa normalisasi ini endpoint `/utils/stats` melempar
  `AttributeError: 'str' object has no attribute 'tzinfo'` begitu ada satu saja
  dataset berstatus ACTIVE/IDLE. Nilai yang sudah berupa datetime dilewatkan apa
  adanya supaya pemanggil lama tetap jalan.
  """
  if isinstance(value, datetime.datetime):
    return value
  if not value:
    return None
  try:
    return datetime.datetime.fromisoformat(str(value))
  except ValueError:
    return None


def _build_stats(datasets):
  created_at = []
  task_type = []
  accuracy = []
  total_models = 0
  total_dataset = len(datasets)
  total_workers = len(WorkerManager.get_tasks())
  recent_activity = []
  get_key_eval = {TaskType.Clustering:"Silhouette",
                  TaskType.Classification:"Accuracy",
                  TaskType.Regression:"MAPE",
                  TaskType.TimeSeries:"MAPE"
                  # Anomaly sengaja tidak ikut. Sejak `_summarise` ia memang punya
                  # metrik, tapi semuanya deskriptif — `AnomalyRate` menyatakan
                  # BERAPA BANYAK yang ditandai, bukan seberapa BENAR. Memasukkannya
                  # ke rata-rata akurasi akan membandingkan dua hal yang berbeda.
                  }
  now = DTEncoder.now()
  for data in datasets:
    if data.status not in [StatusProcess.ACTIVE,StatusProcess.IDLE]: continue
    data_created_at = _as_datetime(data.meta.created_at)
    task_type.append(str(data.task_type))
    models = data.models
    total_models += len(models)
    now = DTEncoder.now()

    time = ""
    if data_created_at:                            # meta lama bisa saja kosong
      created_at.append(data_created_at)
      time = DTEncoder.time_ago(data_created_at, now)
      if now.date() == data_created_at.date():
        recent = {"id":data.name,"description":"Dataset Created","status":str(data.status),"time":time}
        recent_activity.append(recent)

    for m in models:
      model_created_at = _as_datetime(m.meta.created_at)
      if model_created_at and model_created_at.date() == now.date():
        recent = {"id":m.name,"description":"Model Created","status":str(m.status),"time":time}
        recent_activity.append(recent)

      evaluation = m.evaluation
      key = get_key_eval.get(data.task_type)
      if not key or key not in evaluation: continue   # e.g. Anomaly has no metric
      metric = evaluation[key]
      if key == "MAPE": metric = max(0,1-metric)
      accuracy.append(metric)
  avg_accuracy = statistics.mean(accuracy) if accuracy else 0
  sum_data = dict(Counter(dt.date().isoformat() for dt in created_at))
  sum_task_type = dict(Counter(t for t in task_type))

  return {"avg_accuracy":avg_accuracy,"sum_data":sum_data,
          "sum_task_type":sum_task_type,"total_model":total_models,
          "total_dataset":total_dataset,"total_workers":total_workers,"recent_activity":recent_activity}


if __name__ == "__main__":
  import pprint
  dataset_name = "TimeSeries-0743b615"
  db = next(get_session())

