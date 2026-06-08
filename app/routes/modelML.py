from __future__ import annotations
from asyncio import Task
from typing import Optional
from fastapi import HTTPException
from app.database.orm import Dataset, ModelML
from app.routes.dataset import check_integrity_dataset
from collections import Counter
from app.workers.manager import WorkerManager
from app.database.schemas import (
    ALGO_LIST,
    AnomalyRequestSchema,
    InferenceRequestSchema,
    StatusProcess,
    TaskType,
)
from app.database.db import get_session
from app.config import Config
from app.pull import pulling
from app.helpers import DTEncoder, pca
from app.logger import Logger
import os,datetime,shutil,pandas as pd,statistics
from contextlib import closing
from app.core.anomaly import Anomaly
from app.workers.manager import WorkerManager

"""
This Module for checking and function pass from router
"""


def check_dataset_pretrained(dataset: Dataset):
  check_integrity_dataset(dataset)
  if not dataset.top_model and not dataset.models: raise HTTPException(status_code=404, detail="Dataset is not trained !")
  for m in dataset.models: check_integrity_model(m)


def check_available_results(dataset: Dataset):
  paths = Config.dir / "storages" / dataset.name / "results"
  if not os.listdir(paths): raise HTTPException(status_code=500, detail="Results is not found!")


def check_integrity_model(m: Optional[ModelML]):
  if not m: raise HTTPException(status_code=404, detail="Model is not found!")
  if not m.is_active:
    raise HTTPException(
        status_code=500, detail=f"Model : {m.algorithm} is not active"
    )
  if m.status != StatusProcess.SUCCESS_TRAIN:
    raise HTTPException(
        status_code=500,
        detail=f"Model : {m.algorithm} is not TRAIN | Status : {m.status}",
    )
  pathfile = Config.dir / f"{m.path}.pkl"
  if not os.path.exists(pathfile):
    raise HTTPException(
        status_code=500, detail=f"Model : {m.algorithm} file is not found"
    )

def auto_initialize(dataset: Dataset, n_models: int,db):
  dataset_name = dataset.name
  logger = Logger(dataset_name)
  logger.info("Initialize ...")
  pulling(dataset_name,db)
  core = dataset.task_type.core()
  if dataset.is_valid and dataset.status == StatusProcess.SUCCESS_PULL:
    core.find_top_model(dataset, n_models,logger,db)
    logger.info("Success Initialize !")
  else: logger.warning(f"Status : {dataset.status if dataset else 'Not Found'}")


def train_model(dataset: Dataset, algorithm: str, db, use_this=False):
  check_integrity_dataset(dataset)
  core = dataset.task_type.core()
  logger = Logger(dataset.name)
  logger.info("Training Model ...")
  algorithms_list = dataset.task_type.algorithms().keys()
  if algorithm not in algorithms_list:
    raise HTTPException(
        status_code=500,
        detail=f"{algorithm} not in algorithms_list : {algorithms_list}",
    )

  args = (dataset, algorithm, logger)
  if not core:
    raise HTTPException(
        status_code=500, detail=f"TaskType :{dataset.task_type} Core is not found !"
    )
  core.train(*args)
  if use_this or not dataset.top_model:
    dataset.top_model = dataset.models[-1].name
  db.commit()


def change_top_model(dataset: Dataset, algorithm: str):
  logger = Logger(dataset.name)
  check_dataset_pretrained(dataset)
  before_top_model = dataset.top_model
  algorithm_list = [m.name for m in dataset.models if m.algorithm == algorithm]
  if not algorithm_list:
    raise HTTPException(
        status_code=404, detail=f"{algorithm} not in models: {algorithm_list}"
    )
  dataset.top_model = algorithm_list[0]
  logger.info(
      f"Change top model dataset [{dataset.name}] from : {before_top_model} to : {dataset.top_model}"
  )


def find_top_model(dataset: Dataset, n_top: int):
  logger = Logger(dataset.name)
  logger.info(
      f"Start Compare Models {dataset.task_type} | {dataset.name} | n-top {n_top}"
  )
  core = dataset.task_type.core()
  try:
    if not core:
      raise HTTPException(
          status_code=500,
          detail=f"TaskType :{dataset.task_type} Core is not found !",
      )
    core.find_top_model(dataset.name, n_top, logger=logger)
    logger.info("Result is commit")
  except Exception as e:
    logger.error(str(e))


def inference(payload: InferenceRequestSchema, db):
  dataset = Dataset.get_by_name(payload.dataset_name, db)
  if not dataset:
    raise HTTPException(status_code=404, detail="dataset is not found!")
  check_dataset_pretrained(dataset)
  core = dataset.task_type.core()
  logger = Logger(dataset.name)
  logger.info(f"Inference | X : {payload.X}")
  task_type = dataset.task_type
  payload.check_integrity_payload(task_type)  # Check X data type

  args = (payload, dataset, logger)
  if not core:
    raise HTTPException(status_code=500, detail=f"TaskType :{task_type} Core is not found !")
  return core.inference(*args)


def auto_anomaly(payload: AnomalyRequestSchema, dataset):
  task_type = TaskType.Anomaly
  logger = Logger(payload.dataset_name)
  logger.info("Run Auto Anomaly ...")
  Anomaly.train(dataset,payload.algorithm,payload.fraction,logger=logger)

  task_name = f"{payload.dataset_name}-{task_type}"
  if WorkerManager.is_active(task_name): return {"detail": f"{task_name} is still active"}
  WorkerManager.create(payload.dataset_name, task_type)


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
  if not core: raise HTTPException(status_code=500, detail=f"TaskType :{dataset.task_type} Core is not found !")
  fpath = Config.dir / "storages" / dataset.name / "results"
  if os.path.exists(fpath):
    shutil.rmtree(fpath)
    os.makedirs(fpath, exist_ok=True)


def get_stats():
  db = next(get_session())
  datasets = db.query(Dataset).all()
  created_at = []
  task_type = []
  accuracy = []
  total_models = 0
  total_dataset = len(datasets)
  total_workers = len(WorkerManager.get_tasks())
  recent_activity = []
  get_key_eval = {TaskType.Clustering:"Silhouette",
                  TaskType.Regression:"MAPE",
                  TaskType.TimeSeries:"MAPE"
                  }
  now = DTEncoder.now()
  for data in datasets:
    if data.status not in [StatusProcess.ACTIVE,StatusProcess.IDLE]: continue
    meta = data.meta
    data_created_at = datetime.datetime.fromisoformat(meta["created_at"])
    created_at.append(data_created_at)
    task_type.append(str(data.task_type))
    models = data.models
    total_models += len(models)
    now = DTEncoder.now()
    time = DTEncoder.time_ago(data_created_at,now)
    if now.date() == data_created_at.date():
      recent = {"id":data.name,"description":"Dataset Created","status":str(data.status),"time":time}
      recent_activity.append(recent)

    for m in models:
      model_created_at = datetime.datetime.fromisoformat(m.meta['created_at'])
      if model_created_at.date()  == now.date():
        recent = {"id":m.name,"description":"Model Created","status":str(m.status),"time":time}
        recent_activity.append(recent)

      evaluation = m.evaluation
      key = get_key_eval[data.task_type]
      metric = evaluation[key]
      if key == "MAPE": metric = max(0,1-metric)
      accuracy.append(metric)
  avg_accuracy = statistics.mean(accuracy)
  sum_data = dict(Counter(dt.date().isoformat() for dt in created_at))
  sum_task_type = dict(Counter(t for t in task_type))

  return {"avg_accuracy":avg_accuracy,"sum_data":sum_data,
          "sum_task_type":sum_task_type,"total_model":total_models,
          "total_dataset":total_dataset,"total_workers":total_workers,"recent_activity":recent_activity}


if __name__ == "__main__":
  import pprint
  dataset_name = "TimeSeries-0743b615"
  db = next(get_session())

