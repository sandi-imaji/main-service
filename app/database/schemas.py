from pydantic import BaseModel
from typing import Any, Optional, List, Union, Dict
from enum import Enum, auto,StrEnum
import json
import os
import datetime

filepath = "./algorithm_list.json"
if not os.path.exists(filepath):
  raise ValueError(f"algorithm list : [{filepath}] is not found!")
with open("./algorithm_list.json", "r") as f:
  ALGO_LIST = json.loads(f.read())


class StatusProcess(Enum):
  PENDING = auto()  # Menunggu proses dimulai
  RUNNING_PULL = auto()  # Sedang jalan
  SUCCESS_PULL = auto()  # Tugas Selesai dengan sukses
  ERROR_PULL = auto()  # Ada masalah atau kegagalan
  RUNNING_TRAIN = auto()
  SUCCESS_TRAIN = auto()
  ERROR_TRAIN = auto()
  RUNNING_COMPARE = auto()
  IDLE = auto()
  ACTIVE = auto()
  PAUSED = auto()  # Proses dihentikan
  CANCELLED = auto()  # Proses dibatalkan
  QUEUED = auto()  # Menunggu antrian
  VALIDATING = auto()  # Memeriksa data / model

  def __str__(self):
    return self.name


class TaskType(Enum):
  Classification = auto()
  Regression = auto()
  Clustering = auto()
  TimeSeries = auto()
  Anomaly = auto()
  ClassificationDummy = auto()
  RegressionDummy = auto()
  ClusteringDummy = auto()
  TimeSeriesDummy = auto()
  AnomalyDummy = auto()

  @classmethod
  def tolist(cls):
    return list(cls.__members__.keys())

  @classmethod
  def dummies(cls):
    lists = list(cls.__members__.values())
    return [d for d in lists if "Dummy" in str(d)]

  @classmethod
  def not_dummies(cls):
    lists = list(cls.__members__.values())
    return [d for d in lists if "Dummy" not in str(d)]

  def is_classification(self):
    return self in [TaskType.Classification, TaskType.ClassificationDummy]

  def is_regression(self):
    return self in [TaskType.Regression, TaskType.RegressionDummy]

  def is_clustering(self):
    return self in [TaskType.Clustering, TaskType.ClusteringDummy]

  def is_anomaly(self):
    return self in [TaskType.Anomaly, TaskType.AnomalyDummy]

  def is_timeseries(self):
    return self in [TaskType.TimeSeries, TaskType.TimeSeriesDummy]

  def is_supervised(self):
    return self.is_classification() or self.is_regression()

  def is_unsupervised(self):
    return self.is_clustering()

  def is_dummies(self):
    return self in [
        TaskType.ClassificationDummy,
        TaskType.RegressionDummy,
        TaskType.TimeSeriesDummy,
        TaskType.AnomalyDummy,
        TaskType.ClusteringDummy,
    ]

  @staticmethod
  def from_str(task_type: str):
    try:
      return TaskType[task_type]
    except KeyError:
      raise ValueError(f"Invalid task type: {task_type}")

  def core(self):
    if self.is_supervised():
      from app.core import Supervised

      return Supervised
    elif self.is_unsupervised():
      from app.core import Unsupervised

      return Unsupervised
    elif self.is_timeseries():
      from app.core import TimeSeries

      return TimeSeries
    else:
      return None

  def module(self):
    if self.is_classification():
      from pycaret import classification

      return classification
    elif self.is_regression():
      from pycaret import regression

      return regression
    elif self.is_clustering():
      from pycaret import clustering

      return clustering
    elif self.is_timeseries():
      from pycaret import time_series

      return time_series
    elif self.is_anomaly():
      from pycaret import anomaly

      return anomaly
    else:
      raise TypeError("Task Type invalid !")

  def algorithms(self) -> dict:
    return ALGO_LIST[self.name]

  def __str__(self):
    return self.name

  @property
  def base(self) -> str:
    return self.name.replace("Dummy", "")

  @property
  def module_name(self) -> str:
    if self.is_supervised(): return f"app.core.supervised"
    elif self.is_unsupervised(): return f"app.core.unsupervised"
    elif self.is_timeseries(): return f"app.core.time_series"
    elif self.is_anomaly(): return f"app.core.anomaly"
    else: raise 

class MetaDataset(BaseModel):
  created_by: str
  created_at: str
  last_update: str = ""
  current_dt: str = ""
  pulling_time: Union[int, float] = 0.0
  train_time: float = 0
  size_of: int
  n_rows: int
  n_cols: int
  notes: Optional[str]
  train_size: float
  missing_values: int
  is_outlier: Optional[bool]
  random_seed: Optional[int]
  columns: List[Any]
  path: str


class MetaModel(BaseModel):
  created_by: str
  created_at: str
  train_time: float = 0
  last_update: str = ""
  size_of: int
  notes: str

class DatasetHandling(StrEnum):
  NONE = "NONE"
  REMOVE = "REMOVE"
  MEAN = "MEAN"
  MEDIAN = "MEDIAN"
  MAX = "MAX"
  MIN = "MIN"
  NEIGHBOR_VALUE = "NEIGHBOR_VALUE"

  def is_handling(self): return self != DatasetHandling.NONE

class PreprocessingSchema(BaseModel):
  missing_handling: DatasetHandling = DatasetHandling.NEIGHBOR_VALUE
  outlier_handling: DatasetHandling = DatasetHandling.NONE
  scale: bool = False
  dim_reduce: bool = False
  interval_finetune: int = 30 # days
  retention: int = 30# days

class DatasetRequestSchema(BaseModel):
  description: str = ""
  task_type: str
  features: List
  target: Union[str, int]
  start_date: str
  end_date: str
  time_start: str = "00:00:00"
  time_end: str = "23:59:00"
  interval: int = 0
  preprocessing:Optional[PreprocessingSchema] = None

  def tt(self):
    return TaskType.from_str(self.task_type)

class DatasetResponseSchema(BaseModel):
  task_type: str
  description: str
  names: str
  features: List[str]
  target: Union[str, int]
  start_date: str
  end_date: str
  interval: int
  is_valid: bool
  meta: Union[MetaDataset, dict] = {}
  preprocessing: Optional[PreprocessingSchema] = None
  top_model: Optional[str] = None
  models: List[Any] = []
  status: str


class AnomalyRequestSchema(BaseModel):
  dataset_name: str
  algorithm: str
  fraction: float


class ModelTrainSchema(BaseModel):
  dataset_name: str
  algorithm: str
  use_this: bool


class ModelResponseSchema(BaseModel):
  name: str
  dataset_name: str
  algorithm: str
  is_active: bool
  finetune: bool
  evaluation: dict
  meta: Optional[dict]
  status: str


class ViewModels(BaseModel):
  name: str
  evaluation: Dict
  algorithm: str
  description: str
  path: str
  size: Union[int, str]
  status: str


class InferenceRequestSchema(BaseModel):
  dataset_name: str
  X: Any

  def check_integrity_payload(self, task_type: TaskType):
    from fastapi import HTTPException

    if task_type.is_supervised() or task_type.is_unsupervised():
      if not isinstance(self.X, dict):
        raise HTTPException(
            status_code=500,
            detail=f"X is invalid : {self.X} \nX should follow this format :  {{key:[value]}}",
        )
    elif task_type.is_timeseries():
      if not isinstance(self.X, int):
        raise HTTPException(
            status_code=500,
            detail=f"X is invalid : {self.X}| X (fh) must int data type",
        )
    else:
      raise HTTPException(status_code=404, detail="Not Implemented Error")


class InferenceResponseSchema(BaseModel):
  timestamp: datetime.datetime
  results: dict


class FindTopModelRequestSchema(BaseModel):
  dataset_name: str
  n_top: int


class ChangeTopModelSchema(BaseModel):
  dataset_name: str
  algorithm: str


class InitiateRequestSchema(BaseModel):
  description: str
  task_type: str
  features: List
  target: Union[str, int]
  start_date: str
  end_date: str
  time_start: str = "00:00:00"
  time_end: str = "23:59:00"
  interval: int = 0
  n_models: int
  preprocessing:Optional[PreprocessingSchema] = None

  def tt(self): return TaskType.from_str(self.task_type)


class InitiateResponseSchema(BaseModel):
  success: bool
  msg: str


class SupervisedResponseSchema(BaseModel):
  target_name: str
  value_pred: float
  value_actual: float
  timestamps: str


class UnsupervisedResponseSchema(BaseModel):
  timestamps: str
  result: dict


class TimeSeriesResponseSchema(BaseModel):
  target_name: str
  result: dict
  timestamps: str


if __name__ == "__main__":
  a = TaskType.Regression
  print(a.module_name)
