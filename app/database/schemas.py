from pydantic import BaseModel, Field
from typing import Any, Optional, List, Union, Dict
from enum import Enum, auto,StrEnum
import json
import os
import datetime

from sqlmodel import JSON, TypeDecorator

from app.helpers import DTEncoder


class StatusProcess(Enum):
  PENDING = auto()  # Menunggu proses dimulai
  RUNNING_PULL = auto()  # Sedang jalan
  SUCCESS_PULL = auto()  # Tugas Selesai dengan sukses
  ERROR_PULL = auto()  # Ada masalah atau kegagalan
  RUNNING_TRAIN = auto()
  SUCCESS_TRAIN = auto()
  ERROR_TRAIN = auto()
  RUNNING_COMPARE = auto()
  RUNNING_FINETUNE = auto()
  IDLE = auto()
  ACTIVE = auto()
  PAUSED = auto()  # Proses dihentikan
  CANCELLED = auto()  # Proses dibatalkan
  QUEUED = auto()  # Menunggu antrian
  VALIDATING = auto()  # Memeriksa data / model

  def is_fail(self) -> bool:
    return self in (StatusProcess.ERROR_PULL,StatusProcess.ERROR_TRAIN,StatusProcess.CANCELLED,StatusProcess.QUEUED)
  def is_normal(self): return self in (self.IDLE,self.ACTIVE)
  def is_waiting(self) -> bool: return "RUNNING" in self.name
  def __str__(self): return self.name


_ALGORITHM_LIST: Optional[Dict[str, Dict[str, str]]] = None

def _algorithm_list() -> Dict[str, Dict[str, str]]:
  """Load & cache algorithm_list.json ({task_type: {algorithm: description}}).
  Same source as core's ALGO_LIST; loaded lazily so the database layer stays
  free of a core import."""
  global _ALGORITHM_LIST
  if _ALGORITHM_LIST is None:
    from app.config import Config
    with open(Config.dir / "algorithm_list.json", "r") as f:
      _ALGORITHM_LIST = json.load(f)
  return _ALGORITHM_LIST


class TaskType(Enum):
  Classification = auto()
  Regression = auto()
  Clustering = auto()
  TimeSeries = auto()
  Anomaly = auto()

  @classmethod
  def tolist(cls):
    return list(cls.__members__.keys())

  def is_classification(self) -> bool: return self == TaskType.Classification
  def is_regression(self) -> bool : return self == TaskType.Regression
  def is_clustering(self) -> bool: return self == TaskType.Clustering
  def is_anomaly(self) -> bool: return self == TaskType.Anomaly
  def is_timeseries(self) -> bool: return self == TaskType.TimeSeries
  def is_supervised(self) -> bool: return self.is_classification() or self.is_regression()
  def is_unsupervised(self) -> bool: return self.is_clustering()

  @staticmethod
  def from_str(task_type: str):
    try: return TaskType[task_type]
    except KeyError: raise ValueError(f"Invalid task type: {task_type}")

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
    else: return None

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

  def algorithms(self) -> Dict[str, str]:
    """{algorithm_key: description} available for this task type (from
    algorithm_list.json). Used by the algorithms endpoint, train-time
    validation, and ModelML.view_model for the human-readable description."""
    return _algorithm_list().get(self.name, {})

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
  outlier_handling: bool = False
  scale: bool = False
  dim_reduce: bool = False
  interval_finetune: int = 5 # days

  def to_args_pycaret(self) -> dict:
    pca_components = 2 if self.dim_reduce else None
    return dict(remove_outliers=self.outlier_handling, normalize=self.scale,pca=self.dim_reduce,pca_components=pca_components)

class PreprocessingSchemaType(TypeDecorator):
  impl = JSON
  cache_ok = True
  def process_bind_param(self, value, dialect):
    if value is None: return None
    if isinstance(value, PreprocessingSchema): return value.model_dump(mode="json")
    return value
  def process_result_value(self, value, dialect):
    if value is None: return None
    return PreprocessingSchema.model_validate(value)


class MetaDataset(BaseModel):
  created_by: str = "IMAJI"
  created_at: str = Field(default_factory=lambda: DTEncoder.now().isoformat())
  tags : Optional[Dict] = None
  last_update: str | None = None
  current_dt: str | None = None
  pulling_time: float = 0.0
  train_time: float = 0
  sizeof: float = 0.0
  n_rows: int = 0
  n_cols: int = 0
  notes: str = ""
  train_size: float = 0.0
  missing_values: int = 0 
  outliers_information: Optional[Dict] = None
  random_seed: int = 42
  columns: List = []

class MetaDatasetType(TypeDecorator):
  impl = JSON
  cache_ok = True
  def process_bind_param(self, value, dialect):
    if value is None: return None
    if isinstance(value, MetaDataset): return value.model_dump(mode="json")
    return value
  def process_result_value(self, value, dialect):
    if value is None: return None
    return MetaDataset.model_validate(value)

class MetaModelML(BaseModel):
  created_by: str = "IMAJI"
  created_at: str = Field(default_factory=lambda: DTEncoder.now().isoformat())
  train_time: float = 0.0
  last_update: str | None = None
  sizeof: float = 0.0
  notes: str = ""

# Backward-compat alias: the core modules were not updated after the
# MetaModel -> MetaModelML rename. See migration notes.
MetaModel = MetaModelML

class MetaModelMLType(TypeDecorator):
  impl = JSON
  cache_ok = True
  def process_bind_param(self, value, dialect):
    if value is None: return None
    if isinstance(value, MetaModelML): return value.model_dump(mode="json")
    return value
  def process_result_value(self, value, dialect):
    if value is None: return None
    return MetaModelML.model_validate(value)

class CreateDatasetSchema(BaseModel):
  description: str = ""
  task_type: str
  features: List
  tags: Optional[Dict] = None # {tagname:row_id}
  target: Union[str, int]
  start_date: str
  end_date: str
  time_start: str = "00:00:00"
  time_end: str = "23:59:00"
  interval: int = 0
  preprocessing:PreprocessingSchema = Field(default_factory=PreprocessingSchema)
  n_models: int = 0
  def tt(self): return TaskType.from_str(self.task_type)

class ResponseDatasetSchema(BaseModel):
  task_type: str;
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
  n_models: int


class AnomalyRequestSchema(BaseModel): dataset_name: str; algorithm: str; fraction: float

class ModelTrainSchema(BaseModel): dataset_name: str; algorithm: str; use_this: bool

class ModelResponseSchema(BaseModel):
  name: str; dataset_name: str; algorithm: str; is_active: bool
  finetune: bool; evaluation: dict; meta: MetaModelML; status: str

class ViewModels(BaseModel):
  name: str; evaluation: Dict; algorithm: str; description: str
  path: str; size: float; status: str

class InferenceRequestSchema(BaseModel):
  dataset_name: str
  X: Any

  def check_integrity_payload(self, task_type: TaskType):
    from fastapi import HTTPException

    if task_type.is_supervised() or task_type.is_unsupervised():
      if not isinstance(self.X, dict):
        raise HTTPException(status_code=500, detail=f"X is invalid : {self.X} \nX should follow this format :  {{key:[value]}}")
    elif task_type.is_timeseries():
      if not isinstance(self.X, int):
        raise HTTPException(status_code=500, detail=f"X is invalid : {self.X}| X (fh) must int data type")
    else: raise HTTPException(status_code=404, detail="Not Implemented Error")

class FindTopModelRequestSchema(BaseModel):
  dataset_name: str
  n_top: int

class ChangeTopModelSchema(BaseModel):
  dataset_name: str; 
  algorithm: str

class InitiateResponseSchema(BaseModel):
  success: bool
  msg: str

class SupervisedResponseSchema(BaseModel):
  target_name: str
  value_pred: float
  value_actual: float

class UnsupervisedResponseSchema(BaseModel):
  timestamps: str
  result: dict

class TimeSeriesResponseSchema(BaseModel):
  target_name: str
  result: dict

class InferenceResponseSchema(BaseModel):
  timestamp: datetime.datetime
  features: dict
  results: SupervisedResponseSchema | UnsupervisedResponseSchema | TimeSeriesResponseSchema

if __name__ == "__main__":
  a = TaskType.Regression
  print(a.module_name)
