from fastapi import HTTPException
from pydantic import BaseModel
from pathlib import Path
from app.exceptions import AlreadyExistsException, InvalidStateException, NotFoundException,ValidationException, handle_exception
from sqlmodel import SQLModel, Field, Column, JSON, Session, select, Relationship, delete, func
from app.database.schemas import (MetaDataset, PreprocessingSchema,PreprocessingSchemaType, ResponseDatasetSchema,
    TaskType, StatusProcess, ModelResponseSchema, ViewModels,CreateDatasetSchema,MetaDatasetType,MetaModelMLType,MetaModelML
)
from sqlalchemy import Boolean, String
from app.utils.security import safe_path_join
from typing import Optional, List
import sqlalchemy as sa,pandas as pd,os, datetime,uuid,shutil
from app.config import WORKER, Config
from app.helpers import init_storages
from app.logger import LOGGER_GLOBAL
from app.helpers import DTEncoder

class Dataset(SQLModel, table=True):
  id: int = Field(default=None, primary_key=True, nullable=False)
  task_type: TaskType = Field(sa_column=Column(sa.Enum(TaskType), nullable=False))
  description: str = Field(sa_column=Column(String, default=""))
  name: str = Field(sa_column=Column(String, unique=True, nullable=False))
  features: List[str] = Field(default_factory=list, sa_column=Column(JSON))
  target: str  # is n_cluster if task_type is clustering
  status: StatusProcess = Field(sa_column=Column(
      sa.Enum(StatusProcess), nullable=False, default=StatusProcess.PENDING))
  top_model: Optional[str]
  start_date: str # 20250827
  end_date: str   # 20250127
  time_start: str # 00:00:00
  time_end: str   # 23:59:00
  interval: int   # minutes
  preprocessing: PreprocessingSchema = Field(default_factory=PreprocessingSchema, sa_column=Column(PreprocessingSchemaType))
  is_valid: bool = Field(sa_column=Column(Boolean), default=False)
  meta: MetaDataset = Field(default_factory=MetaDataset, sa_column=Column(MetaDatasetType))
  n_models: int = 0
  # One-to-many ke ModelML
  models: List["ModelML"] = Relationship(back_populates="dataset", sa_relationship_kwargs={
                                         "cascade": "all, delete-orphan"})
  
  @staticmethod
  def create_from_request(payload:CreateDatasetSchema,db:Session):
    """Create new dataset with secure path handling."""
    name = f"{payload.task_type}-{str(uuid.uuid4())[:8]}"

    # Check if name already exists (get_by_name raises NotFoundException when free)
    try:
      Dataset.get_by_name(name, db)
      raise AlreadyExistsException("Dataset", name)
    except NotFoundException:
      pass

    # Initialize storage securely
    init_storages(name)

    # Create dataset record
    q = Dataset(name=name, is_valid=False, status=StatusProcess.PENDING, **payload.model_dump())
    return q

  def to_response(self) -> ResponseDatasetSchema:
    features = [f for f in self.features]
    target = self.target

    return ResponseDatasetSchema(names=self.name, description=self.description, task_type=self.task_type.name,
                                 features=features, target=target,
                                 start_date=self.start_date, end_date=self.end_date,
                                 meta=self.meta, is_valid=self.is_valid, interval=self.interval,
                                 status=self.status.name, top_model=self.top_model, models=self.get_models(),n_models=self.n_models,preprocessing=self.preprocessing)
  
  def remove(self,db:Session):
    from app.workers.manager import WorkerManager
    from app.database.historicalDB import get_influx_storage
    LOGGER_GLOBAL.info(f"Delete Dataset {self.name} ...")
    try :
      ifx_client = get_influx_storage()
      ifx_client.delete_dataset(self.name)
      LOGGER_GLOBAL.info(f"Historical (Predictions) Successfully remove ...")
    except Exception as e: raise handle_exception(e,LOGGER_GLOBAL)
    try:
      filepath = safe_path_join(Config.dir,"storages",self.name)
      if filepath.exists(): shutil.rmtree(filepath)
      LOGGER_GLOBAL.info(f"Storages Successfully remove ...")
    except Exception as e: raise handle_exception(e,LOGGER_GLOBAL)

    try:
      WorkerManager.delete_by_dataset(self.name)
      LOGGER_GLOBAL.info(f"Worker state deleted for dataset: {self.name}")
    except Exception as e: raise handle_exception(e,LOGGER_GLOBAL)

    db.delete(self)
    db.commit()

  @classmethod
  def remove_all(cls,db:Session):
    dataset = cls.get_all(db)
    if dataset:
      for data in dataset: data.remove(db)
    LOGGER_GLOBAL.info("Delete ALL Dataset Successfully")

  @classmethod
  def to_responses(cls, datas, db: Session):
    total = db.exec(select(func.count(cls.id))).one()
    n_datas = len(datas)
    return {"recordsTotal": total, "recordsFiltered": n_datas, "data": datas}

  def open_dataframe(self) -> pd.DataFrame:
    path = Config.dir/"storages"/self.name/"data.csv"
    if not os.path.exists(path): raise FileNotFoundError(f"{path} is not found!")
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["dt"])
    return df
  
  @property
  def path(self) -> Path: return safe_path_join(Config.dir,"storages",self.name)

  def to_train_request(self, n_top: int = 1):
    """Describe this dataset as a core train request (no DB/HTTP knowledge).

    Returns the per-task request subclass. Only supervised is wired today; the
    other task types build their own request when they are migrated.
    """
    from app.core.contracts import (SupervisedTrainRequest, ClusteringTrainRequest,
                                     TimeSeriesTrainRequest)
    common = dict(
      df=self.open_dataframe(),
      preprocessing=self.preprocessing.to_args_pycaret() if self.preprocessing else {},
      out_dir=self.path / "top_model",
      task=self.task_type.name,
      n_top=n_top,
    )
    if self.task_type.is_supervised():
      return SupervisedTrainRequest(target=self.target, **common)
    if self.task_type.is_clustering():
      # clustering stores the requested cluster count in `target`.
      return ClusteringTrainRequest(n_clusters=int(self.target), **common)
    if self.task_type.is_timeseries():
      # time series stores the forecast horizon in `target`.
      return TimeSeriesTrainRequest(fh=int(self.target), **common)
    raise ValidationException(f"to_train_request not supported for {self.task_type.name} yet")

  def to_anomaly_train_request(self, fraction: float, algorithm: str):
    """Anomaly training request. `fraction` and `algorithm` are per-request knobs
    (not stored on the dataset), so they are passed in rather than derived here."""
    from app.core.contracts import AnomalyTrainRequest
    return AnomalyTrainRequest(
      df=self.open_dataframe(),
      preprocessing=self.preprocessing.to_args_pycaret() if self.preprocessing else {},
      out_dir=self.path / "top_model",
      task=self.task_type.name,
      fraction=fraction,
      algorithm=algorithm,
    )

  def to_predict_request(self, features: dict):
    """Describe this dataset as a core PredictRequest (no DB/HTTP knowledge)."""
    from app.core.contracts import PredictRequest
    return PredictRequest(
      features=features,
      models=[(m.algorithm, str(Config.dir / m.path)) for m in self.models],
      task=self.task_type.name,
    )

  def to_forecast_request(self):
    """Describe this dataset as a core ForecastRequest (no timestamps/DB)."""
    from app.core.contracts import ForecastRequest
    return ForecastRequest(
      models=[(m.algorithm, str(Config.dir / m.path)) for m in self.models],
      fh=int(self.target),
      task=self.task_type.name,
    )

  def to_cluster_assign_request(self, features: dict, reference=None):
    """Describe this dataset as a core ClusterAssignRequest.

    `reference` (baris latih + kolom label) hanya diperlukan algoritma yang tidak
    punya `predict`; pemanggil yang menyediakannya dari clusters.csv.
    """
    from app.core.contracts import ClusterAssignRequest
    return ClusterAssignRequest(
      features=features,
      models=[(m.algorithm, str(Config.dir / m.path)) for m in self.models],
      task=self.task_type.name,
      reference=reference,
    )

  def to_cluster_request(self, df):
    """Describe this dataset as a core ClusterRequest. `df` is the combined
    training-plus-new-point frame the service assembled (no DB/storage here)."""
    from app.core.contracts import ClusterRequest
    return ClusterRequest(
      df=df,
      algorithms=[m.algorithm for m in self.models],
      n_clusters=int(self.target),
      preprocessing=self.preprocessing.to_args_pycaret() if self.preprocessing else {},
      task=self.task_type.name,
    )

  def to_anomaly_predict_request(self, features: dict):
    """PredictRequest for anomaly: a single model at the fixed `<storage>/anomaly`
    path (anomaly has no per-algorithm ModelML rows)."""
    from app.core.contracts import PredictRequest
    return PredictRequest(
      features=features,
      models=[("anomaly", str(self.path / "anomaly"))],
      task=self.task_type.name,
    )

  def check_features(self, X: dict) -> bool:
    columns = set(X.keys())
    if set(self.features) != columns: return False
    if not all(len(x) for x in list(X.values())): return False
    return True

  def save(self, db: Session):
    """
    Menyimpan atau meng-update instance ini ke database.
    """
    if self.id is None: db.add(self)
    else:
      existing = db.get(Dataset, self.id)
      if existing:
        for field, value in self.model_dump().items(): setattr(existing, field, value)
      else: db.add(self)
    db.commit()
    db.refresh(self)

  def get_models(self):
    if not self.models or not self.is_valid: return []
    else: return [i.view_model() for i in self.models if i.is_active]

  def check_integrity(self):
    """Validate dataset integrity with security checks."""
    if not self.is_valid: raise InvalidStateException("Dataset",str(self.is_valid),"True")
    if not self.features: raise InvalidStateException("Dataset",str(self.features),"There are features/columns, such as ['X1', 'X2', 'X3']")

    # Use secure path joining
    if not self.path.exists(): raise NotFoundException("Dataset","Storages")
    if not safe_path_join(self.path,"data.csv").exists(): raise NotFoundException("Dataset","Dataset CSV")

  @classmethod
  def _stmt(cls, *fields): return select(*fields) if fields else select(cls)

  @classmethod
  def get_by_id(cls, id, db: Session, * fields):
    data = db.exec(cls._stmt(*fields).where(cls.id == id)).first()
    if not data: raise NotFoundException("Dataset",id)
    return data

  @classmethod
  def get_by_name(cls, name, db: Session, *fields):
    data = db.exec(cls._stmt(*fields).where(cls.name == name)).first()
    if not data: raise NotFoundException("Dataset",name)
    return data

  @classmethod
  def get_by_task_type(cls, task_type:TaskType, db: Session, *fields):
    rows = db.exec(cls._stmt(*fields).where(cls.task_type == task_type)).all()
    if not rows: raise NotFoundException("Dataset",task_type.name)
    return rows

  @classmethod
  def get_all(cls, db: Session, to_response: bool = True):
    rows = db.exec(cls._stmt()).all()
    if not rows: return []
    if to_response: return [row.to_response() for row in rows]
    return rows

  @classmethod
  def delete_all(cls, db):
    """ Delete Only ORM """
    db.exec(delete(cls))
    db.commit()

  def is_time_to_finetune(self) -> bool:
    now = DTEncoder.now()
    current_dt = self.meta.current_dt
    if not current_dt or not self.preprocessing:
      return False
    # current_dt is persisted as an ISO string (JSON column); parse it before
    # date arithmetic — otherwise `str + timedelta` raises TypeError and the
    # whole worker tick fails (skipping inference too).
    if isinstance(current_dt, str):
      current_dt = datetime.datetime.fromisoformat(current_dt)
    time_to_finetune = current_dt + datetime.timedelta(days=self.preprocessing.interval_finetune)
    return time_to_finetune.date() <= now.date()
  
  def get_df_sample(self,n_sample:int=100) -> pd.DataFrame: return self.get_df().sample(n_sample)

  def get_df(self) -> pd.DataFrame:
    pathfile = safe_path_join(Config.dir, "storages", self.name, "data.csv")
    if not pathfile.exists(): raise NotFoundException("Storages",self.name)
    df = pd.read_csv(pathfile)
    return df


class ModelML(SQLModel, table=True):
  id: int = Field(primary_key=True, nullable=False)
  dataset_id: int = Field(foreign_key="dataset.id")
  name: str = Field(nullable=False)
  algorithm: str = Field(sa_column=Column(String, nullable=False))
  is_active: bool = Field(sa_column=Column(Boolean), default=True)
  evaluation: dict = Field(sa_column=Column(JSON))
  finetune: bool = Field(sa_column=Column(Boolean), default=False)
  meta: MetaModelML = Field(default_factory=MetaModelML, sa_column=Column(MetaModelMLType))
  status: StatusProcess = Field(sa_column=Column(sa.Enum(StatusProcess), nullable=False, default=StatusProcess.PENDING))
  path: str = Field(sa_column=Column(String, default=None))
  dataset: Dataset = Relationship(back_populates="models")

  def to_response(self) -> ModelResponseSchema:
    return ModelResponseSchema(name=self.name, dataset_name=self.dataset.name,
                               algorithm=self.algorithm, is_active=self.is_active,
                               finetune=self.finetune, evaluation=self.evaluation, meta=self.meta, status=self.status.name)

  def save(self, db: Session):
    """
    Menyimpan atau meng-update instance ini ke database.
    """
    if self.id is None: db.add(self)
    else:
      existing = db.get(ModelML, self.id)
      if existing:
        for field, value in self.model_dump().items(): setattr(existing, field, value)
      else: db.add(self)
    db.commit()
    db.refresh(self)

  def view_model(self) -> dict:
    desc = self.dataset.task_type.algorithms().get(self.algorithm, "")
    size = self.meta.sizeof
    return ViewModels(name=self.name, evaluation=self.evaluation, size=size, status=str(self.status), algorithm=self.algorithm, description=desc, path=self.path).model_dump()

  @classmethod
  def _stmt(cls, *fields): return select(*fields) if fields else select(cls)

  @classmethod
  def get_by_id(cls, id, db: Session, *
                fields): return db.exec(cls._stmt(*fields).where(cls.id == id)).first()

  @classmethod
  def get_by_name(cls, name, db: Session, *fields): return db.exec(cls._stmt(*fields).where(cls.name == name)).first()


  
