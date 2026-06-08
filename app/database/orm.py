from sqlalchemy.orm import query
from sqlmodel import SQLModel, Field, Column, JSON, Session, select, Relationship, delete, func
from app.database.schemas import (
    MetaDataset, PreprocessingSchema, DatasetResponseSchema,
    TaskType, StatusProcess, ModelResponseSchema, ViewModels
)
from sqlalchemy import Boolean, String
from app.utils.security import safe_path_join
from typing import Optional, List
import sqlalchemy as sa,pandas as pd,os, pathlib, datetime
from app.config import Config

class Dataset(SQLModel, table=True):
  id: int = Field(default=None, primary_key=True, nullable=False)
  task_type: TaskType = Field(
      sa_column=Column(sa.Enum(TaskType), nullable=False))
  description: str = Field(sa_column=Column(String, default=""))
  name: str = Field(sa_column=Column(String, unique=True, nullable=False))
  features: List[str] = Field(default_factory=list, sa_column=Column(JSON))
  target: str  # is n_cluster if task_type is clustering
  status: StatusProcess = Field(sa_column=Column(
      sa.Enum(StatusProcess), nullable=False, default=StatusProcess.PENDING))
  top_model: Optional[str]
  start_date: str  # 20250827
  end_date: str   # 20250127
  time_start: str  # 00:00:00
  time_end: str   # 23:59:00
  interval: int   # minutes
  preprocessing: Optional[PreprocessingSchema] = Field(default=None, sa_column=Column(JSON))
  is_valid: bool = Field(sa_column=Column(Boolean), default=False)
  meta: Optional[MetaDataset] = Field(default={}, sa_column=Column(JSON))
  n_models:int
  # One-to-many ke ModelML
  models: List["ModelML"] = Relationship(back_populates="dataset", sa_relationship_kwargs={
                                         "cascade": "all, delete-orphan"})

  def to_response(self) -> DatasetResponseSchema:
    features = [f for f in self.features]
    target = self.target

    return DatasetResponseSchema(names=self.name, description=self.description, task_type=self.task_type.name,
                                 features=features, target=target,
                                 start_date=self.start_date, end_date=self.end_date,
                                 meta=self.meta, is_valid=self.is_valid, interval=self.interval,
                                 status=self.status.name, top_model=self.top_model, models=self.get_models(),n_models=self.n_models,preprocessing=self.preprocessing)

  @classmethod
  def to_responses(cls, datas, db: Session):
    total = db.exec(select(func.count(cls.id))).one()
    n_datas = len(datas)
    return {"recordsTotal": total, "recordsFiltered": n_datas, "data": datas}

  def open_dataframe(self) -> pd.DataFrame:
    path = Config.dir/"storages"/self.name/"data.csv"
    if not os.path.exists(path):
      raise FileNotFoundError(f"{path} is not found!")
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["dt"])
    return df

  def check_path(self) -> bool:
    path = Config.dir/self.name
    if not os.path.exists(path):
      return False
    if not os.path.exists(path/"data.csv"):
      return False
    return True

  def check_features(self, X: dict) -> bool:
    columns = set(X.keys())
    if set(self.features) != columns:
      return False
    if not all(len(x) for x in list(X.values())):
      return False
    return True

  def save(self, db: Session):
    """
    Menyimpan atau meng-update instance ini ke database.
    """
    if self.id is None:
      db.add(self)
    else:
      existing = db.get(Dataset, self.id)
      if existing:
        for field, value in self.model_dump().items():
          setattr(existing, field, value)
      else:
        db.add(self)
    db.commit()
    db.refresh(self)

  def get_models(self):
    if not self.models or not self.is_valid: return []
    else: return [i.view_model() for i in self.models if i.is_active]

  def check_integrity(self):
    from fastapi import HTTPException
    """Validate dataset integrity with security checks."""
    if not self.is_valid: raise HTTPException(status_code=400, detail="Dataset is not valid")
    if not self.features: raise HTTPException(status_code=400, detail="Features not found")
    if not self.meta: raise HTTPException(status_code=400, detail="Dataset has no metadata")

    # Use secure path joining
    try: safe_path_join(Config.dir, "storages", self.name, "data.csv")
    except Exception as e: raise HTTPException(status_code=400, detail=str(e))

  @classmethod
  def _stmt(cls, *fields): return select(*fields) if fields else select(cls)

  @classmethod
  def get_by_id(cls, id, db: Session, * fields): return db.exec(cls._stmt(*fields).where(cls.id == id)).first()

  @classmethod
  def get_by_name(cls, name, db: Session, *fields): return db.exec(cls._stmt(*fields).where(cls.name == name)).first()

  @classmethod
  def get_by_task_type(cls, task_type, db: Session, *fields):
    query = cls._stmt(*fields).where(cls.task_type.contains(task_type))
    return db.exec(query).all()

  @classmethod
  def get_all(cls, db: Session, to_response: bool = True):
    rows = db.exec(cls._stmt()).all()
    if not rows:
      return None
    if to_response:
      return [row.to_response() for row in rows]
    return rows

  @classmethod
  def delete_all(cls, db):
    db.exec(delete(cls))
    db.commit()

  def is_time_to_finetune(self) -> bool:
    now = datetime.datetime.now()
    current_dt = datetime.datetime.fromisoformat(self.meta.get("current_dt",now.isoformat()))
    if self.preprocessing:
      interval_finetune = self.preprocessing['interval_finetune']
      time_to_finetune = current_dt + datetime.timedelta(days=interval_finetune)
      if time_to_finetune.date() == now.date() or time_to_finetune.date() < now.date(): return True
      else: return False
    else: return False


class ModelML(SQLModel, table=True):
  id: int = Field(primary_key=True, nullable=False)
  dataset_id: int = Field(foreign_key="dataset.id")
  name: str = Field(nullable=False)
  algorithm: str = Field(sa_column=Column(String, nullable=False))
  is_active: bool = Field(sa_column=Column(Boolean), default=True)
  evaluation: dict = Field(sa_column=Column(JSON))
  finetune: bool = Field(sa_column=Column(Boolean), default=False)
  meta: Optional[dict] = Field(default=None, sa_column=Column(JSON))
  status: str = Field(sa_column=Column(sa.Enum(StatusProcess), nullable=False, default=StatusProcess.PENDING))
  path: str = Field(sa_column=Column(String, default=None))

  dataset: Dataset = Relationship(back_populates="models")

  def to_response(self) -> ModelResponseSchema:
    return ModelResponseSchema(name=self.name, dataset_name=self.dataset.name,
                               algorithm=self.algorithm, is_active=self.is_active,
                               finetune=self.finetune, evaluation=self.evaluation, meta=self.meta, status=self.status.name)

  def check_path(self) -> bool:
    path = pathlib.Path(self.path)
    if not os.path.exists(f"{path}.pkl"):
      return False
    return True

  def save(self, db: Session):
    """
    Menyimpan atau meng-update instance ini ke database.
    """
    if self.id is None:
      db.add(self)
    else:
      existing = db.get(ModelML, self.id)
      if existing:
        for field, value in self.model_dump().items():
          setattr(existing, field, value)
      else:
        db.add(self)
    db.commit()
    db.refresh(self)

  def view_model(self) -> dict:
    desc = self.dataset.task_type.algorithms().get(self.algorithm, "")
    size = self.meta.get("size_of", "")
    return ViewModels(name=self.name, evaluation=self.evaluation, size=size, status=str(self.status),
                      algorithm=self.algorithm, description=desc, path=self.path).model_dump()

  @classmethod
  def _stmt(cls, *fields): return select(*fields) if fields else select(cls)

  @classmethod
  def get_by_id(cls, id, db: Session, *
                fields): return db.exec(cls._stmt(*fields).where(cls.id == id)).first()

  @classmethod
  def get_by_name(cls, name, db: Session, *fields): return db.exec(cls._stmt(*fields).where(cls.name == name)).first()


if __name__ == "__main__": pass
