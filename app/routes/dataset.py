"""
Dataset logic module
Handles dataset operations with security and proper error handling
"""

from fastapi import HTTPException
from app.database.orm import Dataset
from app.database.schemas import DatasetRequestSchema, StatusProcess, TaskType
from app.database.influx import get_influx_storage
from app.config import Config
from app.utils.security import safe_path_join
import os
import uuid
import shutil
import pandas as pd
from typing import Optional
from app.helpers import init_storages_dataset, pca
from app.workers.manager import WorkerManager



def check_integrity_dataset(dataset: Dataset):
  """Validate dataset integrity with security checks."""
  if not dataset.is_valid:
    raise HTTPException(status_code=400, detail="Dataset is not valid")
  if not dataset.features:
    raise HTTPException(status_code=400, detail="Features not found")
  if not dataset.meta:
    raise HTTPException(status_code=400, detail="Dataset has no metadata")

  # Use secure path joining
  try:
    safe_path_join(Config.dir, "storages", dataset.name, "data.csv")
  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))


def create_dataset(payload: DatasetRequestSchema, db):
  """Create new dataset with secure path handling."""
  name = f"{payload.task_type}-{str(uuid.uuid4())[:8]}"

  # Validate task type
  if isinstance(payload.task_type, TaskType):
    if payload.task_type.is_supervised():
      if not payload.target:
        raise HTTPException(status_code=400, detail="Target is required")

  # Check if name already exists
  if Dataset.get_by_name(name, db):
    raise HTTPException(status_code=400, detail="Dataset already exists")

  # Initialize storage securely
  init_storages_dataset(name)

  # Create dataset record
  q = Dataset(name=name, is_valid=False,
              status=StatusProcess.PENDING, **payload.model_dump())
  return q


def clean_all_datasets(datasets, db) -> dict:
  """Remove all datasets and their files."""
  if not datasets:
    raise HTTPException(status_code=400, detail="No datasets to delete")

  for d in datasets:
    try:
      filepath = safe_path_join(Config.dir, "storages", d.name)
      if os.path.exists(filepath):
        shutil.rmtree(filepath)
    except Exception as e:
      print(f"Warning: Could not delete dataset files for {d.name}: {e}")

    # Delete worker state if exists
    try:
      WorkerManager.delete_by_dataset(d.name)
    except Exception as e:
      print(f"Warning: Could not delete worker state for {d.name}: {e}")

    # Delete InfluxDB prediction results
    try:
      influx_client = get_influx_storage()
      influx_client.delete_dataset(d.name)
    except Exception as e:
      print(
          f"Warning: Could not delete InfluxDB predictions for {d.name}: {e}")

    db.delete(d)
    db.commit()

  return {"detail": f"Successfully removed {len(datasets)} datasets"}


def delete_dataset(name, db, logger=None):
  """Delete a single dataset by name."""
  if logger:
    logger.info("Deleting dataset...")

  if name == "ALL-":
    datasets = db.query(Dataset).all()
    if logger: logger.info(f"Deleting all {len(datasets)} datasets")
    return clean_all_datasets(datasets, db)

  dataset = Dataset.get_by_name(name, db)
  if not dataset:
    raise HTTPException(status_code=404, detail="Dataset not found")

  dataset_name = dataset.name

  try:
    path = safe_path_join(Config.dir, "storages", dataset.name)
    if os.path.exists(path): shutil.rmtree(path)
  except Exception as e: print(f"Warning: Could not delete dataset files for {dataset_name}: {e}")

  # Delete worker state if exists
  try:
    WorkerManager.delete_by_dataset(dataset_name)
    if logger: logger.info(f"Worker state deleted for dataset: {dataset_name}")
  except Exception as e:
    if logger: logger.warning(f"Could not delete worker state for {dataset_name}: {e}")

  # Delete InfluxDB prediction results
  try:
    influx_client = get_influx_storage()
    influx_client.delete_dataset(dataset_name)
    if logger: logger.info(f"InfluxDB predictions deleted for dataset: {dataset_name}")
  except Exception as e:
    if logger:
      logger.warning(
          f"Could not delete InfluxDB predictions for {dataset_name}: {e}"
      )

  db.delete(dataset)
  db.commit()

  if logger: logger.info(f"Dataset deleted successfully | dataset_name: {dataset_name}")

  return {"detail": "Dataset deleted successfully", "dataset_name": dataset_name}


def get_df_sample(dataset: Optional[Dataset], n_samples):
  """Get a random sample from dataset."""
  if not dataset:
    raise HTTPException(status_code=404, detail="Dataset not found")

  check_integrity_dataset(dataset)

  try:
    pathfile = safe_path_join(Config.dir, "storages", dataset.name, "data.csv")
  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))

  df = pd.read_csv(pathfile)
  return df.sample(n=n_samples)


def sample_by_class(dataset: Dataset, n_samples: int):
  """Get stratified sample by class."""
  check_integrity_dataset(dataset)

  try:
    pathfile = safe_path_join(Config.dir, "storages", dataset.name, "data.csv")
  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))

  df = pd.read_csv(pathfile)

  if dataset.target not in df.columns:
    raise HTTPException(status_code=400, detail="Target column not found")

  classes = df[dataset.target].unique()
  n_samples_per_class = n_samples // len(classes)

  samples = []
  for c in classes:
    samples.append(df[df[dataset.target] == c].sample(n=n_samples_per_class))

  return pd.concat(samples)


def split_data(payload):
  """Split dataset into train/test with security validation."""
  train_size = payload.train_size
  dataset = payload.dataset

  check_integrity_dataset(dataset)

  try:
    pathfile = safe_path_join(Config.dir, "storages", dataset.name, "data.csv")
  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))

  df = pd.read_csv(pathfile)

  if dataset.target not in df.columns:
    raise HTTPException(status_code=400, detail="Target column not found")

  train = df.sample(frac=train_size, random_state=42)
  test = df.drop(train.index)

  # Save splits securely
  base_path = safe_path_join(Config.dir, "storages", dataset.name)
  train.to_csv(base_path / "train.csv", index=False)
  test.to_csv(base_path / "test.csv", index=False)

  return {"train": len(train), "test": len(test)}


def get_by_name(name: str, db):
  """Get dataset by name with validation."""
  dataset = Dataset.get_by_name(name, db)
  if not dataset:
    raise HTTPException(status_code=404, detail="Dataset not found")
  return dataset


def processing_pca(payload, db, background_tasks):
  """Process PCA with background task."""
  dataset = Dataset.get_by_name(payload.dataset_name, db)
  if not dataset: raise HTTPException(status_code=404, detail="Dataset not found")

  check_integrity_dataset(dataset)

  try: pathfile = safe_path_join(Config.dir, "storages", dataset.name, "data.csv")
  except Exception as e: raise HTTPException(status_code=400, detail=str(e))

  df = pd.read_csv(pathfile)

  if payload.columns:
    selected_features = [ f for f in dataset.features if f in payload.columns and f != dataset.target ]
  else:
    selected_features = [f for f in dataset.features if f != dataset.target]

  def process():
    df_transformed, components, variance = pca( df, payload.n_components, dataset.target, selected_features)
    # Update dataset with PCA info
    dataset.meta = {
        **(dataset.meta or {}),
        "pca": {
            "components": components,
            "variance": variance,
            "n_components": payload.n_components,
        },
    }
    db.add(dataset)
    db.commit()

  background_tasks.add_task(process)

  return {"status": "processing", "dataset_name": payload.dataset_name}


def get_all_datasets(db):
  """Get all datasets."""
  return db.query(Dataset).all()


def get_dataset_by_name(name: str, db):
  """Get dataset by name."""
  return Dataset.get_by_name(name, db)


def get_df_describe(dataset: Dataset):
  """Get dataset statistics using pandas describe()."""
  if not dataset:
    raise HTTPException(status_code=404, detail="Dataset not found")

  check_integrity_dataset(dataset)

  try:
    pathfile = safe_path_join(Config.dir, "storages", dataset.name, "data.csv")
  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))

  df = pd.read_csv(pathfile)
  return df.describe().to_dict()


def dim_reduce(dataset: Dataset):
  """Get PCA dimensionality reduction results."""
  if not dataset:
    raise HTTPException(status_code=404, detail="Dataset not found")

  check_integrity_dataset(dataset)

  # Return existing PCA results if available
  if dataset.meta and "pca" in dataset.meta:
    return dataset.meta["pca"]

  raise HTTPException(
      status_code=404, detail="PCA not processed for this dataset")
