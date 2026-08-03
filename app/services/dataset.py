"""
Dataset service layer.

Thin operations built on top of the `Dataset` model (app.database.DB).
Route handlers stay small by delegating the real work here.
"""
from app.database.DB import Dataset
from app.database.schemas import CreateDatasetSchema, TaskType
from app.exceptions import ValidationException
from app.helpers import pca


def create_dataset(payload: CreateDatasetSchema, db) -> Dataset:
  """Validate the request and build a new (unsaved) Dataset record."""
  task_type = payload.tt()
  if task_type.is_supervised() and not payload.target: raise ValidationException("Target is required for a supervised task")
  q = Dataset.create_from_request(payload, db)
  q.meta.tags = payload.tags
  return q


def delete_dataset(name: str, db, logger=None) -> dict:
  """Delete a single dataset, or every dataset when name == 'ALL-'."""
  if name == "ALL-":
    Dataset.remove_all(db)
    return {"detail": "All datasets deleted"}

  dataset = Dataset.get_by_name(name, db)
  dataset.remove(db)
  if logger:
    logger.info(f"Dataset deleted: {name}")
  return {"detail": "Dataset deleted", "dataset_name": name}


def delete_datasets_by_task_type(task_type: TaskType, db) -> dict:
  """Delete all datasets belonging to a task type."""
  datasets = Dataset.get_by_task_type(task_type, db)
  for dataset in datasets:
    dataset.remove(db)
  return {"detail": f"Deleted {len(datasets)} datasets", "task_type": task_type.name}


def sample_dataset(dataset: Dataset, n: int = 10) -> list:
  """Return a random sample of rows as records."""
  dataset.check_integrity()
  return dataset.get_df_sample(n).to_dict(orient="records")


def describe_dataset(dataset: Dataset) -> dict:
  """Return pandas `describe()` statistics for the dataset."""
  dataset.check_integrity()
  return dataset.get_df().describe().to_dict()


def reduce_dimensions(dataset: Dataset, n_components: int = 2) -> dict:
  """Compute a PCA projection of the dataset."""
  dataset.check_integrity()
  df = dataset.get_df()
  model, transformed = pca(df, n_components)
  return {
    "n_components": n_components,
    "explained_variance": model.explained_variance_ratio_.tolist(),
    "points": transformed.tolist(),
  }
