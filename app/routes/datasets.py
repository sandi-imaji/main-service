"""Dataset endpoints (prefix: /datasets)."""
from typing import Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from app.database.base import get_session
from app.database.DB import Dataset
from app.database.schemas import CreateDatasetSchema, TaskType
from app.exceptions import ValidationException
from app.helpers import read_logs
from app.workers.manager import WorkerManager
from app.routes.base import get_dataset_or_404, with_logging
from app.services import clustering as clustering_service
from app.services import dataset as dataset_service
from app.services import dataset_table
from app.services import tasks

router = APIRouter(tags=["datasets"])


def _parse_task_type(task_type: str) -> TaskType:
  try:
    return TaskType.from_str(task_type)
  except Exception:
    raise ValidationException(f"Invalid task type: {task_type}")


# --- Queries ---------------------------------------------------------------

@router.get("", response_model=dict)
async def list_datasets(db: Session = Depends(get_session)):
  """List all datasets with pagination info."""
  datasets = Dataset.get_all(db)
  return Dataset.to_responses(datasets, db)


@router.get("/names", response_model=List[str])
async def list_dataset_names(db: Session = Depends(get_session)):
  """List all dataset names."""
  datasets = Dataset.get_all(db, to_response=False)
  return [d.name for d in datasets]


@router.get("/active", response_model=List[Dict])
async def list_active_datasets(db: Session = Depends(get_session)):
  """List datasets with their task type (and cluster names for clustering)."""
  datasets = Dataset.get_all(db, to_response=False)
  result = []
  for dataset in datasets:
    entry = {"name": dataset.name, "task_type": str(dataset.task_type)}
    if dataset.task_type.is_clustering():
      entry["clusters"] = clustering_service.get_cluster_unique(dataset)
    result.append(entry)
  return result


@router.get("/filter/{task_type}", response_model=dict)
async def filter_datasets_by_task_type(task_type: str, db: Session = Depends(get_session)):
  """Filter datasets by task type."""
  datasets = Dataset.get_by_task_type(_parse_task_type(task_type), db)
  return Dataset.to_responses(datasets, db)


@router.get("/{name}", response_model=dict)
async def get_dataset(name: str, db: Session = Depends(get_session)):
  """Get a dataset by name, including its worker status."""
  dataset = get_dataset_or_404(name, db)
  task_name = f"{dataset.name}-{dataset.task_type}"
  worker_active = WorkerManager.is_active(task_name)
  if dataset.status.is_normal():
    WorkerManager.update_flag_dataset(name, not worker_active)
  data = dataset.to_response().model_dump()
  data["worker_status"] = worker_active
  return data


@router.get("/{name}/status", response_model=dict)
async def get_dataset_status(name: str, db: Session = Depends(get_session)):
  """Get a dataset's processing status."""
  dataset = get_dataset_or_404(name, db)
  return {"status_code": dataset.status, "status_name": dataset.status.name}


@router.get("/{name}/sample")
async def get_dataset_sample(name: str, db: Session = Depends(get_session)):
  """Get a random sample of rows from a dataset."""
  return dataset_service.sample_dataset(get_dataset_or_404(name, db), 10)


@router.get("/{name}/table")
async def get_dataset_table(
  name: str,
  page: int = Query(1, ge=1),
  page_size: int = Query(dataset_table.DEFAULT_PAGE_SIZE, ge=1, le=dataset_table.MAX_PAGE_SIZE),
  sort: str | None = None,
  order: str = Query("asc", pattern="^(asc|desc)$"),
  db: Session = Depends(get_session),
):
  """Satu halaman isi dataset.

  Sumbernya bergantung task type: Clustering memakai `results/clusters.csv`
  together with the mapped cluster names; everything else serves `data.csv`. The
  source name is returned as well so the client can show where the data came
  from.
  """
  dataset = get_dataset_or_404(name, db)
  return dataset_table.page(dataset, page=page, page_size=page_size, sort=sort, order=order)


@router.get("/{name}/table/download")
async def download_dataset_table(
  name: str,
  format: str = Query("csv", pattern="^(csv|xlsx)$"),
  db: Session = Depends(get_session),
):
  """Download the WHOLE table — identical to what is shown on screen."""
  dataset = get_dataset_or_404(name, db)
  if format == "xlsx":
    buffer, filename = dataset_table.to_excel(dataset)
    media = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
  else:
    buffer, filename = dataset_table.to_csv(dataset)
    media = "text/csv"
  return StreamingResponse(buffer, media_type=media, headers={
      "Content-Disposition": f'attachment; filename="{filename}"',
  })


@router.get("/{name}/describe")
async def get_dataset_describe(name: str, db: Session = Depends(get_session)):
  """Get descriptive statistics for a dataset."""
  return dataset_service.describe_dataset(get_dataset_or_404(name, db))


@router.get("/{name}/pca")
async def get_dataset_pca(name: str, db: Session = Depends(get_session)):
  """Get a PCA projection of a dataset."""
  return dataset_service.reduce_dimensions(get_dataset_or_404(name, db))


@router.get("/{name}/logs")
async def get_dataset_logs(name: str):
  """Get a dataset's logs."""
  return {"log": read_logs(name)}


# --- Mutations -------------------------------------------------------------

@router.post("", response_model=dict)
@with_logging("create_dataset")
async def create_dataset(
  payload: CreateDatasetSchema,
  background_task: BackgroundTasks,
  db: Session = Depends(get_session),
  logger=None,
):
  """Create a new dataset and start pulling its data in the background."""
  dataset = dataset_service.create_dataset(payload, db)
  dataset.save(db)
  background_task.add_task(tasks.pull_dataset, dataset.name)
  return dataset.model_dump()


@router.delete("/{name}", response_model=dict)
@with_logging("delete_dataset")
async def delete_dataset(name: str, db: Session = Depends(get_session), logger=None):
  """Delete a dataset by name (use 'ALL-' to delete every dataset)."""
  return dataset_service.delete_dataset(name, db, logger)


@router.delete("/filter/{task_type}")
async def delete_datasets_by_task_type(task_type: str, db: Session = Depends(get_session)):
  """Delete all datasets of a given task type."""
  return dataset_service.delete_datasets_by_task_type(_parse_task_type(task_type), db)
