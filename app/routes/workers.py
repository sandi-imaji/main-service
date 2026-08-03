"""Worker task endpoints (prefix: /workers)."""
from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.database.base import get_session
from app.database.schemas import TaskType
from app.exceptions import NotFoundException
from app.workers.manager import WorkerManager
from app.routes.base import get_dataset_or_404

router = APIRouter(tags=["workers"])


def _require_task(task_name: str):
  task = WorkerManager.get_tasks(task_name)
  if not task:
    raise NotFoundException("Task", task_name)
  return task


@router.get("/tasks")
async def list_tasks():
  """List all worker tasks."""
  return WorkerManager.get_tasks()


@router.get("/task")
async def get_task(dataset_name: str):
  """List worker tasks for a dataset."""
  return WorkerManager.get_tasks(dataset_name)


@router.post("/start")
async def start_task(task_name: str):
  """Start a worker task."""
  _require_task(task_name)
  if WorkerManager.is_active(task_name):
    return {"detail": f"Task {task_name} is still Active"}
  WorkerManager.start(task_name)
  return {"detail": f"{task_name} is start ..."}


@router.post("/create")
async def create_task(dataset_name: str, is_anomaly: bool = False, db: Session = Depends(get_session)):
  """Create a worker task for a dataset."""
  dataset = get_dataset_or_404(dataset_name, db)
  task_type = TaskType.Anomaly if is_anomaly else dataset.task_type
  task_name = f"{dataset_name}-{task_type}"
  if WorkerManager.is_active(task_name):
    return {"detail": f"{task_name} is still active"}
  WorkerManager.create(dataset_name, task_type)
  return {"detail": f"create task {task_name} ..."}


@router.post("/stop")
async def stop_task(task_name: str):
  """Stop a worker task."""
  _require_task(task_name)
  WorkerManager.stop(task_name)
  return {"detail": f"{task_name} is stop ..."}


@router.post("/stop-by-dataset")
async def stop_task_by_dataset(dataset_name: str, task_type: str):
  """Stop a worker task identified by dataset name and task type."""
  task_name = f"{dataset_name}-{task_type}"
  _require_task(task_name)
  WorkerManager.stop(task_name)
  return {"detail": f"{task_name} is stop ..."}


@router.delete("/delete")
async def delete_task(task_name: str):
  """Delete a worker task."""
  _require_task(task_name)
  WorkerManager.delete(task_name)
  return {"detail": f"{task_name} is delete ..."}
