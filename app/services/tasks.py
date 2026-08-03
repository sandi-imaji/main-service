"""
Background jobs.

FastAPI closes the request-scoped DB session as soon as the response is sent,
so jobs scheduled with `BackgroundTasks` must NOT reuse it. Every job here opens
its own session and re-fetches the dataset by name, keeping background work
isolated from the request lifecycle.
"""
from app.database.base import get_db_session
from app.database.DB import Dataset
from app.database.schemas import AnomalyRequestSchema, StatusProcess
from app.logger import Logger
from app.pull import pulling
from app.services import anomaly as anomaly_service
from app.services import rebuild as rebuild_service
from app.services import timeseries as timeseries_service
from app.services import train as train_service


def pull_dataset(dataset_name: str):
  """Pull raw data for a freshly created dataset."""
  with get_db_session() as db:
    pulling(dataset_name, db)


def initialize_dataset(dataset_name: str, n_models: int):
  """Pull data, then train the top models only when n_models > 0.

  n_models == 0 means "just pull & save the dataset" (no training) — the pull
  itself persists the dataset, so there is nothing more to do.
  """
  with get_db_session() as db:
    pulling(dataset_name, db)
    if n_models <= 0:
      return
    dataset = Dataset.get_by_name(dataset_name, db)
    if dataset.is_valid and dataset.status == StatusProcess.SUCCESS_PULL:
      train_service.find_top_models(dataset, n_models, db, Logger(dataset_name))


def rebuild_dataset(dataset_name: str, n_models: int = 0):
  """Bangun ulang penuh: hapus worker + artefak, polling ulang, cari model."""
  try:
    rebuild_service.rebuild(dataset_name, n_models)
  except Exception as e:
    # Tidak ada pemanggil yang tersisa — respons HTTP sudah dikirim. Kalau lolos
    # dari sini, galatnya hilang ke lapisan ASGI tanpa jejak di log dataset.
    Logger(dataset_name).error(f"Rebuild gagal: {e}")


def retrain_dataset(dataset_name: str, n_models: int = 0):
  """Latih ulang di atas data yang ada: hapus worker + artefak, cari model."""
  try:
    rebuild_service.retrain(dataset_name, n_models)
  except Exception as e:
    Logger(dataset_name).error(f"Retrain gagal: {e}")


def find_top_models(dataset_name: str, n_top: int):
  """Search and persist the best-performing models for a dataset."""
  with get_db_session() as db:
    dataset = Dataset.get_by_name(dataset_name, db)
    train_service.find_top_models(dataset, n_top, db, Logger(dataset_name))


def train_one_model(dataset_name: str, algorithm: str, use_this: bool):
  """Train a single algorithm for a dataset."""
  with get_db_session() as db:
    dataset = Dataset.get_by_name(dataset_name, db)
    train_service.train_model(dataset, algorithm, db, use_this=use_this)


def run_auto_anomaly(payload: AnomalyRequestSchema):
  """Train an anomaly model and start its worker."""
  with get_db_session() as db:
    dataset = Dataset.get_by_name(payload.dataset_name, db)
    anomaly_service.auto_anomaly(payload, dataset)


def finetune_forecast(dataset_name: str):
  """Refit an expired time-series dataset.

  `finetune` raises when the fresh pull is empty (so the streamer's error
  counter can see it). Here there is no caller left to catch it — the HTTP
  response is already sent — so it is logged instead of escaping into the ASGI
  layer as an unhandled background-task error.
  """
  try:
    timeseries_service.finetune(dataset_name)
  except Exception as e:
    Logger(dataset_name).error(f"Finetune gagal: {e}")


def refresh_forecast_actuals(dataset_name: str):
  """Pull the latest actuals for a time-series dataset from InfluxDB."""
  with get_db_session() as db:
    dataset = Dataset.get_by_name(dataset_name, db)
    timeseries_service.get_actual_save_all(dataset, Logger(dataset_name))
