"""Model endpoints (prefix: /models)."""
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlmodel import Session, select

from app.database.base import get_session
from app.database.DB import Dataset, ModelML
from app.database.historicalDB import InfluxDBStorage, get_influx_storage
from app.database.schemas import (
  AnomalyRequestSchema,
  CreateDatasetSchema,
  InferenceRequestSchema,
  ModelTrainSchema,
  TaskType,
)
from app.exceptions import InvalidStateException, ValidationException
from app.logger import Logger, LOGGER_GLOBAL
from app.routes.base import get_dataset_or_404, get_model_or_404
from app.services import clustering as clustering_service
from app.services import dataset as dataset_service
from app.services import model as model_service
from app.services import anomaly as anomaly_service
from app.services import dispatch as dispatch_service
from app.services import live_performance
from app.services import tasks
from app.services import timeseries as timeseries_service

router = APIRouter(tags=["models"])


# --- Initialization & inference -------------------------------------------

@router.post("/auto-initiate")
async def auto_initiate(
  payload: CreateDatasetSchema,
  background_task: BackgroundTasks,
  db: Session = Depends(get_session),
):
  """Create a dataset and train its top models in the background."""
  dataset = dataset_service.create_dataset(payload, db)
  dataset.n_models = payload.n_models
  dataset.save(db)
  LOGGER_GLOBAL.debug(f"Auto-initiating dataset: {dataset.name}")
  background_task.add_task(tasks.initialize_dataset, dataset.name, payload.n_models)
  return payload.model_dump()


@router.post("/auto-inference")
async def run_auto_inference(dataset_name: str, db: Session = Depends(get_session)):
  """Run inference using a dataset's top model."""
  dataset = get_dataset_or_404(dataset_name, db)
  logger = Logger(dataset_name)
  logger.debug(f"auto-inference {dataset_name} | target: {dataset.target}")
  return dispatch_service.auto_inference(dataset, logger).model_dump()


@router.post("/inference")
async def run_inference(payload: InferenceRequestSchema, db: Session = Depends(get_session)):
  """Run inference on a dataset with an explicit input payload."""
  return model_service.inference(payload, db)


@router.post("/anomaly/auto")
async def run_auto_anomaly(
  payload: AnomalyRequestSchema,
  background_task: BackgroundTasks,
  db: Session = Depends(get_session),
):
  """Train an anomaly model and start its worker in the background."""
  get_dataset_or_404(payload.dataset_name, db)
  background_task.add_task(tasks.run_auto_anomaly, payload)
  return payload.model_dump()


@router.get("/{dataset_name}/clusters")
async def get_clusters(dataset_name: str, db: Session = Depends(get_session)):
  """Get clustering results for a dataset."""
  dataset = get_dataset_or_404(dataset_name, db)
  logger = Logger(dataset.name)
  try:
    return clustering_service.get_clusters(dataset, logger)
  except FileNotFoundError as e:
    raise HTTPException(status_code=404, detail=str(e))
  except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))
  except Exception as e:
    logger.error(f"Error getting clusters: {e}")
    raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@router.post("/{dataset_name}/rebuild")
async def rebuild_dataset(
  dataset_name: str,
  background_tasks: BackgroundTasks,
  n_models: int = 0,
  db: Session = Depends(get_session),
):
  """Rebuild a dataset from scratch: re-pull the data, then search for the best
  models.

  Every worker of the dataset is dropped first, and so are its models and the
  contents of `results/` — the outcome is a genuinely new generation, not a pile
  on top of the old one. Workers are NOT restarted automatically; start them
  yourself once you have seen the result.

  `n_models` is optional; 0 means use the dataset's own `n_models`.
  """
  dataset = get_dataset_or_404(dataset_name, db)
  LOGGER_GLOBAL.info(f"Rebuild requested: {dataset.name}")
  background_tasks.add_task(tasks.rebuild_dataset, dataset.name, n_models)
  return {"status": "rebuilding", "dataset": dataset.name,
          "message": "Rebuild running in the background: workers dropped, data re-pulled, models retrained"}


@router.post("/{dataset_name}/retrain")
async def retrain_dataset(
  dataset_name: str,
  background_tasks: BackgroundTasks,
  n_models: int = 0,
  db: Session = Depends(get_session),
):
  """Retrain the models on the data ALREADY on disk — no re-pulling.

  Identical to rebuild in what it cleans up (workers, models, `results/`); it
  simply does not fetch new data.
  """
  dataset = get_dataset_or_404(dataset_name, db)
  if not dataset.is_valid:
    raise InvalidStateException("Dataset", "Not Valid", "Valid")
  LOGGER_GLOBAL.info(f"Retrain requested: {dataset.name}")
  background_tasks.add_task(tasks.retrain_dataset, dataset.name, n_models)
  return {"status": "retraining", "dataset": dataset.name,
          "message": "Retrain running in the background: workers dropped, models retrained on existing data"}


@router.get("/{dataset_name}/anomaly/metrics")
async def get_anomaly_metrics(dataset_name: str, db: Session = Depends(get_session)):
  """Detection summary from the last anomaly training on this dataset.

  Anomaly has no `ModelML` row, so its metrics cannot ride along in
  `model.evaluation` like the other tasks — they are stored beside the model
  artifact and read from here.

  A `{}` response means it has never been trained (or was trained before this
  summary existed); that is a legitimate state, not an error.
  """
  dataset = get_dataset_or_404(dataset_name, db)
  return anomaly_service.read_metrics(dataset)


@router.get("/{dataset_name}/performance/live")
async def get_live_performance(
  dataset_name: str,
  start: str | None = None,
  end: str | None = None,
  limit: int = 1000,
  db: Session = Depends(get_session),
):
  """Model performance from predictions versus realised actuals.

  The metrics on `ModelML.evaluation` are frozen at training time; this endpoint
  answers a different question — is the model STILL accurate today. Each
  algorithm is reported alongside its training error and the ratio between them,
  so degradation shows up as a number rather than a hunch.

  TimeSeries only: other tasks never learn the true value of the points they
  predicted, so there is nothing to compare against.
  """
  dataset = get_dataset_or_404(dataset_name, db)
  return live_performance.running_metrics(dataset, start=start, end=end, limit=limit)


@router.get("/{dataset_name}/clusters/report")
async def get_clusters_report(
  dataset_name: str,
  start: str | None = None,
  end: str | None = None,
  db: Session = Depends(get_session),
):
  """Clustering summary for a single date range (composition + episodes).

  Replaces `/clusters` for the analysis page: what goes over the wire is a
  summary rather than every row — tens of KB instead of hundreds.
  """
  dataset = get_dataset_or_404(dataset_name, db)
  if not dataset.task_type.is_clustering():
    raise ValidationException(f"Dataset '{dataset_name}' is not a Clustering dataset")
  try:
    return clustering_service.report(dataset, start=start, end=end)
  except FileNotFoundError as e:
    raise HTTPException(status_code=404, detail=str(e))
  except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))


@router.get("/{dataset_name}/forecast")
async def get_forecast(
  dataset_name: str,
  background_tasks: BackgroundTasks,
  db: Session = Depends(get_session),
):
  """Get a time-series forecast, refreshing the model if it is expired."""
  dataset = get_dataset_or_404(dataset_name, db)
  logger = Logger(dataset.name)
  if timeseries_service.is_expired(dataset, logger):
    logger.info("Dataset is expired, triggering update...")
    background_tasks.add_task(tasks.finetune_forecast, dataset_name)
    return {"status": "updating", "message": "Dataset is expired, update in progress"}
  return timeseries_service.forecast(dataset, logger)


@router.get("/{dataset_name}/forecast/get_actual")
async def refresh_forecast_actuals(
  dataset_name: str,
  background_tasks: BackgroundTasks,
  db: Session = Depends(get_session),
):
  """Refresh a time-series dataset's actual values from InfluxDB."""
  dataset = get_dataset_or_404(dataset_name, db)
  if dataset.task_type != TaskType.TimeSeries:
    raise HTTPException(status_code=400, detail="Dataset is not a Time Series")
  Logger(dataset.name).info("Updating TimeSeries data from InfluxDB ...")
  background_tasks.add_task(tasks.refresh_forecast_actuals, dataset_name)
  return {"status": "Fetching actual data ..."}


# --- Training --------------------------------------------------------------

@router.post("/find-top")
async def find_top_models(
  dataset_name: str,
  background_tasks: BackgroundTasks,
  db: Session = Depends(get_session),
):
  """Search for the best-performing models in the background."""
  dataset = get_dataset_or_404(dataset_name, db)
  if not dataset.is_valid:
    raise InvalidStateException("Dataset", "Not Valid", "Valid")
  Logger(dataset.name).info(f"Find Top Model: {dataset.name} | n_top: {dataset.n_models}")
  background_tasks.add_task(tasks.find_top_models, dataset.name, dataset.n_models)
  return {"status": "Processing", "message": "Finding top models in background", "dataset": dataset.name}


@router.post("/train")
async def train_model(
  payload: ModelTrainSchema,
  background_tasks: BackgroundTasks,
  db: Session = Depends(get_session),
):
  """Train a single model in the background."""
  dataset = get_dataset_or_404(payload.dataset_name, db)
  background_tasks.add_task(tasks.train_one_model, dataset.name, payload.algorithm, payload.use_this)
  return {"status": "training", "message": f"Training {payload.algorithm} in background", "dataset": dataset.name}


@router.post("/{dataset_name}/change-top/{algorithm}")
async def change_top_model(dataset_name: str, algorithm: str, db: Session = Depends(get_session)):
  """Set a dataset's top model to the given algorithm."""
  dataset = get_dataset_or_404(dataset_name, db)
  model_service.change_top_model(dataset, algorithm)
  db.commit()
  return {"status": "success", "message": f"Top model changed to {algorithm}", "dataset": dataset_name}


# --- Model queries ---------------------------------------------------------

@router.get("", response_model=List[dict])
async def list_models(db: Session = Depends(get_session)):
  """List all trained models."""
  models = db.exec(select(ModelML)).all()
  return [m.to_response() for m in models]


@router.get("/algorithms/{task_type}")
async def list_algorithms(task_type: str):
  """List available algorithms for a task type."""
  try:
    return TaskType.from_str(task_type).algorithms()
  except Exception:
    raise ValidationException(f"Invalid task type: {task_type}")


@router.get("/by-dataset/{dataset_name}")
async def list_models_by_dataset(dataset_name: str, db: Session = Depends(get_session)):
  """List all models belonging to a dataset."""
  return get_dataset_or_404(dataset_name, db).models


@router.get("/history/anomaly")
async def get_all_anomaly_history(n_day: int = 3):
  """Get anomaly prediction history across all datasets."""
  with InfluxDBStorage() as influx:
    data = influx.query_inference(start=f"-{24 * n_day}h", end=f"{24 * n_day}h", task_type="Anomaly")
    if not data:
      raise HTTPException(status_code=404, detail="Data is not found!")
    return data


@router.get("/{dataset_name}/history")
async def get_prediction_history(dataset_name: str, db: Session = Depends(get_session)):
  """Get a dataset's prediction history (from InfluxDB, or clusters for clustering)."""
  try:
    dataset = get_dataset_or_404(dataset_name, db)
    LOGGER_GLOBAL.info(f"Get Predictions History: {dataset.name} | {dataset.task_type}")
    if dataset.task_type.is_clustering():
      return clustering_service.get_clusters(dataset, Logger(dataset_name))
    influx = get_influx_storage()
    return influx.query_inference(dataset_name=dataset_name, task_type=str(dataset.task_type))
  except HTTPException:
    raise
  except FileNotFoundError as e: raise HTTPException(status_code=404, detail=str(e))
  except ValueError as e: raise HTTPException(status_code=400, detail=str(e))
  except Exception as e:
    LOGGER_GLOBAL.error(str(e))
    raise HTTPException(status_code=500, detail=str(e))

@router.get("/{dataset_name}/history/anomaly")
async def get_anomaly_history(dataset_name: str, n_day: int = 3):
  """Get a dataset's anomaly prediction history."""
  with InfluxDBStorage() as influx:
    data = influx.query_inference(dataset_name=dataset_name, task_type="Anomaly")
    if not data:
      raise HTTPException(status_code=404, detail="Data is not found!")
    return data


@router.get("/{model_name}")
async def get_model(model_name: str, db: Session = Depends(get_session)):
  """Get a model by name."""
  return get_model_or_404(model_name, db).to_response()


# --- Cleanup ---------------------------------------------------------------

@router.delete("/by-dataset/{dataset_name}")
async def delete_models(dataset_name: str, db: Session = Depends(get_session)):
  """Delete all models belonging to a dataset."""
  dataset = get_dataset_or_404(dataset_name, db)
  count = model_service.clean_models(dataset, db)
  return {"status": "success", "message": f"Cleaned {count} models", "dataset": dataset_name}


@router.delete("/{dataset_name}/results")
async def clean_model_results(dataset_name: str, db: Session = Depends(get_session)):
  """Clean a dataset's inference results."""
  dataset = get_dataset_or_404(dataset_name, db)
  model_service.clean_results(dataset)
  return {"status": "success", "message": "Results cleaned", "dataset": dataset_name}
