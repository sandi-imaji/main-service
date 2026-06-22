import pprint
from app.workers.manager import WorkerManager
from fastapi import Depends, BackgroundTasks, APIRouter, WebSocket, HTTPException, Query, WebSocketDisconnect
from sqlmodel import Session
from typing import Callable, List, Dict
from sqlmodel import Session


from app.database.db import get_session
from app.database.influx import InfluxDBStorage, get_influx_storage
from app.database.orm import Dataset, ModelML

from app.database.schemas import (
  AnomalyRequestSchema,
  DatasetRequestSchema,
  ModelTrainSchema,
  InferenceRequestSchema,
  FindTopModelRequestSchema,
  InitiateRequestSchema,
  TaskType,
)
from app.exceptions import ValidationException, NotFoundException
from app.routes.base import with_logging, get_dataset_or_404
from app.routes.dataset import (
  create_dataset,
  delete_dataset,
  clean_all_datasets,
  get_df_sample,
  get_df_describe,
  dim_reduce,
)

from app.routes.base import get_model_or_404
from app.pull import async_get_realtime, pulling, get_realtime,query_tagnames
from app.helpers import read_logs
from app.config import Config
from app.logger import Logger, LOGGER_GLOBAL

from app.routes.streamer import (
  anomaly_manager,
  regression_manager,
  clustering_manager,
  forecast_manager,
  logger_manager
)

from app.routes.modelML import (
  clean_results,
  inference,
  clean_models,
  find_top_model,
  train_model,
  change_top_model,
  auto_anomaly,
  auto_initialize, get_stats
)
from app.core.unsupervised import Unsupervised
from app.core.time_series import TimeSeries
from app.core.anomaly import Anomaly
from app.core.supervised import Supervised

dataset_router = APIRouter(tags=["datasets"])
model_router = APIRouter(tags=["models"])
stream_router = APIRouter(tags=["stream"])
worker_router = APIRouter(tags=["workers"])
utils_router = APIRouter(tags=["utils"])

# ============================================================================
# Dataset CRUD Routes
# ============================================================================


@dataset_router.get("", response_model=dict)
async def list_datasets(db: Session = Depends(get_session)):
  """List all datasets with pagination info"""
  datasets = Dataset.get_all(db)
  return Dataset.to_responses(datasets or [], db)


@dataset_router.get("/names", response_model=List[str])
async def list_dataset_names(db: Session = Depends(get_session)):
  """List all dataset names"""
  datasets = Dataset.get_all(db, to_response=False)
  if not datasets: return []
  return [d.name for d in datasets]


@dataset_router.get("/active", response_model=List[Dict])
async def active_dataset(db: Session = Depends(get_session)):
  """List only valid dataset names"""
  datasets = Dataset.get_all(db, to_response=False)
  if not datasets:
    return []
  result = []
  for data in datasets:
    d = {"name": data.name, "task_type": str(data.task_type)}
    if data.task_type.is_clustering():
      core = data.task_type.core()
      clusters = core.get_cluster_unique(data)
      d["clusters"] = clusters
    result.append(d)

  return result


@dataset_router.get("/filter/{task_type}", response_model=dict)
async def filter_by_task_type(task_type: str, db: Session = Depends(get_session)):
  """Filter datasets by task type"""
  try:
    TaskType.from_str(task_type)
  except:
    raise ValidationException(f"Invalid task type: {task_type}")

  datasets = Dataset.get_by_task_type(task_type, db)
  return Dataset.to_responses(datasets or [], db)


@dataset_router.get("/{name}", response_model=dict)
async def get_dataset(name: str, db: Session = Depends(get_session)):
  """Get dataset by name"""
  dataset = get_dataset_or_404(name, db)
  task_name = f"{dataset.name}-{dataset.task_type}"
  status = WorkerManager.is_active(task_name)
  if dataset.status.is_normal():
    flag = not status
    WorkerManager.update_flag_dataset(name,flag)
  data = dataset.to_response().model_dump()
  data["worker_status"] = status
  return data


@dataset_router.post("", response_model=dict)
@with_logging("create_dataset")
async def create_dataset_route(
  payload: DatasetRequestSchema,
  background_task: BackgroundTasks,
  db: Session = Depends(get_session),
  logger=None,
):
  """Create new dataset and trigger pulling if not dummy"""
  task_type = payload.tt()

  # Create dataset based on type
  new_dataset = create_dataset(payload, db)
  logger.info(f"Created dataset: {new_dataset.name}")

  # Save to database
  new_dataset.save(db)

  # Trigger background pulling for non-dummy datasets
  if not task_type.is_dummies():
    background_task.add_task(pulling, dataset_name=new_dataset.name)
    logger.info(f"Background pulling started for: {new_dataset.name}")

  return new_dataset.model_dump()


@dataset_router.delete("/{name}", response_model=dict)
@with_logging("delete_dataset")
async def delete_dataset_route(
  name: str, db: Session = Depends(get_session), logger=None
):
  """Delete dataset by name"""
  return delete_dataset(name, db, logger)


@dataset_router.delete("/filter/{task_type}")
async def delete_by_task_type(task_type: str, db: Session = Depends(get_session)):
  """Delete all datasets of specific task type"""
  datasets = Dataset.get_by_task_type(task_type, db)
  return clean_all_datasets(datasets, db)


# ============================================================================
# Dataset Analysis Routes
# ============================================================================


@dataset_router.get("/{name}/sample")
async def get_sample(name: str, db: Session = Depends(get_session)):
  """Get sample data from dataset"""
  dataset = get_dataset_or_404(name, db)
  return get_df_sample(dataset, 10)


@dataset_router.get("/{name}/describe")
async def get_description(name: str, db: Session = Depends(get_session)):
  """Get dataset statistics"""
  dataset = get_dataset_or_404(name, db)
  return get_df_describe(dataset)


@dataset_router.get("/{name}/pca")
async def get_pca(name: str, db: Session = Depends(get_session)):
  """Get PCA dimensionality reduction"""
  dataset = get_dataset_or_404(name, db)

  return dim_reduce(dataset)

# ============================================================================
# Utility Routes
# ============================================================================


@dataset_router.get("/utils/tagname")
async def search_tagname(query: str):
  """Search tagname from external API using async HTTP client"""
  # url = f"{Config.sl_host}application/api/tags/search.php?search={query}"
  # body = {"authtoken": "test"}
  #
  # try:
  #   res = requests.post(url, data=body, timeout=5.0, verify=False)
  #   if res.status_code != 200: return []
  #   data = res.json().get("data", [])
  #   return [item["text"].split(":")[0] for item in data]
  # except Exception as e:
  #   print(e)
  #   return []
  return query_tagnames(query)


@dataset_router.get("/utils/task-types")
async def list_task_types():
  """List supported task types"""
  return ["Regression", "Clustering", "TimeSeries", "Anomaly"]


@dataset_router.get("/utils/realtime/{tagname}")
async def get_realtime_value(tagname: str):
  """Get realtime value for specific row_id"""
  return get_realtime(tagname, logger=LOGGER_GLOBAL)


@dataset_router.get("/{name}/logs")
async def get_logs(name: str):
  """Get dataset logs"""
  return {"log": read_logs(name)}


# ============================================================================
# MODEL ROUTES
# ============================================================================

# ============================================================================
# Auto Initialization
# ============================================================================


@model_router.post("/auto-initiate")
@with_logging("auto_initiate")
async def auto_initiate(
  payload: InitiateRequestSchema,
  background_task: BackgroundTasks,
  db: Session = Depends(get_session),
  logger=None
):
  """Auto initialize dataset with training"""
  dataset_payload = DatasetRequestSchema(**payload.model_dump())

  # Create dataset
  dataset = create_dataset(dataset_payload, db)
  dataset.n_models = payload.n_models
  dataset.save(db)

  LOGGER_GLOBAL.debug(f"Auto-initiating dataset: {dataset.name}")
  background_task.add_task(
    auto_initialize, dataset=dataset, n_models=payload.n_models,db=db
  )
  # pprint.pprint(payload.model_dump())

  return payload.model_dump()


# ============================================================================
# Inference Routes
# ============================================================================


@model_router.post("/auto-inference")
async def run_auto_inference(dataset_name: str, db: Session = Depends(get_session)):
  dataset = Dataset.get_by_name(dataset_name, db)
  if not dataset:
    raise HTTPException(status_code=404, detail="dataset is not found!")
  LOGGER_GLOBAL.debug(f"auto-inference dataset {dataset_name} | target : {dataset.target}")
  logger = Logger(dataset_name)
  core = dataset.task_type.core()
  results = core.auto_inference(dataset, logger)
  return results.model_dump()

@model_router.post("/inference")
async def run_inference(
  payload: InferenceRequestSchema, db: Session = Depends(get_session)
):
  """Run inference on dataset"""
  return inference(payload, db)

@model_router.post("/anomaly/auto")
async def run_auto_anomaly(
  payload:AnomalyRequestSchema,background_task:BackgroundTasks, db: Session = Depends(get_session)):
  """
  train & start worker
  """
  dataset = get_dataset_or_404(payload.dataset_name,db)
  background_task.add_task(auto_anomaly,payload=payload,dataset=dataset)
  return payload.model_dump()


@model_router.get("/{dataset_name}/clusters")
async def get_clusters(dataset_name: str, db: Session = Depends(get_session)):
  """Get clusters for unsupervised learning"""

  dataset = get_dataset_or_404(dataset_name, db)
  logger = Logger(dataset.name)
  try: return Unsupervised.get_clusters(dataset, logger)
  except FileNotFoundError as e: raise HTTPException(status_code=404, detail=str(e))
  except ValueError as e: raise HTTPException(status_code=400, detail=str(e))
  except Exception as e:
    logger.error(f"Error getting clusters: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@model_router.get("/{dataset_name}/forecast")
async def get_forecast(
  dataset_name: str,
  background_tasks: BackgroundTasks,
  db: Session = Depends(get_session),
):
  """Get time series forecast"""
  dataset = get_dataset_or_404(dataset_name, db)
  logger = Logger(dataset.name)

  # Check if data is expired and needs update
  if TimeSeries.is_expired(dataset, logger):
    logger.info("Dataset is expired, triggering update...")
    background_tasks.add_task(TimeSeries.finetune, dataset_name=dataset_name)
    return {
      "status": "updating",
      "message": "Dataset is expired, update in progress",
    }

  return TimeSeries.inference(dataset, logger)

@model_router.get("/{dataset_name}/forecast/get_actual")
async def get_actual_save_forecast(
  dataset_name: str,
  background_tasks: BackgroundTasks,
  db: Session = Depends(get_session),
):
  """Get time series forecast"""
  dataset = get_dataset_or_404(dataset_name, db)
  if dataset.task_type != TaskType.TimeSeries: raise HTTPException(status_code=500, detail="dataset is not Time Series")

  logger = Logger(dataset.name)

  logger.info("Updating TimeSeries Data From InfluxDB ...")
  background_tasks.add_task(TimeSeries.get_actual_save_all(dataset, logger))
  return {"status": "Get Data Actual ..."}

# ============================================================================
# Model Training Routes
# ============================================================================

@model_router.post("/find-top")
async def find_top_models(
  dataset_name: str,
  background_tasks: BackgroundTasks,
  db: Session = Depends(get_session),
):
  """Find top performing models"""
  dataset = get_dataset_or_404(dataset_name, db)
  logger = Logger(dataset.name)
  if dataset.is_valid:
    logger.info(f"Find Top Model : {dataset.name} | n_top : {dataset.n_models}")
    core = dataset.task_type.core()
    background_tasks.add_task(core.find_top_model, dataset_name=dataset.name, n_top=dataset.n_models,logger=logger)
    return {
      "status": "processing",
      "message": "Finding top models in background",
      "dataset": dataset.name,
    }
  else: raise HTTPException(status_code=400,detail=f"Dataset is not valid : {dataset.status}")



@model_router.post("/train")
async def train_single_model(
  payload: ModelTrainSchema,
  background_tasks: BackgroundTasks,
  db: Session = Depends(get_session),
):
  """Train single model"""
  dataset = get_dataset_or_404(payload.dataset_name, db)

  background_tasks.add_task(
    train_model,
    dataset=dataset,
    algorithm=payload.algorithm,
    db=db,
    use_this=payload.use_this,
  )

  return {
    "status": "training",
    "message": f"Training {payload.algorithm} in background",
    "dataset": dataset.name,
  }


@model_router.post("/{dataset_name}/change-top/{algorithm}")
async def switch_top_model(
  dataset_name: str, algorithm: str, db: Session = Depends(get_session)
):
  """Change the top model for a dataset"""
  dataset = get_dataset_or_404(dataset_name, db)
  change_top_model(dataset, algorithm)
  db.commit()

  return {
    "status": "success",
    "message": f"Top model changed to {algorithm}",
    "dataset": dataset_name,
  }


# ============================================================================
# Model Query Routes
# ============================================================================


@model_router.get("", response_model=List[dict])
async def list_all_models(db: Session = Depends(get_session)):
  """List all trained models"""
  models = db.query(ModelML).all()
  return [m.to_response() for m in models]


@model_router.get("/{model_name}")
async def get_model(model_name: str, db: Session = Depends(get_session)):
  """Get model by name"""
  model = get_model_or_404(model_name, db)
  return model.to_response()


@model_router.get("/by-dataset/{dataset_name}")
async def get_models_by_dataset(dataset_name: str, db: Session = Depends(get_session)):
  """Get all models for a dataset"""
  dataset = get_dataset_or_404(dataset_name, db)
  return dataset.models


@model_router.get("/algorithms/{task_type}")
async def list_algorithms(task_type: str):
  """List available algorithms for task type"""
  try:
    tt = TaskType.from_str(task_type)
    return tt.algorithms()
  except:
    raise ValidationException(f"Invalid task type: {task_type}")


# ============================================================================
# Cleanup Routes
# ============================================================================


@model_router.delete("/by-dataset/{dataset_name}")
async def delete_models(dataset_name: str, db: Session = Depends(get_session)):
  """Delete all models for a dataset"""
  dataset = get_dataset_or_404(dataset_name, db)
  count = clean_models(dataset, db)

  return {
    "status": "success",
    "message": f"Cleaned {count} models",
    "dataset": dataset_name,
  }


@model_router.delete("/{dataset_name}/results")
async def clean_model_results(dataset_name: str, db: Session = Depends(get_session)):
  """Clean inference results for a dataset"""
  dataset = get_dataset_or_404(dataset_name, db)
  clean_results(dataset)

  return {"status": "success", "message": "Results cleaned", "dataset": dataset_name}


# ============================================================================
# Get Historical Predictions (Almost from InfluxDB)
# ============================================================================


@model_router.get("/{dataset_name}/history")
async def get_historical_predictions(
  dataset_name: str, n_day: int = 3, db: Session = Depends(get_session)
):
  try:
    dataset = get_dataset_or_404(dataset_name, db)
    interval = dataset.interval
    n_points = (24 * n_day * 60) // interval
    LOGGER_GLOBAL.info(f"Get Predictions History : {dataset.name} | {dataset.task_type}")
    if dataset.task_type.is_clustering():
      logger = Logger(dataset_name)
      data = dataset.task_type.core().get_clusters(dataset, logger)
      return data
    else:
      influx = get_influx_storage()
      data = influx.query_inference(dataset_name=dataset_name,task_type=str(dataset.task_type), start=f"-{24 * n_day}h", end=f"{24 * n_day}h")[:n_points]
      return data
  except Exception as e:
    LOGGER_GLOBAL.error(str(e))
    raise HTTPException(status_code=500, detail=str(e))

@model_router.get("/{dataset_name}/history/anomaly")
async def get_historical_anomaly(dataset_name, n_day: int = 3):
  with InfluxDBStorage() as influx:
    data = influx.query_inference(dataset_name=dataset_name, task_type="Anomaly")
    if not data: raise HTTPException(status_code=404, detail="Data is not found!")
    return data

@model_router.get("/history/anomaly")
async def get_all_hist_anomaly(n_day: int = 3):
  with InfluxDBStorage() as influx:
    data = influx.query_inference(start=f"-{24 * n_day}h", end=f"{24 * n_day}h", task_type="Anomaly")
    if not data: raise HTTPException(status_code=404, detail="Data is not found!")
    return data

# ============================================================================
# Stream Router
# ============================================================================

@stream_router.websocket("/realtime/anomaly")
async def realtime_get_anomaly(
  websocket: WebSocket,
  dataset_name: str,
  db: Session = Depends(get_session),
  interval: float = Query(default=3.0, ge=1.0, le=60.0, description="Update interval in seconds")
):
  """WebSocket for real-time anomaly detection with optimized resource usage.

  Features:
  - Shared prediction state untuk multiple clients (satu dataset)
  - Auto-retry dengan exponential backoff
  - Heartbeat/ping-pong untuk deteksi client zombie
  - Graceful error handling dan cleanup
  """
  await anomaly_manager.handle_connection(
    websocket=websocket,
    dataset_name=dataset_name,
    db=db,
    sleep_interval=interval
  )


@stream_router.websocket("/realtime/regression")
async def realtime_get_regression(
  websocket: WebSocket,
  dataset_name: str,
  db: Session = Depends(get_session),
  interval: float = Query(default=5.0, ge=1.0, le=60.0, description="Update interval in seconds")
):
  """WebSocket for real-time regression inference with optimized resource usage.

  Features:
  - Shared prediction state untuk multiple clients (satu dataset)
  - Auto-retry dengan exponential backoff
  - Heartbeat/ping-pong untuk deteksi client zombie
  - Graceful error handling dan cleanup
  """
  await regression_manager.handle_connection(
    websocket=websocket,
    dataset_name=dataset_name,
    db=db,
    sleep_interval=interval
  )


@stream_router.websocket("/realtime/clustering")
async def realtime_get_clustering(
  websocket: WebSocket,
  dataset_name: str,
  db: Session = Depends(get_session),
  interval: float = Query(default=5.0, ge=1.0, le=60.0, description="Update interval in seconds")
):
  """WebSocket for real-time clustering analysis with optimized resource usage.

  Features:
  - Shared prediction state untuk multiple clients (satu dataset)
  - Auto-retry dengan exponential backoff
  - Heartbeat/ping-pong untuk deteksi client zombie
  - Graceful error handling dan cleanup
  """
  await clustering_manager.handle_connection(
    websocket=websocket,
    dataset_name=dataset_name,
    db=db,
    sleep_interval=interval
  )


@stream_router.websocket("/realtime/forecast")
async def realtime_get_forecast(
  websocket: WebSocket,
  dataset_name: str,
  db: Session = Depends(get_session),
  interval: float = Query(default=10.0, ge=1.0, le=300.0, description="Update interval in seconds")
):
  """WebSocket for real-time time series forecasting with optimized resource usage.

  Features:
  - Shared prediction state untuk multiple clients (satu dataset)
  - Auto-retry dengan exponential backoff
  - Expiration check dan auto-update notification
  - Heartbeat/ping-pong untuk deteksi client zombie
  - Graceful error handling dan cleanup
  """
  await forecast_manager.handle_connection(
    websocket=websocket,
    dataset_name=dataset_name,
    db=db,
    sleep_interval=interval
  )


@stream_router.websocket("/realtime/logs")
async def realtime_websocket_logs(
  websocket: WebSocket,
  dataset_name: str
):
  """WebSocket for real-time log streaming dengan initial history.

  Features:
  - Kirim 200 baris log history saat pertama connect
  - Streaming log baru secara real-time menggunakan watchdog
  - Shared observer untuk multiple clients (satu dataset)
  - Heartbeat/ping-pong untuk deteksi client zombie
  - Graceful error handling dan cleanup
  """
  await logger_manager.handle_connection(
    websocket=websocket,
    dataset_name=dataset_name,
    require_dataset=False,
    max_lines=200
  )

# ============================================================================
# Worker Router
# ============================================================================

@worker_router.get("/tasks")
async def get_tasks():
  tasks = WorkerManager.get_tasks()
  return tasks

@worker_router.get("/task")
async def get_task(dataset_name: str):
  tasks = WorkerManager.get_tasks(dataset_name)
  return tasks

@worker_router.post("/start")
async def start(task_name: str):
  task = WorkerManager.get_tasks(task_name)
  if not task: raise NotFoundException("Task", task_name)
  if WorkerManager.is_active(task_name): return {"detail": f"Task {task_name} is still Active"}
  WorkerManager.start(task_name)
  return {"detail": f"{task_name} is start ..."}

@worker_router.post("/create")
async def create(
  dataset_name: str, is_anomaly: bool = False, db: Session = Depends(get_session)
):
  dataset = get_dataset_or_404(dataset_name, db)
  task_type = dataset.task_type
  if is_anomaly: task_type = TaskType.Anomaly
  task_name = f"{dataset_name}-{task_type}"
  if WorkerManager.is_active(task_name): return {"detail": f"{task_name} is still active"}
  WorkerManager.create(dataset_name, task_type)
  return {"detail": f"create task {task_name} ..."}

@worker_router.post("/stop")
async def stop(task_name: str):
  task = WorkerManager.get_tasks(task_name)
  if not task: raise NotFoundException("Task", task_name)
  WorkerManager.stop(task_name)
  return {"detail": f"{task_name} is stop ..."}

@worker_router.post("/stop-by-dataset")
async def stop_by_dataset_name(dataset_name: str, task_type: str):
  task_name = f"{dataset_name}-{task_type}"
  task = WorkerManager.get_tasks(task_name)
  if not task: raise NotFoundException("Task", task_name)
  WorkerManager.stop(task_name)
  return {"detail": f"{task_name} is stop ..."}

@worker_router.delete("/delete")
async def delete(task_name: str):
  task = WorkerManager.get_tasks(task_name)
  if not task: raise NotFoundException("Task", task_name)
  WorkerManager.delete(task_name)
  return {"detail": f"{task_name} is delete ..."}

@utils_router.get("/stats")
async def get_stats_dashboard():
  try:
    stats = get_stats()
    return stats
  except Exception: raise HTTPException(status_code=500, detail="Stats is not found!")

@utils_router.post("/clusters/unique")
async def naming_clusters(dataset_name: str, data_naming: dict, db: Session = Depends(get_session)):
  try:
    dataset = get_dataset_or_404(dataset_name, db)
    logger = Logger(dataset_name)
    core = dataset.task_type.core()
    logger.info("Save naming clusters ...")
    data_naming = data_naming["data_naming"]
    core.update_naming_clusters(dataset, data_naming)
    result = {algo: list(data_naming[algo].values()) for algo in data_naming.keys()}
    logger.info(f"Naming : {result}")
    return result
  except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@utils_router.get("/clusters/unique")
async def get_clusters_unique(dataset_name: str, db: Session = Depends(get_session)):
  try:
    dataset = get_dataset_or_404(dataset_name, db)
    logger = Logger(dataset_name)
    core = dataset.task_type.core()
    logger.info("get clusters unique...")
    data = core.get_naming_clusters(dataset.name)
    if not data: data = core.get_cluster_unique(dataset)
    else: data = {key: list(data[key].values()) for key in data}
    return data
  except Exception as e: raise HTTPException(status_code=500, detail=str(e))
  
@utils_router.get("/actual")
async def get_actual_point(tagname:str):
  data = get_realtime(tagname,logger=LOGGER_GLOBAL)
  return {"value":data}

@utils_router.get("/config")
async def get_config():
  return Config.export()

@utils_router.post("/config")
async def post_config(data:dict):
  Config.import_data(data)
  return data
