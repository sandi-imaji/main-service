"""Utility endpoints: stats, config, cluster naming, tag lookups (prefix: /utils)."""
from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.config import Config
from app.database.base import get_session
from app.database.historicalDB import InfluxDBStorage
from app.logger import Logger
from app.pull import query_tagnames
from app.routes.base import get_dataset_or_404
from app.services import clustering as clustering_service
from app.services.model import get_stats
from app.utils.realtime_cache import realtime_cache

router = APIRouter(tags=["utils"])

@router.get("/historyDB/{name}")
async def get_historyDB(name:str):
  with InfluxDBStorage() as influx: return influx.query_inference(dataset_name=name,task_type="Anomaly")

# --- Dashboard -------------------------------------------------------------

@router.get("/stats")
async def get_dashboard_stats():
  """Aggregate statistics for the dashboard."""
  return get_stats()


# --- Cluster naming --------------------------------------------------------

@router.post("/clusters/unique")
async def save_cluster_names(dataset_name: str, data_naming: dict, db: Session = Depends(get_session)):
  """Persist human-friendly names for a clustering dataset's clusters."""
  dataset = get_dataset_or_404(dataset_name, db)
  logger = Logger(dataset_name)
  naming = data_naming["data_naming"]
  clustering_service.update_naming_clusters(dataset, naming)
  result = {algo: list(naming[algo].values()) for algo in naming}
  logger.info(f"Naming: {result}")
  return result


@router.get("/clusters/unique")
async def get_cluster_names(dataset_name: str, db: Session = Depends(get_session)):
  """Get cluster names for a dataset, falling back to the raw unique clusters."""
  dataset = get_dataset_or_404(dataset_name, db)
  Logger(dataset_name).info("get clusters unique...")
  data = clustering_service.get_naming_clusters(dataset.name)
  if not data:
    return clustering_service.get_cluster_unique(dataset)
  return {key: list(data[key].values()) for key in data}


# --- Tag / realtime lookups ------------------------------------------------

@router.get("/tagname")
async def search_tagnames(query: str):
  """Search tag names from the external API."""
  return query_tagnames(query)


@router.get("/task-types")
async def list_task_types():
  """List supported task types."""
  return ["Regression", "Clustering", "TimeSeries", "Anomaly"]


@router.get("/actual")
async def get_realtime_value(tagname: str):
  """Get the latest realtime value for a tag (cached, non-blocking)."""
  return {"value": await realtime_cache.get(tagname)}


@router.get("/realtime/{tagname}")
async def get_realtime_by_tag(tagname: str):
  """Get the latest realtime value for a tag (path-param variant, cached)."""
  return await realtime_cache.get(tagname)


# --- Config ----------------------------------------------------------------
@router.get("/config")
async def get_config():
  """Get the service configuration, including InfluxDB retention."""
  data = Config.export()
  with InfluxDBStorage() as influx:
    data["retention"] = influx.get_retention()
  return data


@router.post("/config")
async def update_config(data: dict):
  """Update the service configuration and InfluxDB retention."""
  Config.import_data(data)
  retention = data["data"]["retention"]
  with InfluxDBStorage() as influx:
    influx.update_retention(int(retention))
  return data
