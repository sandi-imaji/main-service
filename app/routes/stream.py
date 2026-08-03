"""Realtime WebSocket endpoints (prefix: /stream).

Each connection is delegated to a shared manager that handles multi-client
prediction sharing, heartbeat, retry/backoff, and cleanup.
"""
from fastapi import APIRouter, Depends, Query, WebSocket
from sqlmodel import Session

from app.database.base import get_session
from app.services.streamer import (
  anomaly_manager,
  clustering_manager,
  forecast_manager,
  logger_manager,
  regression_manager,
)

router = APIRouter(tags=["stream"])


@router.websocket("/realtime/anomaly")
async def ws_anomaly(
  websocket: WebSocket,
  dataset_name: str,
  db: Session = Depends(get_session),
  interval: float = Query(default=3.0, ge=1.0, le=60.0, description="Update interval in seconds"),
):
  """Stream real-time anomaly detection."""
  await anomaly_manager.handle_connection(websocket=websocket, dataset_name=dataset_name, db=db, sleep_interval=interval)


@router.websocket("/realtime/regression")
async def ws_regression(
  websocket: WebSocket,
  dataset_name: str,
  db: Session = Depends(get_session),
  interval: float = Query(default=5.0, ge=1.0, le=60.0, description="Update interval in seconds"),
):
  """Stream real-time regression inference."""
  await regression_manager.handle_connection(websocket=websocket, dataset_name=dataset_name, db=db, sleep_interval=interval)


@router.websocket("/realtime/clustering")
async def ws_clustering(
  websocket: WebSocket,
  dataset_name: str,
  db: Session = Depends(get_session),
  interval: float = Query(default=5.0, ge=1.0, le=60.0, description="Update interval in seconds"),
):
  """Stream real-time clustering analysis."""
  await clustering_manager.handle_connection(websocket=websocket, dataset_name=dataset_name, db=db, sleep_interval=interval)


@router.websocket("/realtime/forecast")
async def ws_forecast(
  websocket: WebSocket,
  dataset_name: str,
  db: Session = Depends(get_session),
  interval: float = Query(default=10.0, ge=1.0, le=300.0, description="Update interval in seconds"),
):
  """Stream real-time time-series forecasting."""
  await forecast_manager.handle_connection(websocket=websocket, dataset_name=dataset_name, db=db, sleep_interval=interval)


@router.websocket("/realtime/logs")
async def ws_logs(
  websocket: WebSocket,
  dataset_name: str,
  channel: str = Query(default="main", pattern="^(main|worker)$",
                       description="main = train/pulling/inference, worker = the pm2 process"),
):
  """Stream real-time logs, starting with the last 200 lines of history.

  `dataset_name="system"` streams global.log (API surface + worker lifecycle).
  """
  await logger_manager.handle_connection(websocket=websocket, dataset_name=dataset_name,
                                         require_dataset=False, max_lines=200, channel=channel)
