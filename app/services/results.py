"""
Inference result DTOs (output layer).

The service layer builds these from the core's pure `PredictResult`, then either
serialises them to HTTP or writes them to InfluxDB. Kept out of `app.core` so the
core stays free of storage/output concerns.
"""
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel

from app.core.contracts import PredictResult, AnomalyResult, ForecastResult
from app.database.historicalDB import InfluxDBStorage
from app.helpers import DTEncoder


class SupervisedResultSchema(BaseModel):
  features: Dict[str, float | int]
  predictions: Dict[str, float | int]
  timestamp: datetime
  actual: Optional[int | float] = None
  task_type: str = "Regression"
  dataset_name: str
  is_valid: bool = False

  def write_to_influx(self, influx: InfluxDBStorage, logger):
    timestamp = DTEncoder.to_utc(self.timestamp)
    write_result = influx.write_inference(
      dataset_name=self.dataset_name,
      task_type=self.task_type,
      timestamp=timestamp,
      results=self.predictions,
      features=self.features,
      actual=self.actual,
    )
    logger.info(f"[Efficient] Write result: {write_result}")

  @classmethod
  def from_result(cls, result: PredictResult, dataset_name: str, actual=None) -> "SupervisedResultSchema":
    """Wrap a core PredictResult into a serialisable/persistable result."""
    return cls(features=result.features, predictions=result.predictions,
               timestamp=DTEncoder.now(), dataset_name=dataset_name,
               actual=actual, is_valid=True)

  @classmethod
  def failed(cls, features: dict, algorithms: List[str], dataset_name: str) -> "SupervisedResultSchema":
    """Build a non-valid result (used by the worker loop so it never crashes)."""
    flat = {k: (v[0] if isinstance(v, (list, tuple)) and v else v) for k, v in features.items()}
    return cls(features=flat, predictions={a: 0.0 for a in algorithms},
               timestamp=DTEncoder.now(), dataset_name=dataset_name)


class AnomalyResultSchema(BaseModel):
  timestamp: datetime
  features: Dict[str, int | float]
  is_anomaly: bool
  anomaly_score: int | float
  dataset_name: str
  task_type: str = "Anomaly"
  is_valid: bool = False

  def write_to_influx(self, influx: InfluxDBStorage, logger) -> None:
    write_result = influx.write_inference(
        dataset_name=self.dataset_name,
        task_type=self.task_type,
        timestamp=DTEncoder.to_utc(self.timestamp),
        results={"is_anomaly": self.is_anomaly, "anomaly_score": self.anomaly_score},
        features=self.features,
    )
    logger.info(f"[Efficient] Write result: {write_result}")

  @classmethod
  def from_result(cls, result: AnomalyResult, dataset_name: str) -> "AnomalyResultSchema":
    """Wrap a core AnomalyResult into a serialisable/persistable result."""
    return cls(timestamp=DTEncoder.now(), features=result.features,
               is_anomaly=result.is_anomaly, anomaly_score=result.anomaly_score,
               dataset_name=dataset_name, is_valid=True)

  @classmethod
  def invalid(cls, dataset_name: str) -> "AnomalyResultSchema":
    """Build a non-valid result (used by the worker loop so it never crashes)."""
    return cls(timestamp=DTEncoder.now(), features={"": 0.0}, is_anomaly=False, anomaly_score=0.0, dataset_name=dataset_name, is_valid=False)


class TimeSeriesResultSchema(BaseModel):
  timestamps: List[datetime]
  forecast: Dict[str, List[int | float]]
  dataset_name: str
  task_type: str = "TimeSeries"
  is_valid: bool = False

  @property
  def timestamps_(self) -> List[str]:
    return [str(i) for i in self.timestamps]

  def write_to_influx(self, influx: InfluxDBStorage, logger) -> None:
    model_names = list(self.forecast.keys())
    for idx, timestamp in enumerate(self.timestamps):
      results = {col: self.forecast[col][idx] for col in model_names}
      influx.write_inference(
          dataset_name=self.dataset_name,
          task_type=self.task_type,
          timestamp=DTEncoder.to_utc(timestamp),
          results=results,
          features={},
      )
      logger.info(f"[Efficient] Write result: {timestamp}")
    logger.info(f"Total Saved : {len(self.timestamps)}")

  @classmethod
  def from_forecast(cls, result: ForecastResult, timestamps: List[datetime],
                    dataset_name: str) -> "TimeSeriesResultSchema":
    """Wrap a core ForecastResult (+ the service-derived timestamps)."""
    return cls(timestamps=timestamps, forecast=result.forecast,dataset_name=dataset_name, is_valid=True)

  @classmethod
  def invalid(cls, dataset_name: str, timestamps: Optional[List[datetime]] = None) -> "TimeSeriesResultSchema":
    """Build a non-valid result (used by callers so they never crash)."""
    return cls(timestamps=timestamps or [], forecast={},dataset_name=dataset_name, is_valid=False)


class UnsupervisedResultSchema(BaseModel):
  features: Dict[str, int | float]
  features_reduce: Optional[Dict[str, int | float]] = None
  timestamp: datetime
  dataset_name: str
  clusters: Dict
  # Konteks per algoritma: {"distance", "radius", "ratio", "unusual", "share",
  # "since", "duration_minutes", "duration_exact"}. Label saja tidak memberi
  # tahu apakah kondisi sekarang wajar — rasio terhadap radius cluster dan
  # pangsa historisnyalah yang menjawab itu.
  context: Optional[Dict[str, Dict]] = None
  task_type: str = "Clustering"
  is_valid: bool = False
