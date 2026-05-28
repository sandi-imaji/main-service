"""
InfluxDB Client untuk menyimpan hasil inference ML (unified untuk semua task type).
Menggunakan format pivot table: 1 point per model prediction.

Schema:
- measurement: "inference_results"
- tags: dataset_name, task_type, model_name
- fields: value, actual, features
"""

import atexit
import json
import math
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
from influxdb_client.client.influxdb_client import InfluxDBClient
from influxdb_client.client.write.point import Point
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client.domain.write_precision import WritePrecision

from app.config import Config
from app.logger import Logger
from app.helpers import DTEncoder


def _clean_value(val: Any) -> Any:
    """
    Clean value untuk JSON serialization:
    - Convert numpy types ke Python native types
    - Convert NaN, Infinity, -Infinity ke None
    """
    if val is None:
        return None

    # Handle numpy types
    if hasattr(val, 'item'):
        val = val.item()

    # Handle float special values
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return val

    return val


class InfluxDBStorage:
  """
  Unified InfluxDB storage untuk semua ML task types.

  Task Types:
  - supervised: value=prediction, actual=ground_truth
  - unsupervised: value=cluster_id, actual=None
  - timeseries: value=forecast_value, actual=None
  - anomaly: value=anomaly_score, actual=None
  """

  def __init__(
      self,
      url: str = "http://localhost:8086",
      token: Optional[str] = None,
      org: str = "tech",
  ):
    self.logger = Logger("influx_storage")
    self.bucket = getattr(Config, "influxdb_bucket", "ml-buckets")
    self.org = org or getattr(Config, "influxdb_org", "tech")

    # Get token from param, config, or raise error
    token = token or getattr(Config, "influxdb_token", None)
    self.logger.info(f"BUCKETS : {self.bucket}  | ORG : {self.org}")
    if not token:
      raise ValueError(
          "InfluxDB token required. Set INFLUXDB_TOKEN env var or Config.influxdb_token"
      )

    try:
      self.client = InfluxDBClient(url=url, token=token, org=self.org)
      self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
      self.query_api = self.client.query_api()
      self.logger.info(f"InfluxDB connected: {url}")
    except Exception as e:
      self.logger.error(f"Failed to connect InfluxDB: {e}")
      raise

  def _create_point(
      self,
      dataset_name: str,
      task_type: str,
      model_name: str,
      value: Union[float, int],
      timestamp: Optional[Union[datetime, str]] = None,
      actual: Optional[Union[float, int]] = None,
      features: Optional[Dict] = None,
  ) -> Point:
    """Create InfluxDB Point untuk single model prediction."""
    # Parse timestamp
    if timestamp is None:
      timestamp = DTEncoder.now()
    elif isinstance(timestamp, str):
      timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
      timestamp = timestamp.replace(tzinfo=timezone.utc)

    point = (
        Point("inference_results")
        .tag("dataset_name", dataset_name)
        .tag("task_type", task_type)
        .tag("model_name", model_name)
        .field("value", float(value))
        .time(timestamp, WritePrecision.NS)
    )

    # Optional fields
    if actual is not None:
      point = point.field("actual", float(actual))
    if features:
      point = point.field("features", json.dumps(features))

    return point

  def write_inference(
      self,
      dataset_name: str,
      task_type: str,
      results: Dict[str, Union[float, int]],
      timestamp: Optional[Union[datetime, str]] = None,
      actual: Optional[Union[float, int]] = None,
      features: Optional[Dict] = None,
  ) -> bool:
    """
    Write inference results untuk semua model dalam satu timestamp.

    Args:
        dataset_name: Nama dataset
        task_type: supervised | unsupervised | timeseries | anomaly
        timestamp: Waktu inference
        results: Dict {model_name: value} - e.g., {"xgboost": 43.2, "knn": 41.8}
        actual: Ground truth value (untuk supervised)
        features: Feature values sebagai dict

    Returns:
        True jika sukses

    Example:
        # Supervised
        write_inference(
            dataset_name="Reg-123",
            task_type="supervised",
            timestamp="2026-03-16T10:00:00",
            results={"xgboost": 43.2, "knn": 41.8},
            actual=42.5,
            features={"x1": 1.0, "x2": 2.0}
        )

        # Unsupervised
        write_inference(
            dataset_name="Cluster-xyz",
            task_type="unsupervised",
            timestamp="2026-03-16T10:00:00",
            results={"kmeans": 1, "birch": 0},
            features={"f1": 25.4}
        )
    """
    try:
      points = []
      for model_name, value in results.items():
        point = self._create_point(
            dataset_name=dataset_name,
            task_type=task_type,
            model_name=model_name,
            timestamp=timestamp,
            value=value,
            actual=actual,
            features=features,
        )
        points.append(point)

      self.write_api.write(bucket=self.bucket, org=self.org, record=points)
      self.logger.debug(
          f"Written {len(points)} points for {dataset_name} ({task_type})"
      )
      return True

    except Exception as e:
      self.logger.error(f"Failed to write inference: {e}")
      return False

  def query_inference(
      self,
      dataset_name: Optional[str] = None,
      task_type: Optional[str] = None,
      model_name: Optional[str] = None,
      start: Optional[str] = None,
      end: Optional[str] = None,
      limit: int = 1000,
  ) -> List[Dict[str, Any]]:
    """
    Query inference results.

    Args:
        dataset_name: Filter by dataset
        task_type: Filter by task type
        model_name: Filter by model (e.g., "xgboost")
        start: Start time (default: all data, e.g., "-1h", "-1d", "2026-03-16T00:00:00Z")
        end: End time (default: all data, e.g., "1h", "2026-03-16T23:59:59Z")
        limit: Max results

    Returns:
        List of results

    Example:
        # Query all data (no time filter)
        query_inference(dataset_name="Reg-123")

        # Query all xgboost predictions for last hour
        query_inference(model_name="xgboost", start="-1h")

        # Query specific dataset
        query_inference(dataset_name="Reg-123", start="-24h")
    """
    # Build filters
    filters = ['r._measurement == "inference_results"']

    if dataset_name:
      filters.append(f'r.dataset_name == "{dataset_name}"')
    if task_type:
      filters.append(f'r.task_type == "{task_type}"')
    if model_name:
      filters.append(f'r.model_name == "{model_name}"')

    filter_str = " and ".join(filters)

    # Build time range - if no start/end specified, query all data
    if start is None and end is None:
      # Query all data from beginning of time to now
      range_str = "range(start: 0)"
    elif start is None:
      # Only end specified - query from beginning to end
      range_str = f"range(start: 0, stop: {end})"
    elif end is None:
      # Only start specified
      range_str = f"range(start: {start})"
    else:
      # Both specified
      range_str = f"range(start: {start}, stop: {end})"

    # Query 1: Get model predictions pivoted by model_name
    predictions_query = f'''from(bucket: "{self.bucket}")
            |> {range_str}
            |> filter(fn: (r) => {filter_str})
            |> filter(fn: (r) => r._field == "value")
            |> pivot(rowKey:["_time", "dataset_name", "task_type"], columnKey: ["model_name"], valueColumn: "_value")
            |> group()
            |> sort(columns: ["_time"], desc: true)
            |> limit(n: {limit})'''

    # Query 2: Get features_json
    features_query = f'''from(bucket: "{self.bucket}")
            |> {range_str}
            |> filter(fn: (r) => {filter_str})
            |> filter(fn: (r) => r._field == "features")
            |> group()
            |> sort(columns: ["_time"], desc: true)
            |> limit(n: {limit})'''

    # Query 3: Get actual values
    actual_query = f'''from(bucket: "{self.bucket}")
            |> {range_str}
            |> filter(fn: (r) => {filter_str})
            |> filter(fn: (r) => r._field == "actual")
            |> group()
            |> sort(columns: ["_time"], desc: true)
            |> limit(n: {limit})'''

    try:
      # Execute predictions query
      predictions_result = self.query_api.query(
          query=predictions_query, org=self.org
      )

      # Execute features query
      features_result = self.query_api.query(
          query=features_query, org=self.org)

      # Execute actual query
      actual_result = self.query_api.query(query=actual_query, org=self.org)

      # Build features lookup by time (UTC)
      features_by_time = {}
      for table in features_result:
        for record in table.records:
          time_key = record.get_time().isoformat()
          features_json = record.values.get("_value")
          if features_json:
            try:
              features_by_time[time_key] = json.loads(features_json)
            except:
              features_by_time[time_key] = features_json

      # Build actual lookup by time (UTC)
      actual_by_time = {}
      for table in actual_result:
        for record in table.records:
          time_key = record.get_time().isoformat()
          actual_value = record.values.get("_value")
          if actual_value is not None:
            actual_by_time[time_key] = _clean_value(actual_value)

      # Build results in new format
      results = []
      for table in predictions_result:
        for record in table.records:
          time_key = record.get_time().isoformat()

          row = {
              "timestamp": DTEncoder.from_utc(time_key).isoformat(),
              "dataset_name": record.values.get("dataset_name"),
              "task_type": record.values.get("task_type"),
          }

          # Add actual if available
          if time_key in actual_by_time:
            row["actual"] = actual_by_time[time_key]

          # Add model predictions to results dict
          predictions = {}
          for key, value in record.values.items():
            if key not in [
                "_time",
                "_start",
                "_stop",
                "_measurement",
                "dataset_name",
                "task_type",
                "result",
                "table",
                "_field",
            ]:
              if value is not None:
                predictions[key] = _clean_value(value)

          if predictions:
            row["results"] = predictions

          # Add features if available
          if time_key in features_by_time:
            row["features"] = features_by_time[time_key]

          results.append(row)

      return results

    except Exception as e:
      self.logger.error(f"Query failed: {e}")
      return []

  def get_dataframe(self, dataset_name: str, start: str = "-30d") -> pd.DataFrame:
    """Mengambil data inference dengan pivot model + features"""
    # Query Predictions (satu query dengan pivot yang benar)
    query = f'''
        from(bucket: "{self.bucket}")
            |> range(start: 0)
            |> filter(fn: (r) => r._measurement == "inference_results")
            |> filter(fn: (r) => r.dataset_name == "{dataset_name}")
            |> filter(fn: (r) => r._field == "value" or r._field == "actual" or r._field == "features")
            |> pivot(
                rowKey: ["_time", "dataset_name", "task_type"],
                columnKey: ["_field", "model_name"],
                valueColumn: "_value"
            )
            |> sort(columns: ["_time"])
        '''

    try:
      result = self.query_api.query_data_frame(query, org=self.org)

      # Handle list of DataFrames (multiple tables) or single DataFrame
      if isinstance(result, list):
        if len(result) == 0:
          return pd.DataFrame()
        df = pd.concat(result, ignore_index=True)
      else:
        df = result

      if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
      if df.empty:
        return pd.DataFrame()

      # Flatten column names (value_model_name -> model_name, actual_model_name -> actual, dll)
      columns = []
      for col in df.columns:
        if isinstance(col, tuple):
          field, model = col
          if field == "value":
            columns.append(model)
          elif field == "actual":
            columns.append("actual")
          elif field == "features":
            columns.append("features")
          else:
            columns.append(f"{field}_{model}")
        else:
          columns.append(col)

      df.columns = columns

      # Drop duplicate columns (keep first actual)
      if hasattr(df.columns, "duplicated") and df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

      # Parse features jika ada
      if "features" in df.columns:

        def parse_features(x):
          if pd.isna(x) or x is None:
            return {}
          try:
            return json.loads(x) if isinstance(x, str) else x
          except:
            return {}

        df["features"] = df["features"].apply(parse_features)
        # Expand features menjadi kolom terpisah
        features_data = df["features"].tolist()
        features_df = pd.json_normalize(features_data)
        if not features_df.empty:
          features_df.columns = [
              f"feature_{col}" for col in features_df.columns
          ]
        df = pd.concat([df.drop(columns=["features"]), features_df], axis=1)

      # Hapus kolom metadata yang tidak diperlukan
      drop_cols = [
          col for col in df.columns if col.startswith(("result", "table"))
      ]
      if drop_cols:
        df = df.drop(columns=drop_cols)

      # Rename _time ke timestamp
      if "_time" in df.columns:
        df = df.rename(columns={"_time": "timestamp"})

      return df

    except Exception as e:
      self.logger.error(f"Failed to get dataframe: {e}")
      return pd.DataFrame()

  def get_models(
      self,
      dataset_name: Optional[str] = None,
      task_type: Optional[str] = None,
  ) -> List[str]:
    """
    Get list of unique model names.

    Args:
        dataset_name: Filter by dataset (optional)
        task_type: Filter by task type (optional)

    Returns:
        List of model names (e.g., ["xgboost", "knn", "lightgbm"])
    """
    filters = ['r._measurement == "inference_results"']

    if dataset_name:
      filters.append(f'r.dataset_name == "{dataset_name}"')
    if task_type:
      filters.append(f'r.task_type == "{task_type}"')

    filter_str = " and ".join(filters)

    query = f'''from(bucket: "{self.bucket}")
            |> range(start: -30d)
            |> filter(fn: (r) => {filter_str})
            |> keep(columns: ["model_name"])
            |> group()
            |> distinct(column: "model_name")'''

    try:
      result = self.query_api.query(query=query, org=self.org)
      models = []
      for table in result:
        for record in table.records:
          model = record.get_value()
          if model:
            models.append(model)
      return sorted(list(set(models)))
    except Exception as e:
      self.logger.error(f"Failed to get models: {e}")
      return []

  def delete_dataset(
      self,
      dataset_name: str,
      start: Optional[str] = None,
      end: Optional[str] = None,
  ) -> bool:
    """
    Delete all data untuk specific dataset.

    Args:
        dataset_name: Dataset name to delete
        start: Start time (default: 1970-01-01)
        end: End time (default: now)

    Returns:
        True if successful
    """
    try:
      now = datetime.now(timezone.utc)

      if start and start.startswith("-"):
        # "-1h" → 1 hour ago
        hours = int(start.replace("-", "").replace("h", ""))
        start_time = now - timedelta(hours=hours)
        start_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
      else:
        start_str = start or "1970-01-01T00:00:00Z"

      if end and not end.startswith("-"):
        # "12h" → 12 hours from now
        hours = int(end.replace("h", ""))
        end_time = now + timedelta(hours=hours)
        end_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
      else:
        end_str = end or "2099-12-31T23:59:59Z"

      predicate = f'dataset_name="{dataset_name}"'

      self.client.delete_api().delete(
          start=start_str,
          stop=end_str,
          predicate=predicate,
          bucket=self.bucket,
          org=self.org,
      )

      self.logger.info(f"Deleted all data for dataset: {dataset_name}")
      return True

    except Exception as e:
      self.logger.error(f"Failed to delete dataset {dataset_name}: {e}")
      return False

  def ping(self) -> bool:
    """Check if InfluxDB is reachable."""
    try:
      return self.client.ping()
    except Exception:
      return False

  def close(self) -> None:
    """Close InfluxDB connection."""
    try:
      self.write_api.close()
      self.client.close()
      self.logger.info("InfluxDB connection closed")
    except Exception as e:
      self.logger.error(f"Error closing connection: {e}")

  def __enter__(self):
    """Context manager entry."""
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    """Context manager exit - ensures connection is closed."""
    self.close()
    return False  # Don't suppress exceptions


# Global instance
_influx_storage: Optional[InfluxDBStorage] = None


def get_influx_storage() -> InfluxDBStorage:
  """Get singleton InfluxDB storage instance."""
  global _influx_storage
  if _influx_storage is None:
    _influx_storage = InfluxDBStorage(
        url=getattr(Config, "influxdb_url", "http://localhost:8086"),
        token=getattr(Config, "influxdb_token", None),
        org=getattr(Config, "influxdb_org", "tech"),
    )
  return _influx_storage


def close_influx_storage() -> None:
  """Close singleton InfluxDB storage."""
  global _influx_storage
  if _influx_storage is not None:
    _influx_storage.close()
    _influx_storage = None


# Register atexit handler to ensure proper cleanup

atexit.register(close_influx_storage)


# Convenience functions for direct use
def write_inference(
    dataset_name: str,
    task_type: str,
    timestamp: Union[datetime, str],
    results: Dict[str, Union[float, int]],
    actual: Optional[Union[float, int]] = None,
    features: Optional[Dict] = None,
) -> bool:
  """Write inference using singleton instance."""
  storage = get_influx_storage()
  return storage.write_inference(
      dataset_name=dataset_name,
      task_type=task_type,
      timestamp=timestamp,
      results=results,
      actual=actual,
      features=features,
  )


def query_inference(**kwargs) -> List[Dict[str, Any]]:
  """Query inference using singleton instance."""
  storage = get_influx_storage()
  return storage.query_inference(**kwargs)


if __name__ == "__main__":
  import os
  import pprint
  from zoneinfo import ZoneInfo

  dataset_name = "Regression-dd0fbfe4"
  print(Config.influxdb_token)

  # Gunakan context manager untuk auto-close
    # data = writer.write_inference(dataset_name="Reg-123",
    #         timestamp = DTEncoder.to_utc(DTEncoder.now()),
    #         task_type="supervised",
    #         results={"xgboost": 43.2, "knn": 41.8},
    #         actual=42.5,
    #         features={"x1": 1.0, "x2": 2.0})
    # data = writer.query_inference(dataset_name=dataset_name,task_type="TimeSeries",end="24h")
    # pprint.pprint(data)
    # print(len(data))
    # writer.delete_dataset(dataset_name)
