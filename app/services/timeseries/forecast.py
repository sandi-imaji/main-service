"""Read path forecast: proyeksi ke depan + siklus auto-inference.

Komputasi ML-nya murni di `app.core.time_series`; di sini yang diurus hanya
menurunkan timestamp masa depan dan merangkai urutan back-fill → refit →
forecast saat hasil lama sudah kedaluwarsa.
"""
from app.core.time_series import TimeSeries
from app.database.DB import Dataset
from app.helpers import DTEncoder
from app.services.results import TimeSeriesResultSchema
from app.services.timeseries.actuals import get_actual_save
from app.services.timeseries.horizon import _current_dt, is_expired
from app.services.timeseries.retrain import retrain


def forecast(dataset: Dataset, logger) -> TimeSeriesResultSchema:
  """Derive the future timestamps and drive the pure core forecast, wrapping the
  result. This is the read path used by the HTTP endpoint and the streamer."""
  fh = int(dataset.target)
  timestamps = DTEncoder.generate_dt(n=fh, interval_minutes=dataset.interval, last_dt=_current_dt(dataset))
  try:
    result = TimeSeries.forecast(dataset.to_forecast_request(), logger)
    logger.info("Forecast was successful!")
    return TimeSeriesResultSchema.from_forecast(result, timestamps, dataset.name)
  except Exception as e:
    logger.error(f"Forecast failed: {e}")
    return TimeSeriesResultSchema.invalid(dataset.name, timestamps)


def auto_inference(dataset: Dataset, logger) -> TimeSeriesResultSchema:
  """When forecasts have expired: back-fill actuals, refit, then re-forecast."""
  if is_expired(dataset, logger):
    get_actual_save(dataset, logger)
    retrain(dataset, logger)
    return forecast(dataset, logger)
  return TimeSeriesResultSchema.invalid(dataset.name)
