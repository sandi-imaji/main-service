"""
Time Series Forecasting core — pure ML compute.

No Dataset / DB / InfluxDB / HTTP knowledge: everything is expressed through the
contracts in `app.core.contracts`. The Dataset/Influx-aware orchestration
(timestamps, actual back-fill, refit, worker loop) lives in
`app.services.timeseries`.
"""
import os
from typing import List

from pycaret import time_series as mod

from app.config import Config
from app.core.base import BaseMLCore
from app.core.contracts import (
    ForecastRequest,
    ForecastResult,
    TimeSeriesTrainRequest,
    TrainedModel,
    fold_spread,
)
from app.utils.model_cache import get_timeseries_cache


class TimeSeries(BaseMLCore):
  """Time-series forecasting with shared model caching."""

  TOP_ALGO: List[str] = [
      "catboost_cds_dt",
      "xgboost_cds_dt",
      "lightgbm_cds_dt",
      "gbr_cds_dt",
      "huber_cds_dt",
      "ets",
  ]
  _task_type = "timeseries"

  @classmethod
  def get_cache(cls):
    """Get the model cache for time series forecasting."""
    return get_timeseries_cache()

  @staticmethod
  def forecast(req: ForecastRequest, logger) -> ForecastResult:
    """
    Project each saved model `req.fh` steps ahead. Pure compute: no Dataset, no
    timestamps (the service attaches those). Model loading uses the shared cache.
    Returns {algorithm: [values]}; raises on failure (the caller decides).
    """
    logger.info("Forecast models :")
    result = {}
    for algorithm, path in req.models:
      model = TimeSeries._load_model_cached(mod, path, logger)
      res = mod.predict_model(model, req.fh, verbose=Config.verbose.pycaret)["y_pred"].values.tolist()
      result[algorithm] = res
      logger.info(f"Forecast {algorithm} success!")
    return ForecastResult(forecast=result)

  @staticmethod
  def compare_models(req: TimeSeriesTrainRequest, logger) -> List[TrainedModel]:
    """
    Train & rank the top-N forecasting models (PyCaret compare_models over
    TOP_ALGO). Pure compute (no Dataset/DB): saves artifacts into `req.out_dir`
    and returns their descriptors, best first.

    Note: time-series setup does not apply `req.preprocessing` — kept as the
    current behaviour.
    """
    features = [c for c in req.df.columns if c != "dt"]
    mod.setup(req.df[features], fh=req.fh, verbose=Config.verbose.pycaret, ignore_features=["dt"])

    top_models = mod.compare_models(
        n_select=req.n_top, verbose=Config.verbose.pycaret, include=TimeSeries.TOP_ALGO)
    if not isinstance(top_models, list): top_models = [top_models]
    metrics = mod.pull().to_dict("index")
    keys = list(metrics.keys())

    os.makedirs(req.out_dir, exist_ok=True)
    out: List[TrainedModel] = []
    for i in range(min(req.n_top, len(top_models))):
      algorithm = keys[i]
      evaluation = metrics[algorithm]
      evaluation.pop("Model", None)

      path = req.out_dir / algorithm
      mod.save_model(mod.finalize_model(top_models[i]), str(path))
      logger.info(f"Save Model at {path}")

      out.append(TrainedModel.from_saved(algorithm, path, evaluation))

    get_timeseries_cache().clear()
    return out

  @staticmethod
  def retrain_models(req:TimeSeriesTrainRequest,algorithms:List[str],logger):
    """
    Retrain a specific set of algorithms on `req.df` in a single setup.

    Pure compute (no Dataset/DB): saves artifacts into `req.out_dir` and returns
    their descriptors. Used by the finetune service, which owns pulling the new
    data, persistence, and dataset bookkeeping.
    """
    features = [c for c in req.df.columns if c != "dt"]
    mod.setup(req.df[features],fh=req.fh,verbose=Config.verbose.pycaret)

    os.makedirs(req.out_dir, exist_ok=True)
    results: List[TrainedModel] = []
    for algorithm in algorithms:
      model = mod.create_model(algorithm, verbose=Config.verbose.pycaret)
      pulled = mod.pull()
      metric = pulled.loc["Mean"].to_dict()
      metric.update(fold_spread(pulled))          # simpangan antar-fold = sinyal kestabilan

      path = req.out_dir / algorithm
      mod.save_model(mod.finalize_model(model), str(path))
      logger.info(f"Save Model at {path}")

      results.append(TrainedModel.from_saved(algorithm, path, metric))

    get_timeseries_cache().clear()
    return results

  @staticmethod
  def train_one(req: TimeSeriesTrainRequest, algorithm: str, logger) -> TrainedModel:
    """Train a single forecasting algorithm. Pure compute — returns its descriptor."""
    features = [c for c in req.df.columns if c != "dt"]
    mod.setup(req.df[features], fh=req.fh, verbose=Config.verbose.pycaret, ignore_features=["dt"])

    model = mod.create_model(algorithm, verbose=Config.verbose.pycaret)
    pulled = mod.pull()
    metric = pulled.loc["Mean"].to_dict()
    metric.update(fold_spread(pulled))            # simpangan antar-fold = sinyal kestabilan

    os.makedirs(req.out_dir, exist_ok=True)
    path = req.out_dir / algorithm
    mod.save_model(mod.finalize_model(model), str(path))
    logger.info(f"Save Model at {path}")

    get_timeseries_cache().clear()
    return TrainedModel.from_saved(algorithm, path, metric)
