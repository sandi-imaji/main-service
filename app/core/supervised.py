"""
Supervised Learning Module (Classification & Regression)
Optimized for performance with model caching and batch inference support.
Phase 3: Async support dengan unified cache
"""

from typing import List
import os,pandas as pd

from app.database.schemas import TaskType
from app.utils.model_cache import get_supervised_cache
from app.config import Config
from app.core.base import BaseMLCore
from app.core.contracts import (SupervisedTrainRequest, TrainedModel, PredictRequest,
                                PredictResult, fold_spread)


# =============================================================================
# Model Cache - Now menggunakan unified cache dari app.utils.model_cache
# =============================================================================


def get_model_cache():
  """Get the global model cache instance (backward compatibility)."""
  return get_supervised_cache()


# =============================================================================
# Module Loading
# =============================================================================


def module(task_type: TaskType):
  """
  Get the appropriate PyCaret module based on task type.
  Uses lazy loading for better startup performance.
  """
  if task_type.is_classification():
    from pycaret import classification

    return classification
  elif task_type.is_regression():
    from pycaret import regression

    return regression
  else:
    raise ValueError(f"Task Type : {task_type} in Supervised")


def get_config(mod, logger):
  """Log feature names and target from PyCaret config."""
  feature_names = mod.get_config("X").columns.tolist()
  target_name = mod.get_config("target_param")
  logger.info(f"Features : {feature_names}")
  logger.info(f"Target : {target_name}")


# =============================================================================
# Supervised Learning Class
# =============================================================================


class Supervised(BaseMLCore):
  """
  Supervised learning operations with optimizations:
  - Model caching for faster repeated inference
  - Batch inference support
  - Thread-safe operations
  """

  _task_type = "supervised"

  @classmethod
  def get_cache(cls):
    """Get the model cache for supervised learning."""
    return get_supervised_cache()

  @staticmethod
  def predict(req: PredictRequest, logger) -> PredictResult:
    """
    Run inference for the given features across a set of saved models.

    Pure compute: no Dataset, no DB. Model loading uses the shared cache.
    Returns predictions {algorithm: value}; raises on failure (the caller
    decides how to handle it).

    Example predictions: {'knn': 20.19, 'lr': 20.20}
    """
    mod = module(TaskType.from_str(req.task))
    features_df = pd.DataFrame.from_dict(req.features)

    predictions = {}
    logger.info("Inference models: ")
    for algorithm, path in req.models:
      logger.info(f"Inference with {algorithm}")
      model = Supervised._load_model_cached(mod, path, logger)
      res = mod.predict_model(model, features_df)
      predictions[algorithm] = res.to_dict(orient="list")["prediction_label"][0]

    logger.info("Inference finished!")
    return PredictResult(features=features_df.iloc[0].to_dict(), predictions=predictions)

  @staticmethod
  def _setup(req: SupervisedTrainRequest, logger, **setup_kwargs):
    """Configure a PyCaret experiment from a TrainRequest (pure — no Dataset)."""
    mod = module(TaskType.from_str(req.task))
    mod.setup(
        req.df,
        target=req.target,
        verbose=Config.verbose.pycaret,
        ignore_features=["dt"],
        **req.preprocessing,
        **setup_kwargs,
    )
    get_config(mod, logger)
    return mod

  @staticmethod
  def compare_models(req: SupervisedTrainRequest, logger) -> List[TrainedModel]:
    """
    Train the top-N models with PyCaret compare_models. Pure compute:
    saves artifacts into `req.out_dir` and returns their descriptors. It does
    NOT touch the database, ModelML, or dataset status.
    """
    mod = Supervised._setup(req, logger, n_jobs=1)

    top_models = mod.compare_models(n_select=req.n_top, verbose=Config.verbose.pycaret)
    if not isinstance(top_models, list): top_models = [top_models]
    metrics = mod.pull().to_dict("index")
    keys = list(metrics.keys())

    os.makedirs(req.out_dir, exist_ok=True)
    results: List[TrainedModel] = []
    for i in range(min(req.n_top, len(top_models))):
      algorithm = keys[i]
      evaluation = metrics[algorithm]
      evaluation.pop("Model", None)

      path = req.out_dir / algorithm
      mod.save_model(mod.finalize_model(top_models[i]), str(path))
      logger.info(f"Save Model at {path}")

      results.append(TrainedModel.from_saved(algorithm, path, evaluation))

    get_model_cache().clear()
    return results

  @staticmethod
  def train_one(req: SupervisedTrainRequest, algorithm: str, logger) -> TrainedModel:
    """Train a single algorithm. Pure compute — returns its descriptor."""
    mod = Supervised._setup(req, logger)

    model = mod.create_model(algorithm, verbose=Config.verbose.pycaret)
    pulled = mod.pull()
    metric = pulled.loc["Mean"].to_dict()
    metric.update(fold_spread(pulled))            # simpangan antar-fold = sinyal kestabilan

    os.makedirs(req.out_dir, exist_ok=True)
    path = req.out_dir / algorithm
    mod.save_model(mod.finalize_model(model), str(path))
    logger.info(f"Save Model at {path}")

    get_model_cache().clear()
    return TrainedModel.from_saved(algorithm, path, metric)

  @staticmethod
  def retrain_models(req: SupervisedTrainRequest, algorithms: List[str], logger) -> List[TrainedModel]:
    """
    Retrain a specific set of algorithms on `req.df` in a single setup.

    Pure compute (no Dataset/DB): saves artifacts into `req.out_dir` and returns
    their descriptors. Used by the finetune service, which owns pulling the new
    data, persistence, and dataset bookkeeping.
    """
    mod = Supervised._setup(req, logger)

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

    get_model_cache().clear()
    return results
