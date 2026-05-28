"""
Base Core Module for ML Operations
Provides common functionality for all ML task types (Supervised, Unsupervised, TimeSeries, Anomaly)
"""

import os
import asyncio
from typing import Any, Dict, Optional, TypeVar, Callable
from abc import ABC, abstractmethod
from functools import wraps

from app.config import Config
from app.helpers import DTEncoder
from app.database.schemas import MetaModel, StatusProcess
from app.database.orm import ModelML
from app.utils.model_cache import ModelCache

T = TypeVar("T")

class ResultSchema():
  def model_dump(self): raise NotImplementedError()

def async_wrap(func: Callable[..., T]) -> Callable[..., asyncio.Future[T]]:
  """
  Decorator to create async version of a sync function.
  Wraps the sync function in run_in_executor for non-blocking execution.
  """

  @wraps(func)
  async def async_wrapper(*args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
  return async_wrapper


class BaseMLCore(ABC):
  """
  Abstract base class for all ML core modules.

  Provides common functionality:
  - Model caching with LRU
  - Cache statistics
  - Async wrapper for sync methods
  - Common training patterns
  """

  # To be overridden by subclasses
  _cache: Optional[ModelCache] = None
  _task_type: str = "unknown"

  @classmethod
  @abstractmethod
  def get_cache(cls) -> ModelCache:
    """Get the model cache for this task type. Must be implemented by subclasses."""
    pass

  @classmethod
  def get_cache_stats(cls) -> Dict[str, Any]:
    """
    Get model cache statistics.

    Returns:
        Dictionary with cache statistics (hits, misses, size, hit_rate, etc.)
    """
    cache = cls.get_cache()
    stats = cache.get_stats()
    return {
        "hits": stats.hits,
        "misses": stats.misses,
        "size": stats.size,
        "max_size": stats.max_size,
        "hit_rate": stats.hit_rate,
    }

  @classmethod
  def clear_cache(cls) -> None:
    """Clear the model cache."""
    cache = cls.get_cache()
    cache.clear()

  @classmethod
  def _load_model_cached(cls, mod: Any, model_path: str, logger: Any) -> Any:
    """
    Load model with caching support.

    Args:
        mod: PyCaret module (e.g., classification, clustering)
        model_path: Path to the model file
        logger: Logger instance

    Returns:
        Loaded model (cached if available)
    """
    cache = cls.get_cache()
    cached_model = cache.get(model_path)

    if cached_model is not None:
      logger.debug(f"Model cache hit: {model_path}")
      return cached_model

    logger.debug(f"Model cache miss, loading: {model_path}")
    model = mod.load_model(str(model_path))
    cache.put(model_path, model)

    return model

  @classmethod
  def _create_model_metadata(
      cls,
      algorithm: str,
      path_model: str,
      metric: Dict[str, Any],
      train_time: float = 0.0,
  ) -> tuple[str, MetaModel]:
    """
    Create model name and metadata for a trained model.

    Args:
        algorithm: Algorithm name
        path_model: Path to saved model
        metric: Evaluation metrics dictionary
        train_time: Training time in seconds

    Returns:
        Tuple of (model_name, metadata)
    """
    import uuid

    model_name = f"{algorithm}-{str(uuid.uuid4())[:8]}"

    meta = MetaModel(
        created_by="Anonymous",
        created_at=DTEncoder.now(),
        train_time=train_time,
        size_of=os.path.getsize(f"{Config.dir}/{path_model}.pkl"),
        notes="",
    )

    return model_name, meta

  @classmethod
  def _save_model_to_db(
      cls,
      dataset: Any,
      model_name: str,
      algorithm: str,
      path_model: str,
      metric: Dict[str, Any],
      meta: MetaModel,
      logger: Any,
  ) -> ModelML:
    """
    Create and save ModelML object to dataset.

    Args:
        dataset: Dataset object
        model_name: Model name
        algorithm: Algorithm name
        path_model: Path to model
        metric: Evaluation metrics
        meta: Model metadata
        logger: Logger instance

    Returns:
        Created ModelML object
    """
    q_model = ModelML(
        name=model_name,
        algorithm=algorithm,
        is_active=True,
        evaluation=metric,
        meta=meta.model_dump(),
        status=StatusProcess.SUCCESS_TRAIN,
        path=path_model,
    )

    dataset.models.append(q_model)
    logger.info(f"Model {model_name} ({algorithm}) saved to dataset")

    return q_model

  @classmethod
  def train(cls, dataset_name: str, **kwargs) -> Dict[str, Any]:
    """Train models - should be implemented by subclasses."""
    raise NotImplementedError("Subclasses must implement train() method")

  @classmethod
  def inference(cls, dataset: Any, **kwargs) -> ResultSchema:
    """Run inference - should be implemented by subclasses."""
    raise NotImplementedError("Subclasses must implement inference() method")

  @classmethod
  def auto_inference(cls, dataset: Any, **kwargs) -> ResultSchema:
    """Auto inference - should be implemented by subclasses."""
    raise NotImplementedError(
        "Subclasses must implement auto_inference() method")


class InferenceMixin:
  """
  Mixin class providing common inference functionality.
  Can be used by core modules that need async inference support.
  """

  @staticmethod
  def inference_async(dataset: Any, **kwargs):
    """
    Async wrapper for inference method.
    Note: This should be overridden by subclasses or used as a template.

    Usage:
        result = await Supervised.inference_async(dataset, features=features)
    """
    raise NotImplementedError("Subclasses should implement inference_async")

  @staticmethod
  def auto_inference_async(dataset: Any, **kwargs):
    """
    Async wrapper for auto_inference method.
    Note: This should be overridden by subclasses or used as a template.

    Usage:
        result = await Supervised.auto_inference_async(dataset)
    """
    raise NotImplementedError(
        "Subclasses should implement auto_inference_async")




