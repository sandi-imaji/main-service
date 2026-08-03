"""
Base for the pure ML core.

Shared, task-agnostic helpers only: the per-task model cache (LRU) and the
`ALGO_LIST` loaded from `algorithm_list.json`. Deliberately free of any
Dataset / DB / HTTP knowledge — the core talks to the service layer solely
through `app.core.contracts`.
"""

import json
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

from app.config import Config
from app.utils.model_cache import ModelCache


# Algorithm catalogue ({task_type: {algorithm: description}}), read once at import.
_ALGO_PATH = Config.dir / "algorithm_list.json"
if not _ALGO_PATH.exists(): raise ValueError(f"algorithm list : [{_ALGO_PATH}] is not found!")
with open(_ALGO_PATH, "r") as f: ALGO_LIST = json.load(f)


class BaseMLCore(ABC):
  """
  Abstract base for every ML core module.

  Provides the shared model-cache plumbing; each subclass only has to say which
  cache it uses via `get_cache()`. All actual compute (predict / forecast /
  compare_models / …) lives on the task-specific subclasses.
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
    """Cache statistics (hits, misses, size, hit_rate, …)."""
    stats = cls.get_cache().get_stats()
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
    cls.get_cache().clear()

  @classmethod
  def _load_model_cached(cls, mod: Any, model_path: str, logger: Any) -> Any:
    """Load a pycaret model, serving from (and populating) the shared cache."""
    cache = cls.get_cache()
    cached_model = cache.get(model_path)

    if cached_model is not None:
      logger.debug(f"Model cache hit: {model_path}")
      return cached_model

    logger.debug(f"Model cache miss, loading: {model_path}")
    model = mod.load_model(str(model_path))
    cache.put(model_path, model)

    return model
