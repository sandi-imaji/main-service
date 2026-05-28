"""
Unified Model Cache
Consolidated LRU cache untuk semua ML model types.
Reduces code duplication dari 4 separate implementations.
"""

from typing import Dict, Any, Optional, Generic, TypeVar
from threading import Lock
from dataclasses import dataclass

T = TypeVar("T")


@dataclass
class CacheStats:
  """Cache statistics container."""

  hits: int
  misses: int
  size: int
  max_size: int
  hit_rate: float


class ModelCache(Generic[T]):
  """
  Thread-safe LRU cache untuk ML models.

  Features:
  - LRU eviction policy
  - Thread-safe operations
  - Statistics tracking
  - Generic type support

  Usage:
      # Get or create cache
      cache = get_model_cache("supervised", max_size=20)

      # Store model
      cache.put("/path/to/model", loaded_model)

      # Retrieve model
      model = cache.get("/path/to/model")

      # Get stats
      stats = cache.get_stats()
  """

  def __init__(self, max_size: int = 20, name: str = "default"):
    self._cache: Dict[str, T] = {}
    self._access_order: list = []
    self._max_size = max_size
    self._lock = Lock()
    self._hits = 0
    self._misses = 0
    self._name = name

  def get(self, model_path: str) -> Optional[T]:
    """
    Get model from cache if available.

    Args:
        model_path: Path to the model file

    Returns:
        Cached model or None if not found
    """
    with self._lock:
      if model_path in self._cache:
        # Move to end (most recently used)
        self._access_order.remove(model_path)
        self._access_order.append(model_path)
        self._hits += 1
        return self._cache[model_path]
      self._misses += 1
      return None

  def put(self, model_path: str, model: T) -> None:
    """
    Add model to cache, evicting LRU if necessary.

    Args:
        model_path: Path to the model file
        model: Model object to cache
    """
    with self._lock:
      if model_path in self._cache:
        self._access_order.remove(model_path)
      elif len(self._cache) >= self._max_size:
        # Evict least recently used
        lru_key = self._access_order.pop(0)
        del self._cache[lru_key]

      self._cache[model_path] = model
      self._access_order.append(model_path)

  def invalidate(self, model_path: str) -> bool:
    """
    Remove specific model from cache.

    Args:
        model_path: Path to the model file

    Returns:
        True if model was removed, False if not found
    """
    with self._lock:
      if model_path in self._cache:
        del self._cache[model_path]
        self._access_order.remove(model_path)
        return True
      return False

  def clear(self) -> None:
    """Clear all models from cache."""
    with self._lock:
      self._cache.clear()
      self._access_order.clear()

  def get_stats(self) -> CacheStats:
    """
    Get cache statistics.

    Returns:
        CacheStats object with hit rate, size, etc.
    """
    with self._lock:
      total = self._hits + self._misses
      hit_rate = (self._hits / total * 100) if total > 0 else 0
      return CacheStats(
          hits=self._hits,
          misses=self._misses,
          size=len(self._cache),
          max_size=self._max_size,
          hit_rate=round(hit_rate, 2),
      )

  def __contains__(self, model_path: str) -> bool:
    """Check if model is in cache."""
    with self._lock:
      return model_path in self._cache

  def __len__(self) -> int:
    """Get number of cached models."""
    with self._lock:
      return len(self._cache)


# Global cache registry
_caches: Dict[str, ModelCache] = {}


def get_model_cache(name: str, max_size: int = 20) -> ModelCache:
  """
  Get or create cache instance.

  Args:
      name: Cache name (e.g., 'supervised', 'unsupervised', 'anomaly', 'timeseries')
      max_size: Maximum number of models to cache

  Returns:
      ModelCache instance
  """
  if name not in _caches:
    _caches[name] = ModelCache(max_size=max_size, name=name)
  return _caches[name]


def get_all_caches() -> Dict[str, ModelCache]:
  """Get all registered caches."""
  return _caches.copy()


def clear_all_caches() -> None:
  """Clear all registered caches."""
  for cache in _caches.values():
    cache.clear()


def get_all_cache_stats() -> Dict[str, CacheStats]:
  """Get statistics from all caches."""
  return {name: cache.get_stats() for name, cache in _caches.items()}


# Pre-defined cache instances untuk backward compatibility
# These replace the individual cache implementations in core modules


def get_supervised_cache() -> ModelCache:
  """Get cache untuk supervised learning models."""
  return get_model_cache("supervised", max_size=20)


def get_unsupervised_cache() -> ModelCache:
  """Get cache untuk unsupervised learning models."""
  return get_model_cache("unsupervised", max_size=20)


def get_anomaly_cache() -> ModelCache:
  """Get cache untuk anomaly detection models."""
  return get_model_cache("anomaly", max_size=10)


def get_timeseries_cache() -> ModelCache:
  """Get cache untuk time series models."""
  return get_model_cache("timeseries", max_size=15)
