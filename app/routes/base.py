"""
Base route patterns and utilities
"""

import asyncio
import inspect
from functools import wraps
from typing import TypeVar, Callable, Any
from app.logger import Logger
from app.exceptions import handle_exception

from app.database.DB import Dataset,ModelML
from app.exceptions import NotFoundException

T = TypeVar("T")


def _without_logger(sig: inspect.Signature) -> inspect.Signature:
  """Drop the injected `logger` parameter so FastAPI never sees it (otherwise it
  leaks into the OpenAPI schema as a stray query parameter)."""
  params = [p for name, p in sig.parameters.items() if name != "logger"]
  return sig.replace(parameters=params)


def with_logging(operation_name: str):
  """Decorator to add logging to route handlers.

  Injects a `Logger` as the handler's `logger` kwarg, but hides that parameter
  from the handler's public signature so it does not surface in OpenAPI.
  """

  def decorator(func: Callable[..., T]) -> Callable[..., T]:
    def _inject_logger(kwargs):
      if not kwargs.get("logger"):
        # handlers name their path param `name` or `dataset_name`.
        dataset_name = kwargs.get("dataset_name") or kwargs.get("name", "")
        kwargs["logger"] = Logger(dataset_name)
      return kwargs["logger"]

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
      logger = _inject_logger(kwargs)
      logger.info(f"Starting {operation_name}...")
      try:
        result = await func(*args, **kwargs)
        logger.info(f"{operation_name} completed successfully")
        return result
      except Exception as e:
        logger.error(f"{operation_name} failed: {str(e)}")
        raise handle_exception(e, logger)

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
      logger = _inject_logger(kwargs)
      logger.info(f"Starting {operation_name}...")
      try:
        result = func(*args, **kwargs)
        logger.info(f"{operation_name} completed successfully")
        return result
      except Exception as e:
        logger.error(f"{operation_name} failed: {str(e)}")
        raise handle_exception(e, logger)

    wrapper = async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    # FastAPI reads the signature to build params/OpenAPI — hide `logger`.
    wrapper.__signature__ = _without_logger(inspect.signature(func))
    return wrapper

  return decorator


def require_dataset(func: Callable[..., T]) -> Callable[..., T]:
  """Decorator to ensure dataset exists and is valid"""

  @wraps(func)
  def wrapper(*args, **kwargs):
    dataset = kwargs.get("dataset")
    if not dataset:
      from app.exceptions import NotFoundException

      raise NotFoundException("Dataset", kwargs.get("dataset_name", "unknown"))

    if not dataset.is_valid:
      from app.exceptions import InvalidStateException

      raise InvalidStateException("Dataset", str(dataset.status), "valid")

    return func(*args, **kwargs)

  return wrapper


def require_trained_model(func: Callable[..., T]) -> Callable[..., T]:
  """Decorator to ensure dataset has trained models"""

  @wraps(func)
  def wrapper(*args, **kwargs):
    dataset = kwargs.get("dataset")
    if not dataset or not dataset.models:
      from app.exceptions import InvalidStateException

      raise InvalidStateException("Dataset", "untrained", "trained")
    return func(*args, **kwargs)

  return wrapper


# =============================================================================
# Helper Functions for Dataset and Model Retrieval
# =============================================================================


def get_dataset_or_404(name: str, db) -> "Dataset":
  """
  Get dataset by name or raise 404 exception.

  Args:
    name: Dataset name
    db: Database session

  Returns:
    Dataset object

  Raises:
    NotFoundException: If dataset not found
  """

  dataset = Dataset.get_by_name(name, db)
  if not dataset:
    raise NotFoundException("Dataset", name)
  return dataset


def get_model_or_404(name: str, db) -> "ModelML":
  """
  Get model by name or raise 404 exception.

  Args:
    name: Model name
    db: Database session

  Returns:
    ModelML object

  Raises:
    NotFoundException: If model not found
  """

  model = ModelML.get_by_name(name, db)
  if not model:
    raise NotFoundException("Model", name)
  return model
