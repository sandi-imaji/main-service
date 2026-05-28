"""
Base route patterns and utilities
"""

import asyncio
from functools import wraps
from typing import TypeVar, Callable, Any
from fastapi import HTTPException
from app.logger import Logger
from app.exceptions import handle_exception

T = TypeVar("T")


def with_logging(operation_name: str):
  """Decorator to add logging to route handlers"""

  def decorator(func: Callable[..., T]) -> Callable[..., T]:
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
      # Create logger if not provided
      if not kwargs.get("logger"):
        dataset_name = kwargs.get("dataset_name", "")
        kwargs["logger"] = Logger(dataset_name)
      logger = kwargs["logger"]

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
      # Create logger if not provided
      if not kwargs.get("logger"):
        dataset_name = kwargs.get("dataset_name", "")
        kwargs["logger"] = Logger(dataset_name)
      logger = kwargs["logger"]

      logger.info(f"Starting {operation_name}...")
      try:
        result = func(*args, **kwargs)
        logger.info(f"{operation_name} completed successfully")
        return result
      except Exception as e:
        logger.error(f"{operation_name} failed: {str(e)}")
        raise handle_exception(e, logger)

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

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
  from app.database.orm import Dataset
  from app.exceptions import NotFoundException

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
  from app.database.orm import ModelML
  from app.exceptions import NotFoundException

  model = ModelML.get_by_name(name, db)
  if not model:
    raise NotFoundException("Model", name)
  return model
