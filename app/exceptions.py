"""
Common exceptions and error handling utilities
"""

from fastapi import HTTPException
from typing import Any, Dict, Optional
from enum import Enum


class ErrorCode(Enum):
  """Standardized error codes"""

  NOT_FOUND = "NOT_FOUND"
  VALIDATION_ERROR = "VALIDATION_ERROR"
  INTERNAL_ERROR = "INTERNAL_ERROR"
  ALREADY_EXISTS = "ALREADY_EXISTS"
  INVALID_STATE = "INVALID_STATE"
  EXTERNAL_API_ERROR = "EXTERNAL_API_ERROR"


class AppException(HTTPException):
  """Base application exception with consistent format"""

  def __init__(
      self,
      status_code: int,
      error_code: ErrorCode,
      message: str,
      details: Optional[Dict[str, Any]] = None,
  ):
    super().__init__(
        status_code=status_code,
        detail={
            "error_code": error_code.value,
            "message": message,
            "details": details or {},
        },
    )
    self.error_code = error_code

class NotFoundException(AppException):
  """Resource not found"""

  def __init__(self, resource: str, identifier: str):
    super().__init__(
        status_code=404,
        error_code=ErrorCode.NOT_FOUND,
        message=f"{resource} not found",
        details={"resource": resource, "identifier": identifier},
    )


class ValidationException(AppException):
  """Validation error"""

  def __init__(self, message: str, field: Optional[str] = None):
    super().__init__(
        status_code=422,
        error_code=ErrorCode.VALIDATION_ERROR,
        message=message,
        details={"field": field} if field else {},
    )


class InvalidStateException(AppException):
  """Invalid state for operation"""

  def __init__(self, resource: str, current_state: str, expected_state: str):
    super().__init__(
        status_code=400,
        error_code=ErrorCode.INVALID_STATE,
        message=f"Invalid state for {resource}",
        details={"current_state": current_state,
                 "expected_state": expected_state},
    )


class ExternalAPIException(AppException):
  """External API error"""

  def __init__(self, service: str, message: str):
    super().__init__(
        status_code=503,
        error_code=ErrorCode.EXTERNAL_API_ERROR,
        message=f"External service error: {service}",
        details={"service": service, "error": message},
    )


def handle_exception(e: Exception, logger=None) -> AppException:
  """Convert any exception to AppException"""
  if isinstance(e, AppException):
    return e

  if isinstance(e, HTTPException):
    # Preserve the original status code/detail (e.g. a domain-raised 404)
    # instead of masking it as a generic 500.
    return e

  if logger:
    logger.error(f"Unexpected error: {str(e)}")

  return AppException(
      status_code=500,
      error_code=ErrorCode.INTERNAL_ERROR,
      message="Internal server error",
      details={"original_error": str(e)},
  )
