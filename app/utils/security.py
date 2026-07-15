"""
Security utilities for path sanitization and validation
Prevents path traversal attacks and other security issues
"""

import re
from pathlib import Path
from typing import Optional
from fastapi import HTTPException


# Valid pattern for dataset/model/file names (alphanumeric, hyphen,
# underscore, dot - dot is needed for filenames like "data.csv"; ".."
# traversal is explicitly rejected above before this pattern is checked)
VALID_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+$")


def sanitize_path_component(name: str, context: str = "name") -> str:
  """
  Sanitize path component to prevent path traversal attacks.

  Args:
      name: The path component to sanitize
      context: Context for error messages (e.g., "dataset_name", "model_name")

  Returns:
      Sanitized name

  Raises:
      HTTPException: If name contains invalid characters
  """
  if not name or not isinstance(name, str):
    raise HTTPException(
        status_code=400, detail=f"Invalid {context}: must be a non-empty string"
    )

  # Check for path traversal attempts
  if ".." in name or "/" in name or "\\" in name:
    raise HTTPException(
        status_code=400, detail=f"Invalid {context}: path traversal detected"
    )

  # Validate characters
  if not VALID_NAME_PATTERN.match(name):
    raise HTTPException(
        status_code=400,
        detail=f"Invalid {context}: only alphanumeric, hyphen, and underscore allowed",
    )

  return name


def safe_path_join(base: Path, *components: str) -> Path:
  """
  Safely join path components and validate result is within base directory.

  Args:
      base: Base directory path
      components: Path components to join

  Returns:
      Resolved path that is guaranteed to be within base directory

  Raises:
      HTTPException: If resulting path is outside base directory
  """
  # Resolve base path
  base = base.resolve()

  # Sanitize all components
  safe_components = [sanitize_path_component(c) for c in components]

  # Join and resolve
  result = base.joinpath(*safe_components).resolve()

  # Ensure result is within base directory
  if not str(result).startswith(str(base)):
    raise HTTPException(
        status_code=400,
        detail="Path traversal detected: resulting path is outside allowed directory",
    )

  return result


def validate_dataset_name(name: str) -> str:
  """
  Validate dataset name.

  Args:
      name: Dataset name to validate

  Returns:
      Validated name

  Raises:
      HTTPException: If name is invalid
  """
  return sanitize_path_component(name, "dataset_name")


def validate_model_name(name: str) -> str:
  """
  Validate model name.

  Args:
      name: Model name to validate

  Returns:
      Validated name

  Raises:
      HTTPException: If name is invalid
  """
  return sanitize_path_component(name, "model_name")


def safe_file_path(
    base_dir: Path,
    *path_components: str,
    must_exist: bool = False,
    extension: Optional[str] = None,
) -> Path:
  """
  Get a safe file path with optional validation.

  Args:
      base_dir: Base directory
      path_components: Path components
      must_exist: If True, raises error if file doesn't exist
      extension: Optional file extension to validate

  Returns:
      Safe file path

  Raises:
      HTTPException: If path is invalid or file doesn't exist (when must_exist=True)
  """
  file_path = safe_path_join(base_dir, *path_components)

  # Validate extension if specified
  if extension and not str(file_path).endswith(extension):
    raise HTTPException(
        status_code=400, detail=f"Invalid file: must have {extension} extension"
    )

  # Check existence if required
  if must_exist and not file_path.exists():
    raise HTTPException(
        status_code=404, detail=f"File not found: {file_path.name}")

  return file_path
