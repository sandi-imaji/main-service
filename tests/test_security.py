"""
Unit tests for app.utils.security path-sanitization helpers.

Added alongside the dataset/model route tests because a bug here
(VALID_NAME_PATTERN rejecting ".") was silently breaking every endpoint
that touches "data.csv" (sample, describe, pca, delete, ...); these tests
lock in both the fix (dots allowed in filenames) and the traversal
protections that must keep working.
"""

import pytest
from pathlib import Path
from fastapi import HTTPException

from app.utils.security import (
    sanitize_path_component,
    safe_path_join,
    validate_dataset_name,
    validate_model_name,
    safe_file_path,
)


class TestSanitizePathComponent:
  def test_allows_alphanumeric_hyphen_underscore(self):
    assert sanitize_path_component("Regression-abcd1234_x") == "Regression-abcd1234_x"

  def test_allows_dot_for_filenames(self):
    assert sanitize_path_component("data.csv") == "data.csv"

  def test_rejects_empty_string(self):
    with pytest.raises(HTTPException) as exc:
      sanitize_path_component("")
    assert exc.value.status_code == 400

  def test_rejects_double_dot_traversal(self):
    with pytest.raises(HTTPException):
      sanitize_path_component("../etc/passwd")

  def test_rejects_forward_slash(self):
    with pytest.raises(HTTPException):
      sanitize_path_component("a/b")

  def test_rejects_backslash(self):
    with pytest.raises(HTTPException):
      sanitize_path_component("a\\b")

  def test_rejects_special_characters(self):
    with pytest.raises(HTTPException):
      sanitize_path_component("name;rm -rf")


class TestSafePathJoin:
  def test_joins_within_base_directory(self, tmp_path):
    result = safe_path_join(tmp_path, "storages", "Regression-x", "data.csv")
    assert result == (tmp_path / "storages" / "Regression-x" / "data.csv").resolve()

  def test_rejects_traversal_outside_base(self, tmp_path):
    with pytest.raises(HTTPException):
      safe_path_join(tmp_path, "..", "..", "etc", "passwd")

  def test_rejects_component_with_traversal_sequence(self, tmp_path):
    with pytest.raises(HTTPException):
      safe_path_join(tmp_path, "storages", "..%2f..%2fetc")


class TestValidateNameHelpers:
  def test_validate_dataset_name_passthrough(self):
    assert validate_dataset_name("Regression-abcd1234") == "Regression-abcd1234"

  def test_validate_model_name_passthrough(self):
    assert validate_model_name("lr-abcd1234") == "lr-abcd1234"

  def test_validate_dataset_name_rejects_traversal(self):
    with pytest.raises(HTTPException):
      validate_dataset_name("../secrets")


class TestSafeFilePath:
  def test_returns_path_when_extension_matches(self, tmp_path):
    result = safe_file_path(tmp_path, "storages", "ds", "data.csv", extension=".csv")
    assert str(result).endswith("data.csv")

  def test_rejects_wrong_extension(self, tmp_path):
    with pytest.raises(HTTPException):
      safe_file_path(tmp_path, "storages", "ds", "data.csv", extension=".json")

  def test_must_exist_raises_404_when_missing(self, tmp_path):
    with pytest.raises(HTTPException) as exc:
      safe_file_path(tmp_path, "data.csv", must_exist=True)
    assert exc.value.status_code == 404

  def test_must_exist_passes_when_present(self, tmp_path):
    (tmp_path / "data.csv").write_text("x")
    result = safe_file_path(tmp_path, "data.csv", must_exist=True)
    assert result.exists()


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
