"""app.services.validators.split_data — train/test split with guards."""
from types import SimpleNamespace

import pandas as pd
import pytest

from app.exceptions import ValidationException
from app.services import validators


def _payload(df, target, name="ds", train_size=0.7):
  dataset = SimpleNamespace(
      name=name, target=target,
      check_integrity=lambda: None,
      get_df=lambda: df,
  )
  return SimpleNamespace(dataset=dataset, train_size=train_size)


def test_raises_when_target_missing():
  df = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6]})
  with pytest.raises(ValidationException):
    validators.split_data(_payload(df, target="y"))


def test_splits_and_writes_csvs(monkeypatch, tmp_path):
  monkeypatch.setattr(validators.Config, "dir", tmp_path)
  (tmp_path / "storages" / "ds").mkdir(parents=True)
  df = pd.DataFrame({"x1": list(range(10)), "y": list(range(10))})

  out = validators.split_data(_payload(df, target="y", train_size=0.7))

  assert out["train"] + out["test"] == 10
  assert (tmp_path / "storages" / "ds" / "train.csv").exists()
  assert (tmp_path / "storages" / "ds" / "test.csv").exists()
