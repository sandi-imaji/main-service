"""app.core.supervised — pure inference, PyCaret mocked."""
import pandas as pd
import pytest

from app.core.contracts import PredictRequest
from app.core.supervised import Supervised, module
from app.database.schemas import TaskType


class FakeMod:
  """Stand-in for pycaret.regression / classification."""
  def __init__(self, label):
    self._label = label
  def load_model(self, path):
    return f"model@{path}"
  def predict_model(self, model, df):
    out = df.copy()
    out["prediction_label"] = [self._label] * len(df)
    return out


class TestModule:
  def test_regression_and_classification(self):
    from pycaret import regression, classification
    assert module(TaskType.Regression) is regression
    assert module(TaskType.Classification) is classification

  def test_unsupported_raises(self):
    with pytest.raises(ValueError):
      module(TaskType.Clustering)


class TestPredict:
  def setup_method(self):
    Supervised.clear_cache()

  def test_predict_returns_prediction_per_model(self, monkeypatch, mock_logger):
    monkeypatch.setattr("app.core.supervised.module", lambda tt: FakeMod(20.5))

    req = PredictRequest(
        features={"x1": [1.0], "x2": [2.0]},
        models=[("lr", "/models/lr"), ("knn", "/models/knn")],
        task="Regression")

    result = Supervised.predict(req, mock_logger)

    assert result.predictions == {"lr": 20.5, "knn": 20.5}
    assert result.features == {"x1": 1.0, "x2": 2.0}

  def test_predict_uses_cache_for_repeated_path(self, monkeypatch, mock_logger):
    calls = {"n": 0}

    class CountingMod(FakeMod):
      def load_model(self, path):
        calls["n"] += 1
        return super().load_model(path)

    monkeypatch.setattr("app.core.supervised.module", lambda tt: CountingMod(1.0))
    req = PredictRequest(features={"x1": [1.0]},
                         models=[("lr", "/models/same"), ("lr2", "/models/same")],
                         task="Regression")
    Supervised.predict(req, mock_logger)
    assert calls["n"] == 1  # second model reuses the cached load for same path
