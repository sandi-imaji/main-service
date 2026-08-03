"""app.core.anomaly — pure inference, PyCaret mocked."""
import pandas as pd

from app.core.anomaly import Anomaly
from app.core.contracts import AnomalyResult, PredictRequest


class FakeAnomalyMod:
  def __init__(self, flag, score):
    self._flag, self._score = flag, score
  def load_model(self, path):
    return f"model@{path}"
  def predict_model(self, model, df):
    out = df.copy()
    out["Anomaly"] = [self._flag] * len(df)
    out["Anomaly_Score"] = [self._score] * len(df)
    return out


class TestPredict:
  def setup_method(self):
    Anomaly.clear_cache()

  def test_predict_flags_anomaly(self, monkeypatch, mock_logger):
    monkeypatch.setattr("app.core.anomaly.mod", FakeAnomalyMod(1, 0.87))
    req = PredictRequest(features={"x1": [1.0], "x2": [2.0]},
                         models=[("anomaly", "/models/anomaly")], task="Anomaly")

    result = Anomaly.predict(req, mock_logger)

    assert isinstance(result, AnomalyResult)
    assert result.is_anomaly is True
    assert result.anomaly_score == 0.87
    assert result.features == {"x1": 1.0, "x2": 2.0}

  def test_predict_normal_point(self, monkeypatch, mock_logger):
    monkeypatch.setattr("app.core.anomaly.mod", FakeAnomalyMod(0, 0.12))
    req = PredictRequest(features={"x1": [1.0]},
                         models=[("anomaly", "/models/anomaly")], task="Anomaly")

    result = Anomaly.predict(req, mock_logger)
    assert result.is_anomaly is False
