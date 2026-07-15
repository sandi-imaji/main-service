"""
Regression-specific unit tests for app.core.supervised.

Broad Supervised behavior (inference, training, caching, read_results, ...)
is already covered by test_supervised.py. This file focuses on paths that
are specific to the regression auto-inference/finetune flow and were not
previously exercised: the "all-pulled-values-are-zero" guard in
auto_inference, write_to_influx wiring, the get_config() helper, and the
finetune() error path.
"""

import datetime
from contextlib import contextmanager
import pytest
from unittest.mock import Mock

from app.core.supervised import Supervised, SupervisedResultSchema, get_config
from app.database.schemas import TaskType


class FakeModelML:
  def __init__(self, algorithm, name=None):
    self.algorithm = algorithm
    self.name = name or algorithm


class FakeDataset:
  def __init__(self, name, features, target, models=None):
    self.name = name
    self.features = features
    self.target = target
    self.models = models or []


# =============================================================================
# auto_inference: all-pulled-values-are-zero guard
# =============================================================================


class TestAutoInferenceZeroGuard:
  def test_returns_zeroed_result_without_calling_inference(self, monkeypatch, mock_logger):
    dataset = FakeDataset(
        "Regression-x", ["f1", "f2"], "target",
        models=[FakeModelML("lr"), FakeModelML("rf")],
    )
    monkeypatch.setattr(
        "app.core.supervised.pull_realtime", lambda columns, logger: [0, 0, 0]
    )
    inference_called = {"called": False}
    monkeypatch.setattr(
        Supervised, "inference",
        staticmethod(lambda *a, **k: inference_called.update(called=True)),
    )

    result = Supervised.auto_inference(dataset, mock_logger)

    assert inference_called["called"] is False
    assert result.actual == 0
    assert result.predictions == {"lr": 0.0, "rf": 0.0}
    assert result.features == {"f1": 0.0, "f2": 0.0}


class TestAutoInferenceNormalFlow:
  def test_pulls_features_and_sets_actual_from_last_value(self, monkeypatch, mock_logger):
    dataset = FakeDataset("Regression-y", ["f1", "f2"], "target", models=[FakeModelML("lr")])
    monkeypatch.setattr(
        "app.core.supervised.pull_realtime",
        lambda columns, logger: [1.5, 2.5, 42.0],
    )

    captured = {}

    def fake_inference(ds, features, logger):
      captured["features"] = features
      return SupervisedResultSchema(
          features={}, predictions={"lr": 10.0},
          timestamp=datetime.datetime.now(), dataset_name=ds.name, is_valid=True,
      )

    monkeypatch.setattr(Supervised, "inference", staticmethod(fake_inference))

    result = Supervised.auto_inference(dataset, mock_logger)

    assert captured["features"] == {"f1": [1.5], "f2": [2.5]}
    assert result.actual == 42.0
    assert result.predictions == {"lr": 10.0}


# =============================================================================
# SupervisedResultSchema.write_to_influx
# =============================================================================


class TestWriteToInflux:
  def test_forwards_predictions_features_and_actual(self, mock_logger):
    ts = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
    result = SupervisedResultSchema(
        features={"f1": 1.0},
        predictions={"lr": 20.5},
        timestamp=ts,
        actual=21.0,
        dataset_name="Regression-z",
        is_valid=True,
    )
    influx = Mock()
    result.write_to_influx(influx, mock_logger)

    influx.write_inference.assert_called_once()
    _, kwargs = influx.write_inference.call_args
    assert kwargs["dataset_name"] == "Regression-z"
    assert kwargs["results"] == {"lr": 20.5}
    assert kwargs["features"] == {"f1": 1.0}
    assert kwargs["actual"] == 21.0


# =============================================================================
# get_config helper
# =============================================================================


class TestGetConfig:
  def test_logs_feature_names_and_target(self, mock_logger):
    mock_mod = Mock()
    mock_columns = Mock()
    mock_columns.columns.tolist.return_value = ["f1", "f2"]
    mock_mod.get_config = Mock(side_effect=lambda key: {
        "X": mock_columns, "target_param": "target",
    }[key])

    get_config(mock_mod, mock_logger)

    assert mock_logger.info.call_count == 2


# =============================================================================
# finetune: dataset-not-found error path (doesn't touch the real DB file)
# =============================================================================


class TestFinetuneErrorPath:
  def test_missing_dataset_logs_error_and_does_not_raise(
      self, monkeypatch, db_session, mock_logger
  ):
    @contextmanager
    def fake_get_db_session():
      yield db_session

    monkeypatch.setattr(
        "app.core.supervised.get_db_session", fake_get_db_session
    )

    Supervised.finetune("nonexistent-dataset-xyz", mock_logger)

    mock_logger.error.assert_called_once()
    assert "not found" in mock_logger.error.call_args[0][0].lower()


# =============================================================================
# TaskType.Regression sanity (used to route to Supervised core)
# =============================================================================


class TestRegressionTaskType:
  def test_regression_resolves_to_supervised_core(self):
    assert TaskType.Regression.core() is Supervised

  def test_regression_has_algorithms(self):
    algos = TaskType.Regression.algorithms()
    assert isinstance(algos, dict)
    assert len(algos) > 0


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
