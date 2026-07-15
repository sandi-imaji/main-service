"""
Unit tests for app.core.anomaly.
Pycaret is mocked so these tests run fast without a real trained model.
"""

import datetime
import pytest
import pandas as pd
from unittest.mock import Mock

from app.core.anomaly import Anomaly, AnomalyResultSchema
from app.config import Config


class FakeDataset:
  """Lightweight stand-in for the Dataset ORM object."""

  def __init__(self, name, meta=None, task_type=None, df=None, interval=5,
               preprocessing=None):
    self.name = name
    self.meta = meta or {}
    self.task_type = task_type
    self.interval = interval
    self.preprocessing = preprocessing
    self._df = df

  def open_dataframe(self):
    return self._df.copy()


# =============================================================================
# AnomalyResultSchema
# =============================================================================


class TestAnomalyResultSchema:
  def test_invalid_factory(self):
    result = AnomalyResultSchema.invalid("my-dataset")
    assert result.is_valid is False
    assert result.is_anomaly is False
    assert result.anomaly_score == 0.0
    assert result.features == {"": 0.0}
    assert result.dataset_name == "my-dataset"

  def test_task_type_property(self):
    result = AnomalyResultSchema.invalid("x")
    assert result.task_type == "Anomaly"

  def test_write_to_influx_calls_write_inference(self, mock_logger):
    ts = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
    result = AnomalyResultSchema(
        timestamp=ts,
        features={"f1": 1.0},
        is_anomaly=True,
        anomaly_score=0.9,
        dataset_name="ds-1",
        is_valid=True,
    )
    influx = Mock()
    result.write_to_influx(influx, mock_logger)

    influx.write_inference.assert_called_once()
    _, kwargs = influx.write_inference.call_args
    assert kwargs["dataset_name"] == "ds-1"
    assert kwargs["task_type"] == "Anomaly"
    assert kwargs["results"] == {"is_anomaly": True, "anomaly_score": 0.9}
    assert kwargs["features"] == {"f1": 1.0}


# =============================================================================
# Anomaly.get_cache / cache passthrough
# =============================================================================


class TestAnomalyCache:
  def test_get_cache_returns_model_cache(self):
    from app.utils.model_cache import ModelCache
    assert isinstance(Anomaly.get_cache(), ModelCache)

  def test_clear_cache_empties_cache(self):
    cache = Anomaly.get_cache()
    cache.put("some/path", Mock())
    Anomaly.clear_cache()
    assert cache.get("some/path") is None


# =============================================================================
# Anomaly.inference
# =============================================================================


class TestInference:
  def setup_method(self):
    Anomaly.clear_cache()

  def _mock_task_type(self, mock_mod):
    mock_task_type = Mock()
    mock_task_type.Anomaly.module.return_value = mock_mod
    return mock_task_type

  def test_inference_happy_path(self, monkeypatch, mock_logger):
    dataset = FakeDataset("Anomaly-ds")

    predicted = pd.DataFrame(
        {"Anomaly": [1], "Anomaly_Score": [0.75], "other_col": ["x"]}
    )
    mock_mod = Mock()
    mock_mod.load_model = Mock(return_value=Mock())
    mock_mod.predict_model = Mock(return_value=predicted)

    monkeypatch.setattr(
        "app.core.anomaly.TaskType", self._mock_task_type(mock_mod)
    )

    features = {"f1": [1.0], "f2": [2.0]}
    result = Anomaly.inference(dataset, features, mock_logger)

    assert isinstance(result, AnomalyResultSchema)
    assert result.is_valid is True
    assert result.is_anomaly is True
    assert result.anomaly_score == 0.75
    assert result.dataset_name == "Anomaly-ds"
    mock_mod.load_model.assert_called_once()

  def test_inference_uses_cached_model_on_second_call(self, monkeypatch, mock_logger):
    dataset = FakeDataset("Anomaly-ds-cache")
    predicted = pd.DataFrame({"Anomaly": [0], "Anomaly_Score": [0.1]})
    mock_mod = Mock()
    mock_mod.load_model = Mock(return_value=Mock())
    mock_mod.predict_model = Mock(return_value=predicted)

    monkeypatch.setattr(
        "app.core.anomaly.TaskType", self._mock_task_type(mock_mod)
    )

    features = {"f1": [1.0]}
    Anomaly.inference(dataset, features, mock_logger)
    Anomaly.inference(dataset, features, mock_logger)

    # Model file should only be loaded once; second call hits the cache.
    mock_mod.load_model.assert_called_once()

  def test_inference_error_returns_invalid_result(self, monkeypatch, mock_logger):
    dataset = FakeDataset("Anomaly-ds-err")
    mock_mod = Mock()
    mock_mod.load_model = Mock(side_effect=RuntimeError("model missing"))

    monkeypatch.setattr(
        "app.core.anomaly.TaskType", self._mock_task_type(mock_mod)
    )

    result = Anomaly.inference(dataset, {"f1": [1.0]}, mock_logger)

    assert result.is_valid is False
    assert result.is_anomaly is False
    assert result.dataset_name == ""
    mock_logger.warning.assert_called_once()


# =============================================================================
# Anomaly.auto_inference
# =============================================================================


class TestAutoInference:
  def test_uses_columns_from_meta(self, monkeypatch, mock_logger):
    dataset = FakeDataset("Anomaly-auto", meta={"columns": ["f1", "f2"]})

    monkeypatch.setattr(
        "app.core.anomaly.pull_realtime", lambda columns: [10.0, 20.0]
    )

    captured = {}

    def fake_inference(ds, features, logger):
      captured["features"] = features
      return "sentinel"

    monkeypatch.setattr(Anomaly, "inference", staticmethod(fake_inference))

    result = Anomaly.auto_inference(dataset, mock_logger)

    assert result == "sentinel"
    assert captured["features"] == {"f1": [10.0], "f2": [20.0]}

  def test_derives_columns_from_dataframe_when_meta_missing(
      self, monkeypatch, mock_logger
  ):
    df = pd.DataFrame({"dt": ["2024-01-01"], "f1": [1.0], "f2": [2.0]})
    dataset = FakeDataset("Anomaly-auto2", meta={}, df=df)

    monkeypatch.setattr(
        "app.core.anomaly.pull_realtime", lambda columns: [1.0, 2.0]
    )

    captured = {}

    def fake_inference(ds, features, logger):
      captured["features"] = features
      return "sentinel"

    monkeypatch.setattr(Anomaly, "inference", staticmethod(fake_inference))

    Anomaly.auto_inference(dataset, mock_logger)

    assert set(captured["features"].keys()) == {"f1", "f2"}

  def test_pull_falsy_defaults_to_zeros(self, monkeypatch, mock_logger):
    dataset = FakeDataset("Anomaly-auto3", meta={"columns": ["f1", "f2"]})

    monkeypatch.setattr("app.core.anomaly.pull_realtime", lambda columns: None)

    captured = {}

    def fake_inference(ds, features, logger):
      captured["features"] = features
      return "sentinel"

    monkeypatch.setattr(Anomaly, "inference", staticmethod(fake_inference))

    Anomaly.auto_inference(dataset, mock_logger)

    assert captured["features"] == {"f1": [0.0], "f2": [0.0]}


# =============================================================================
# Anomaly.anomaly (composite: inference + optional cluster cross-check)
# =============================================================================


class TestAnomalyComposite:
  def _valid_result(self, is_anomaly=True):
    return AnomalyResultSchema(
        timestamp=datetime.datetime.now(),
        features={"f1": 1.0},
        is_anomaly=is_anomaly,
        anomaly_score=0.5,
        dataset_name="ds",
        is_valid=True,
    )

  def test_non_clustering_task_returns_auto_inference_result_unchanged(
      self, monkeypatch, mock_logger
  ):
    mock_task_type = Mock()
    mock_task_type.is_clustering.return_value = False
    dataset = FakeDataset("ds", task_type=mock_task_type)

    result = self._valid_result(is_anomaly=True)
    monkeypatch.setattr(Anomaly, "auto_inference", staticmethod(lambda d, l: result))

    out = Anomaly.anomaly(dataset, mock_logger)
    assert out is result
    assert out.is_anomaly is True

  def test_clustering_task_confirms_anomaly_when_same_cluster(
      self, monkeypatch, mock_logger
  ):
    mock_core = Mock()
    mock_core.inference.return_value = Mock(clusters={"kmeans": "anomaly"})
    mock_core.is_same_cluster.return_value = True

    mock_task_type = Mock()
    mock_task_type.is_clustering.return_value = True
    mock_task_type.core.return_value = mock_core

    dataset = FakeDataset("ds", task_type=mock_task_type)

    result = self._valid_result(is_anomaly=True)
    monkeypatch.setattr(Anomaly, "auto_inference", staticmethod(lambda d, l: result))

    out = Anomaly.anomaly(dataset, mock_logger)
    assert out.is_anomaly is True

  def test_clustering_task_rejects_anomaly_when_different_cluster(
      self, monkeypatch, mock_logger
  ):
    mock_core = Mock()
    mock_core.inference.return_value = Mock(clusters={"kmeans": "normal"})
    mock_core.is_same_cluster.return_value = False

    mock_task_type = Mock()
    mock_task_type.is_clustering.return_value = True
    mock_task_type.core.return_value = mock_core

    dataset = FakeDataset("ds", task_type=mock_task_type)

    result = self._valid_result(is_anomaly=True)
    monkeypatch.setattr(Anomaly, "auto_inference", staticmethod(lambda d, l: result))

    out = Anomaly.anomaly(dataset, mock_logger)
    assert out.is_anomaly is False


# =============================================================================
# Anomaly.train
# =============================================================================


class TestTrain:
  def test_train_saves_model_and_invalidates_cache(self, tmp_path, monkeypatch, mock_logger):
    dataset_name = "Anomaly-train"
    storage = tmp_path / "storages" / dataset_name
    storage.mkdir(parents=True)
    monkeypatch.setattr(Config, "dir", tmp_path)

    df = pd.DataFrame({"dt": ["2024-01-01", "2024-01-02"], "f1": [1.0, 2.0]})
    dataset = FakeDataset(dataset_name, df=df)

    labeled = df.drop(columns="dt").copy()
    labeled["Anomaly"] = [0, 1]

    mock_mod = Mock()
    mock_mod.setup = Mock()
    mock_mod.create_model = Mock(return_value=Mock())
    mock_mod.assign_model = Mock(return_value=labeled)
    mock_mod.save_model = Mock()

    monkeypatch.setattr("app.core.anomaly.mod", mock_mod)

    cache = Anomaly.get_cache()
    cache.put(f"{storage}/anomaly", Mock())

    result_df = Anomaly.train(dataset, "iforest", 0.1, mock_logger)

    assert (storage / "anomaly.csv").exists()
    mock_mod.save_model.assert_called_once()
    assert cache.get(f"{storage}/anomaly") is None
    assert "Anomaly" in result_df.columns


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
