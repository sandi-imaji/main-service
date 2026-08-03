"""Persistence of the anomaly detection summary.

Anomaly has no `ModelML` row — it attaches to another task's dataset and keeps a
single model at `<storage>/anomaly.pkl`. Its metrics therefore cannot ride along
in `model.evaluation`, and they used to disappear entirely: `auto_anomaly`
discarded the return value of `Anomaly.train_one`, so the summary the core
computed was never visible to anyone.
"""
import json
from types import SimpleNamespace

from app.services import anomaly as anomaly_service


class _Dataset:
  def __init__(self, path):
    self.path = path
    self.name = "Regression-ano"


class TestReadWrite:
  def test_round_trip(self, tmp_path, mock_logger):
    ds = _Dataset(tmp_path)
    summary = {"AnomalyCount": 10, "AnomalyRate": 0.05, "Threshold": 0.0456}
    anomaly_service._save_metrics(ds, summary, mock_logger)
    assert anomaly_service.read_metrics(ds) == summary

  def test_written_next_to_the_model_artifact(self, tmp_path, mock_logger):
    """Must sit beside `anomaly.pkl`, not under `top_model/`."""
    ds = _Dataset(tmp_path)
    anomaly_service._save_metrics(ds, {"AnomalyRate": 0.05}, mock_logger)
    assert (tmp_path / "anomaly_metrics.json").exists()

  def test_never_trained_returns_empty(self, tmp_path):
    """A legitimate state, not an error — older models have no such file."""
    assert anomaly_service.read_metrics(_Dataset(tmp_path)) == {}

  def test_corrupt_file_does_not_break_the_endpoint(self, tmp_path):
    (tmp_path / "anomaly_metrics.json").write_text("{not json")
    assert anomaly_service.read_metrics(_Dataset(tmp_path)) == {}

  def test_retraining_overwrites_rather_than_appends(self, tmp_path, mock_logger):
    ds = _Dataset(tmp_path)
    anomaly_service._save_metrics(ds, {"AnomalyRate": 0.05}, mock_logger)
    anomaly_service._save_metrics(ds, {"AnomalyRate": 0.10}, mock_logger)
    assert anomaly_service.read_metrics(ds) == {"AnomalyRate": 0.10}

  def test_write_failure_does_not_invalidate_training(self, tmp_path, mock_logger):
    """The model is already saved and usable; the summary is only a supplement."""
    ds = _Dataset(tmp_path / "no" / "such" / "dir")     # parent does not exist
    anomaly_service._save_metrics(ds, {"AnomalyRate": 0.05}, mock_logger)
    assert mock_logger.warning.called

  def test_contents_are_valid_json(self, tmp_path, mock_logger):
    ds = _Dataset(tmp_path)
    anomaly_service._save_metrics(ds, {"Threshold": None, "AnomalyCount": 0}, mock_logger)
    json.loads((tmp_path / "anomaly_metrics.json").read_text())


class TestAutoAnomalyPersists:
  def test_train_one_result_is_no_longer_discarded(self, tmp_path, monkeypatch):
    """Regression guard: the old version called `Anomaly.train_one(...)` without
    capturing its result, so the metrics evaporated."""
    summary = {"AnomalyCount": 3, "AnomalyRate": 0.03}
    monkeypatch.setattr(anomaly_service.Anomaly, "train_one",
                        staticmethod(lambda req, logger: SimpleNamespace(evaluation=summary)))
    monkeypatch.setattr(anomaly_service.WorkerManager, "is_active", staticmethod(lambda n: True))

    ds = SimpleNamespace(
        path=tmp_path, name="Regression-ano",
        to_anomaly_train_request=lambda fraction, algorithm: None)
    payload = SimpleNamespace(dataset_name="Regression-ano", fraction=0.05,
                              algorithm="iforest")

    anomaly_service.auto_anomaly(payload, ds)
    assert anomaly_service.read_metrics(ds) == summary
