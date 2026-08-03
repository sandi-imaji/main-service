"""app.services.model — dashboard stats + explicit-payload inference routing."""
from types import SimpleNamespace

import pytest

from app.database.schemas import (
    InferenceRequestSchema,
    StatusProcess,
    TaskType,
)
from app.exceptions import ValidationException
from app.helpers import DTEncoder
from app.services import model as model_service
from app.workers.manager import WorkerManager


def _model(evaluation, created_at):
  return SimpleNamespace(evaluation=evaluation, status=StatusProcess.SUCCESS_TRAIN,
                         name="m", meta=SimpleNamespace(created_at=created_at))


def _dataset(task_type, evaluation, created_at):
  return SimpleNamespace(
      status=StatusProcess.IDLE, task_type=task_type, name="ds",
      meta=SimpleNamespace(created_at=created_at),
      models=[_model(evaluation, created_at)])


# Bentuk yang BENAR-BENAR tersimpan: `MetaDataset.created_at` bertipe `str`, jadi
# fixture harus memakai string ISO. Versi lama memakai objek datetime — itulah
# sebabnya test tetap hijau sementara `/utils/stats` di produksi melempar
# `AttributeError: 'str' object has no attribute 'tzinfo'`.
def _now_iso():
  return DTEncoder.now().isoformat()


class TestBuildStats:
  def test_mixed_task_types_no_keyerror(self, monkeypatch):
    # Classification/Anomaly previously crashed _build_stats with KeyError.
    monkeypatch.setattr(WorkerManager, "get_tasks", staticmethod(lambda *a, **k: []))
    now = _now_iso()
    datasets = [
        _dataset(TaskType.Regression, {"MAPE": 0.2}, now),
        _dataset(TaskType.Classification, {"Accuracy": 0.9}, now),
        _dataset(TaskType.Anomaly, {}, now),          # no metric — must be skipped
    ]
    stats = model_service._build_stats(datasets)
    assert stats["total_dataset"] == 3
    assert 0.0 <= stats["avg_accuracy"] <= 1.0

  def test_empty(self, monkeypatch):
    monkeypatch.setattr(WorkerManager, "get_tasks", staticmethod(lambda *a, **k: []))
    stats = model_service._build_stats([])
    assert stats["avg_accuracy"] == 0
    assert stats["total_model"] == 0

  def test_created_at_string_tidak_melempar(self, monkeypatch):
    """Regression guard untuk crash `/utils/stats` di produksi."""
    monkeypatch.setattr(WorkerManager, "get_tasks", staticmethod(lambda *a, **k: []))
    stats = model_service._build_stats([_dataset(TaskType.Regression, {"MAPE": 0.2},
                                                 _now_iso())])
    hari_ini = DTEncoder.now().date().isoformat()
    assert stats["sum_data"] == {hari_ini: 1}
    # dataset dan model dibuat hari ini, jadi keduanya masuk aktivitas terbaru
    assert len(stats["recent_activity"]) == 2
    assert stats["recent_activity"][0]["time"].endswith("ago")

  def test_created_at_datetime_masih_diterima(self, monkeypatch):
    """Normalisasi tidak boleh merusak pemanggil yang sudah mengirim datetime."""
    monkeypatch.setattr(WorkerManager, "get_tasks", staticmethod(lambda *a, **k: []))
    stats = model_service._build_stats(
        [_dataset(TaskType.Regression, {"MAPE": 0.2}, DTEncoder.now())])
    assert stats["sum_data"] == {DTEncoder.now().date().isoformat(): 1}

  @pytest.mark.parametrize("rusak", ["", None, "bukan-tanggal"])
  def test_created_at_rusak_dilewati_bukan_menjatuhkan_dashboard(self, rusak, monkeypatch):
    """Satu baris meta cacat tidak boleh membuat seluruh dashboard 500."""
    monkeypatch.setattr(WorkerManager, "get_tasks", staticmethod(lambda *a, **k: []))
    stats = model_service._build_stats([_dataset(TaskType.Regression, {"MAPE": 0.2}, rusak)])
    assert stats["sum_data"] == {}
    assert stats["recent_activity"] == []
    assert stats["total_dataset"] == 1            # datasetnya sendiri tetap terhitung


class TestExplicitInference:
  def _patch_common(self, monkeypatch, dataset):
    monkeypatch.setattr(model_service.Dataset, "get_by_name",
                        staticmethod(lambda name, db: dataset))
    monkeypatch.setattr(model_service, "check_dataset_pretrained", lambda ds: None)

  def test_timeseries_rejected(self, monkeypatch):
    dataset = SimpleNamespace(task_type=TaskType.TimeSeries, name="TimeSeries-a")
    self._patch_common(monkeypatch, dataset)
    payload = InferenceRequestSchema(dataset_name="TimeSeries-a", X=3)  # fh int
    with pytest.raises(ValidationException):
      model_service.inference(payload, db=None)

  def test_clustering_delegates(self, monkeypatch):
    import app.services.clustering as clustering_service
    dataset = SimpleNamespace(task_type=TaskType.Clustering, name="Clustering-a")
    self._patch_common(monkeypatch, dataset)
    called = {}
    def _fake_inference(ds, X, logger):
      called["x"] = (ds, X)
      return "RESULT"
    monkeypatch.setattr(clustering_service, "inference", _fake_inference)
    payload = InferenceRequestSchema(dataset_name="Clustering-a", X={"x1": [1.0]})
    result = model_service.inference(payload, db=None)
    assert result == "RESULT"
    assert called["x"][1] == {"x1": [1.0]}
