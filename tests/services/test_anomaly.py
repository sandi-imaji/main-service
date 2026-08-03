"""app.services.anomaly — deteksi anomaly, training, dan loop workernya.

Anomaly menumpang dataset milik task lain (mis. Regression). Dulu satu file
worker melayani supervised DAN anomaly sekaligus, sehingga identitas worker
harus dioper lewat argv dan gampang tertukar. Sekarang file inilah identitasnya
— yang dikunci di sini adalah konsekuensinya: worker anomaly tidak boleh
menyentuh finetune milik dataset induknya, dan hasilnya harus bertanda
"Anomaly", bukan task utama dataset.
"""
from types import SimpleNamespace

import pytest

from app.database.schemas import AnomalyRequestSchema, TaskType
from app.services import anomaly as anomaly_service
from app.services import worker as worker_module
from app.services.results import AnomalyResultSchema
from tests.conftest import FakeDataset


class _StopLoop(Exception):
  """Sentinel to break the worker's infinite loop after one iteration."""


@pytest.fixture
def one_tick(monkeypatch):
  def _sleep(_seconds):
    raise _StopLoop
  monkeypatch.setattr(worker_module.time, "sleep", _sleep)


def _host_dataset(due=True):
  """Dataset yang task utamanya Regression — anomaly cuma menumpang di sini."""
  ds = FakeDataset(task_type=TaskType.Regression)
  ds.finetune_checks = 0
  def _is_time():
    ds.finetune_checks += 1
    return due
  ds.is_time_to_finetune = _is_time
  return ds


class TestWorkerLoop:
  def test_never_touches_the_host_dataset_finetune(self, monkeypatch, mock_logger, one_tick):
    monkeypatch.setattr(anomaly_service, "auto_anomaly_detection",
                        lambda ds, lg: SimpleNamespace(is_valid=False))
    ds = _host_dataset(due=True)
    with pytest.raises(_StopLoop):
      anomaly_service.auto_inference_write_loop(ds, mock_logger)
    assert ds.finetune_checks == 0

  def test_runs_anomaly_detection_not_the_primary_task(self, monkeypatch, mock_logger, one_tick):
    calls = []
    monkeypatch.setattr(anomaly_service, "auto_anomaly_detection",
                        lambda ds, lg: calls.append("anomaly") or SimpleNamespace(is_valid=False))
    with pytest.raises(_StopLoop):
      anomaly_service.auto_inference_write_loop(_host_dataset(), mock_logger)
    assert calls == ["anomaly"]


class TestAutoAnomalyDetection:
  def _dataset(self):
    ds = FakeDataset(task_type=TaskType.Anomaly, features=["x1", "x2"])
    ds.to_anomaly_predict_request = lambda features: features
    return ds

  def test_no_live_data_returns_invalid(self, monkeypatch, mock_logger):
    monkeypatch.setattr(anomaly_service, "pull_realtime", lambda cols, lg: [None, None])
    called = []
    monkeypatch.setattr(anomaly_service.Anomaly, "predict",
                        lambda req, lg: called.append(req))
    out = anomaly_service.auto_anomaly_detection(self._dataset(), mock_logger)
    assert out.is_valid is False
    assert called == []                          # never ran the model on empty input

  def test_valid_result_is_tagged_anomaly(self, monkeypatch, mock_logger):
    monkeypatch.setattr(anomaly_service, "pull_realtime", lambda cols, lg: [1.0, 2.0])
    fake_result = SimpleNamespace(features={"x1": 1.0, "x2": 2.0},
                                  is_anomaly=True, anomaly_score=0.9)
    monkeypatch.setattr(anomaly_service.Anomaly, "predict", lambda req, lg: fake_result)
    out = anomaly_service.auto_anomaly_detection(self._dataset(), mock_logger)
    assert isinstance(out, AnomalyResultSchema)
    assert out.is_valid is True
    assert out.task_type == "Anomaly"            # not the dataset's primary type


class TestAutoAnomalyTraining:
  """Guards two production bugs: the request must carry `algorithm`, and
  `Anomaly.train_one(req, logger)` takes exactly two positional args."""

  def test_train_one_called_with_request_and_logger_only(self, monkeypatch, tmp_path):
    captured = {}

    def fake_to_req(fraction, algorithm):
      captured["req_args"] = (fraction, algorithm)
      return SimpleNamespace(fraction=fraction, algorithm=algorithm)

    def fake_train_one(req, logger):        # exactly 2 positional args
      captured["train_call"] = req
      # Bentuk NYATA yang dikembalikan core: sebuah TrainedModel dengan ringkasan
      # deteksi. Fake lama mengembalikan None, yang tidak pernah terjadi di
      # produksi dan menyembunyikan bahwa hasilnya kini ikut disimpan.
      return SimpleNamespace(evaluation={"AnomalyRate": 0.05})

    dataset = SimpleNamespace(to_anomaly_train_request=fake_to_req, path=tmp_path)
    monkeypatch.setattr(anomaly_service.Anomaly, "train_one", staticmethod(fake_train_one))
    monkeypatch.setattr(anomaly_service.WorkerManager, "is_active", staticmethod(lambda n: True))

    payload = AnomalyRequestSchema(dataset_name="Anomaly-a", algorithm="iforest", fraction=0.05)
    anomaly_service.auto_anomaly(payload, dataset)

    assert captured["req_args"] == (0.05, "iforest")     # algorithm forwarded
    assert captured["train_call"].algorithm == "iforest"

  def test_worker_started_for_anomaly_task_type(self, monkeypatch, tmp_path):
    created = {}
    monkeypatch.setattr(anomaly_service.Anomaly, "train_one",
                        staticmethod(lambda req, lg: SimpleNamespace(evaluation={"AnomalyRate": 0.05})))
    monkeypatch.setattr(anomaly_service.WorkerManager, "is_active", staticmethod(lambda n: False))
    monkeypatch.setattr(anomaly_service.WorkerManager, "create",
                        staticmethod(lambda name, tt: created.update(name=name, task_type=tt)))

    dataset = SimpleNamespace(to_anomaly_train_request=lambda f, a: SimpleNamespace(),
                              path=tmp_path)
    payload = AnomalyRequestSchema(dataset_name="Regression-a", algorithm="iforest", fraction=0.05)
    anomaly_service.auto_anomaly(payload, dataset)

    # worker anomaly, bukan worker task utama dataset
    assert created == {"name": "Regression-a", "task_type": TaskType.Anomaly}
