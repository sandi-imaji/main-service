"""app.services.supervised — inference supervised + loop workernya.

Loop `while True` diputus dengan memonkeypatch `time.sleep` supaya melempar
sentinel: posisinya di luar try/except bagian dalam, jadi loop keluar bersih
setelah tepat satu tick.
"""
from types import SimpleNamespace

import pytest

from app.database.schemas import TaskType
from app.services import supervised as supervised_service
from app.services import worker as worker_module
from tests.conftest import FakeDataset


class _StopLoop(Exception):
  """Sentinel to break the worker's infinite loop after one iteration."""


@pytest.fixture
def one_tick(monkeypatch):
  """Make the loop run exactly one iteration then raise _StopLoop."""
  def _sleep(_seconds):
    raise _StopLoop
  monkeypatch.setattr(worker_module.time, "sleep", _sleep)


def _loop_dataset(due=False):
  ds = FakeDataset(task_type=TaskType.Regression)
  ds.finetune_checks = 0
  def _is_time():
    ds.finetune_checks += 1
    return due
  ds.is_time_to_finetune = _is_time
  return ds


class TestWorkerLoop:
  def test_runs_supervised_inference(self, monkeypatch, mock_logger, one_tick):
    calls = []
    monkeypatch.setattr(supervised_service, "auto_predictions",
                        lambda ds, lg: calls.append("supervised") or SimpleNamespace(is_valid=False))
    with pytest.raises(_StopLoop):
      supervised_service.auto_inference_write_loop(_loop_dataset(), mock_logger)
    assert calls == ["supervised"]

  def test_checks_finetune_every_tick(self, monkeypatch, mock_logger, one_tick):
    monkeypatch.setattr(supervised_service, "auto_predictions",
                        lambda ds, lg: SimpleNamespace(is_valid=False))
    ds = _loop_dataset(due=False)
    with pytest.raises(_StopLoop):
      supervised_service.auto_inference_write_loop(ds, mock_logger)
    assert ds.finetune_checks == 1

  def test_due_dataset_is_finetuned_then_marker_synced(self, monkeypatch, mock_logger, one_tick):
    order = []
    monkeypatch.setattr(supervised_service, "auto_predictions",
                        lambda ds, lg: order.append("infer") or SimpleNamespace(is_valid=False))
    monkeypatch.setattr("app.services.train.finetune",
                        lambda ds, lg: order.append("finetune"))
    monkeypatch.setattr(supervised_service, "_sync_finetune_marker",
                        lambda ds, lg: order.append("sync"))

    with pytest.raises(_StopLoop):
      supervised_service.auto_inference_write_loop(_loop_dataset(due=True), mock_logger)

    # finetune harus selesai (dan penandanya disegarkan) SEBELUM inference tick ini
    assert order == ["finetune", "sync", "infer"]

  def test_not_due_skips_finetune(self, monkeypatch, mock_logger, one_tick):
    calls = []
    monkeypatch.setattr(supervised_service, "auto_predictions",
                        lambda ds, lg: SimpleNamespace(is_valid=False))
    monkeypatch.setattr("app.services.train.finetune",
                        lambda ds, lg: calls.append("finetune"))
    with pytest.raises(_StopLoop):
      supervised_service.auto_inference_write_loop(_loop_dataset(due=False), mock_logger)
    assert calls == []


class TestLoopInfluxWrite:
  def test_valid_result_is_written(self, monkeypatch, mock_logger, one_tick):
    written = []
    result = SimpleNamespace(is_valid=True,
                             write_to_influx=lambda influx, lg: written.append(influx))
    monkeypatch.setattr(supervised_service, "auto_predictions", lambda ds, lg: result)
    monkeypatch.setattr(worker_module, "get_influx_storage", lambda: "INFLUX")
    with pytest.raises(_StopLoop):
      supervised_service.auto_inference_write_loop(_loop_dataset(), mock_logger)
    assert written == ["INFLUX"]

  def test_invalid_result_is_not_written(self, monkeypatch, mock_logger, one_tick):
    calls = []
    monkeypatch.setattr(supervised_service, "auto_predictions",
                        lambda ds, lg: SimpleNamespace(is_valid=False))
    monkeypatch.setattr(worker_module, "get_influx_storage", lambda: calls.append("influx"))
    with pytest.raises(_StopLoop):
      supervised_service.auto_inference_write_loop(_loop_dataset(), mock_logger)
    assert calls == []

  def test_tick_failure_does_not_kill_the_loop(self, monkeypatch, mock_logger):
    """Satu tick gagal harus dicatat lalu loop lanjut — bukan mematikan worker."""
    ticks = {"n": 0}
    def _boom(ds, lg):
      ticks["n"] += 1
      raise RuntimeError("inference meledak")
    def _sleep(_seconds):
      if ticks["n"] >= 2: raise _StopLoop      # biarkan dua tick lewat dulu
    monkeypatch.setattr(supervised_service, "auto_predictions", _boom)
    monkeypatch.setattr(worker_module.time, "sleep", _sleep)

    with pytest.raises(_StopLoop):
      supervised_service.auto_inference_write_loop(_loop_dataset(), mock_logger)

    assert ticks["n"] == 2                     # tick kedua tetap jalan
    mock_logger.error.assert_called()


class TestAutoPredictionsGuard:
  def test_no_live_data_returns_invalid(self, monkeypatch, mock_logger):
    monkeypatch.setattr(supervised_service.Config, "debug_mode", False)
    monkeypatch.setattr(supervised_service, "pull_realtime", lambda cols, lg: [0, 0, 0])
    ds = FakeDataset(task_type=TaskType.Regression, features=["x1", "x2"], target="y")
    out = supervised_service.auto_predictions(ds, mock_logger)
    assert out.is_valid is False
