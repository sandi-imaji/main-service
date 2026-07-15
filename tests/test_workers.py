"""
Unit tests for app.workers.manager (WorkerManager, pm2 subprocess wrapper)
and app.workers.state (WorkerStateManager, in-memory async state).

subprocess calls are always mocked - these tests never touch a real pm2
daemon.
"""

import subprocess
import json
import pytest
from contextlib import contextmanager
from unittest.mock import Mock

from app.workers.manager import WorkerManager, _run
from app.workers.state import WorkerState, WorkerStateManager, WorkerStatus
from app.database.schemas import TaskType, StatusProcess
from app import exceptions
from tests.core.utils import load_tagnames

# Real tagnames from tests/tagname.csv instead of made-up names.
_TAGNAMES = load_tagnames()
FEATURE = _TAGNAMES[0]  # CRAH-2DH2.1-SUPPLY_AIR_TEMP
TARGET = _TAGNAMES[5]  # CRAH-2DH2.1-RETURN_AIR_TEMP


# =============================================================================
# _run() subprocess wrapper
# =============================================================================


class TestRunHelper:
  def test_returns_stdout_on_success(self, monkeypatch):
    completed = subprocess.CompletedProcess(args=["x"], returncode=0, stdout="ok\n")
    monkeypatch.setattr(
        "app.workers.manager.subprocess.run", Mock(return_value=completed)
    )
    assert _run(["pm2", "list"]) == "ok\n"

  def test_called_process_error_raises_invalid_state(self, monkeypatch):
    def raise_err(*a, **k):
      raise subprocess.CalledProcessError(1, "pm2")

    monkeypatch.setattr("app.workers.manager.subprocess.run", raise_err)
    with pytest.raises(exceptions.InvalidStateException):
      _run(["pm2", "start", "x"])

  def test_file_not_found_raises_not_found(self, monkeypatch):
    def raise_err(*a, **k):
      raise FileNotFoundError()

    monkeypatch.setattr("app.workers.manager.subprocess.run", raise_err)
    with pytest.raises(exceptions.NotFoundException):
      _run(["pm2", "start", "x"])


# =============================================================================
# get_script_path
# =============================================================================


class TestGetScriptPath:
  def test_supervised_script(self):
    assert WorkerManager.get_script_path(TaskType.Regression).name == "supervised.py"
    assert WorkerManager.get_script_path(TaskType.Classification).name == "supervised.py"

  def test_unsupervised_script(self):
    assert WorkerManager.get_script_path(TaskType.Clustering).name == "unsupervised.py"

  def test_timeseries_script(self):
    assert WorkerManager.get_script_path(TaskType.TimeSeries).name == "time_series.py"

  def test_anomaly_falls_back_to_anomaly_script(self):
    assert WorkerManager.get_script_path(TaskType.Anomaly).name == "anomaly.py"


# =============================================================================
# get_tasks (pm2 jlist parsing)
# =============================================================================


PM2_JLIST = [
    {
        "name": "Regression-abcd1234-Regression",
        "pm_id": 0,
        "pid": 111,
        "pm2_env": {"status": "online", "restart_time": 0, "exec_mode": "fork", "created_at": 123},
        "monit": {"memory": 1000, "cpu": 1},
    },
    {
        "name": "Clustering-xyz789-Clustering",
        "pm_id": 1,
        "pid": 222,
        "pm2_env": {"status": "stopped", "restart_time": 2, "exec_mode": "fork", "created_at": 456},
        "monit": {"memory": 2000, "cpu": 0},
    },
    {
        "name": "main-service",
        "pm_id": 2,
        "pid": 333,
        "pm2_env": {"status": "online", "restart_time": 0, "exec_mode": "fork", "created_at": 789},
        "monit": {"memory": 3000, "cpu": 2},
    },
]


class TestGetTasks:
  def _mock_jlist(self, monkeypatch, data=PM2_JLIST):
    completed = subprocess.CompletedProcess(
        args=["pm2", "jlist"], returncode=0, stdout=json.dumps(data)
    )
    monkeypatch.setattr(
        "app.workers.manager.subprocess.run", Mock(return_value=completed)
    )

  def test_returns_cleaned_list_excluding_reserved_names(self, monkeypatch):
    self._mock_jlist(monkeypatch)
    tasks = WorkerManager.get_tasks()
    names = [t["name"] for t in tasks]
    assert "main-service" not in names
    assert "Regression-abcd1234-Regression" in names
    assert "Clustering-xyz789-Clustering" in names

  def test_parses_dataset_name_and_task_type(self, monkeypatch):
    self._mock_jlist(monkeypatch)
    tasks = WorkerManager.get_tasks()
    reg_task = next(t for t in tasks if t["name"] == "Regression-abcd1234-Regression")
    assert reg_task["dataset_name"] == "Regression-abcd1234"
    assert reg_task["task_type"] == "Regression"
    assert reg_task["status"] == "online"
    assert reg_task["restarts"] == 0

  def test_lookup_by_task_name_returns_single_dict(self, monkeypatch):
    self._mock_jlist(monkeypatch)
    task = WorkerManager.get_tasks("Regression-abcd1234-Regression")
    assert isinstance(task, dict)
    assert task["name"] == "Regression-abcd1234-Regression"

  def test_lookup_by_missing_task_name_returns_empty_list(self, monkeypatch):
    self._mock_jlist(monkeypatch)
    assert WorkerManager.get_tasks("does-not-exist") == []


# =============================================================================
# is_active
# =============================================================================


class TestIsActive:
  def test_true_when_status_online(self, monkeypatch):
    monkeypatch.setattr(
        WorkerManager, "get_tasks", staticmethod(lambda name=None: {"status": "online"})
    )
    assert WorkerManager.is_active("x") is True

  def test_false_when_status_not_online(self, monkeypatch):
    monkeypatch.setattr(
        WorkerManager, "get_tasks", staticmethod(lambda name=None: {"status": "stopped"})
    )
    assert WorkerManager.is_active("x") is False

  def test_false_when_task_not_found(self, monkeypatch):
    monkeypatch.setattr(WorkerManager, "get_tasks", staticmethod(lambda name=None: []))
    assert WorkerManager.is_active("x") is False


# =============================================================================
# update_flag_dataset (touches the DB, isolated via monkeypatched get_session)
# =============================================================================


class TestUpdateFlagDataset:
  def _make_dataset(self, db_session, name="Regression-flagtest"):
    from app.database.orm import Dataset
    dataset = Dataset(
        name=name, task_type=TaskType.Regression, description="",
        features=[FEATURE], target=TARGET, status=StatusProcess.IDLE,
        top_model=None, start_date="20240101", end_date="20240131",
        time_start="00:00:00", time_end="23:59:00", interval=5,
        preprocessing=None, is_valid=True, meta={}, n_models=1,
    )
    db_session.add(dataset)
    db_session.commit()
    db_session.refresh(dataset)
    return dataset

  def _patch_session(self, monkeypatch, db_session):
    def fake_get_session():
      yield db_session
    monkeypatch.setattr("app.workers.manager.get_session", fake_get_session)

  def test_stop_sets_status_idle(self, monkeypatch, db_session):
    from app.database.orm import Dataset
    dataset = self._make_dataset(db_session)
    dataset.status = StatusProcess.ACTIVE
    db_session.commit()
    self._patch_session(monkeypatch, db_session)

    # update_flag_dataset closes the session it's handed, so re-query
    # afterwards instead of refresh()-ing the now-detached instance.
    WorkerManager.update_flag_dataset(dataset.name, is_stop=True)

    reloaded = Dataset.get_by_name(dataset.name, db_session)
    assert reloaded.status == StatusProcess.IDLE

  def test_start_sets_status_active(self, monkeypatch, db_session):
    from app.database.orm import Dataset
    dataset = self._make_dataset(db_session, name="Regression-flagtest2")
    self._patch_session(monkeypatch, db_session)

    WorkerManager.update_flag_dataset(dataset.name, is_stop=False)

    reloaded = Dataset.get_by_name(dataset.name, db_session)
    assert reloaded.status == StatusProcess.ACTIVE

  def test_missing_dataset_is_noop(self, monkeypatch, db_session):
    self._patch_session(monkeypatch, db_session)
    # Should not raise even though the dataset doesn't exist.
    WorkerManager.update_flag_dataset("does-not-exist", is_stop=True)


# =============================================================================
# create / stop / delete command construction
# =============================================================================


class TestCreate:
  def test_builds_pm2_command_and_updates_flag(self, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "app.workers.manager._run", lambda cmd: captured.setdefault("cmd", cmd)
    )
    flag_calls = []
    monkeypatch.setattr(
        WorkerManager, "update_flag_dataset",
        staticmethod(lambda name, is_stop=False: flag_calls.append((name, is_stop))),
    )

    WorkerManager.create("Regression-abc", TaskType.Regression)

    cmd = captured["cmd"]
    assert cmd[0:2] == ["pm2", "start"]
    assert "--name" in cmd
    assert "Regression-abc-Regression" in cmd
    assert cmd[-1] == "Regression-abc"
    assert flag_calls == [("Regression-abc", False)]

  def test_anomaly_does_not_update_flag(self, monkeypatch):
    monkeypatch.setattr("app.workers.manager._run", lambda cmd: None)
    flag_calls = []
    monkeypatch.setattr(
        WorkerManager, "update_flag_dataset",
        staticmethod(lambda name, is_stop=False: flag_calls.append((name, is_stop))),
    )

    WorkerManager.create("Anomaly-abc", TaskType.Anomaly)

    assert flag_calls == []


class TestStop:
  def test_stop_updates_flag_for_non_anomaly_task(self, monkeypatch):
    monkeypatch.setattr(
        WorkerManager, "get_tasks",
        staticmethod(lambda name=None: {"dataset_name": "Regression-abc"}),
    )
    monkeypatch.setattr("app.workers.manager._run", lambda cmd: None)
    flag_calls = []
    monkeypatch.setattr(
        WorkerManager, "update_flag_dataset",
        staticmethod(lambda name, is_stop=False: flag_calls.append((name, is_stop))),
    )

    WorkerManager.stop("Regression-abc-Regression")

    assert flag_calls == [("Regression-abc", True)]

  def test_stop_skips_flag_for_anomaly_task(self, monkeypatch):
    monkeypatch.setattr(
        WorkerManager, "get_tasks",
        staticmethod(lambda name=None: {"dataset_name": "Anomaly-abc"}),
    )
    monkeypatch.setattr("app.workers.manager._run", lambda cmd: None)
    flag_calls = []
    monkeypatch.setattr(
        WorkerManager, "update_flag_dataset",
        staticmethod(lambda name, is_stop=False: flag_calls.append((name, is_stop))),
    )

    WorkerManager.stop("Anomaly-abc-Anomaly")

    assert flag_calls == []


class TestDelete:
  def test_delete_runs_pm2_delete_flushes_logs_and_updates_flag(self, monkeypatch):
    monkeypatch.setattr(
        WorkerManager, "get_tasks",
        staticmethod(lambda name=None: {"dataset_name": "Regression-abc"}),
    )
    run_calls = []
    monkeypatch.setattr(
        "app.workers.manager._run", lambda cmd: run_calls.append(cmd)
    )
    flag_calls = []
    monkeypatch.setattr(
        WorkerManager, "update_flag_dataset",
        staticmethod(lambda name, is_stop=False: flag_calls.append((name, is_stop))),
    )

    WorkerManager.delete("Regression-abc-Regression")

    assert ["pm2", "delete", "Regression-abc-Regression"] in run_calls
    assert flag_calls == [("Regression-abc", True)]


# =============================================================================
# WorkerState (dataclass)
# =============================================================================


class TestWorkerState:
  def test_post_init_sets_started_at_when_missing(self):
    state = WorkerState(dataset_name="ds", task_type="Regression")
    assert state.started_at is not None

  def test_is_running_property(self):
    state = WorkerState(dataset_name="ds", task_type="Regression", status=WorkerStatus.RUNNING)
    assert state.is_running is True
    state.status = WorkerStatus.IDLE
    assert state.is_running is False

  def test_needs_reload_property(self):
    state = WorkerState(dataset_name="ds", task_type="Regression", model_version=1, latest_model_version=2)
    assert state.needs_reload is True
    state.model_version = 2
    assert state.needs_reload is False

  def test_to_dict_shape(self):
    state = WorkerState(dataset_name="ds", task_type="Regression")
    d = state.to_dict()
    assert d["dataset_name"] == "ds"
    assert d["status"] == "idle"
    assert d["task_type"] == "Regression"
    assert "started_at" in d and "updated_at" in d


# =============================================================================
# WorkerStateManager (in-memory async state)
# =============================================================================


@pytest.fixture
def manager():
  """Fresh manager per test - never share the global singleton."""
  return WorkerStateManager()


class TestCreateWorker:
  async def test_creates_new_worker(self, manager):
    worker = await manager.create_worker("ds-1", "Regression", interval_seconds=60)
    assert worker.dataset_name == "ds-1"
    assert worker.task_type == "Regression"
    assert worker.interval_seconds == 60
    assert worker.status == WorkerStatus.IDLE

  async def test_returns_existing_worker_instead_of_duplicating(self, manager):
    first = await manager.create_worker("ds-1", "Regression")
    second = await manager.create_worker("ds-1", "Regression")
    assert first is second
    assert (await manager.get_all_workers()).__len__() == 1


class TestGetWorker:
  async def test_get_existing_worker(self, manager):
    await manager.create_worker("ds-1", "Regression")
    worker = await manager.get_worker("ds-1")
    assert worker is not None
    assert worker.dataset_name == "ds-1"

  async def test_get_missing_worker_returns_none(self, manager):
    assert await manager.get_worker("missing") is None


class TestGetAllAndRunningWorkers:
  async def test_get_all_workers_no_filter(self, manager):
    await manager.create_worker("ds-1", "Regression")
    await manager.create_worker("ds-2", "Clustering")
    workers = await manager.get_all_workers()
    assert len(workers) == 2

  async def test_get_all_workers_filtered_by_status(self, manager):
    await manager.create_worker("ds-1", "Regression")
    await manager.create_worker("ds-2", "Clustering")
    await manager.update_status("ds-1", WorkerStatus.RUNNING)

    running = await manager.get_all_workers(status=WorkerStatus.RUNNING)
    assert len(running) == 1
    assert running[0].dataset_name == "ds-1"

  async def test_get_running_workers(self, manager):
    await manager.create_worker("ds-1", "Regression")
    await manager.update_status("ds-1", WorkerStatus.RUNNING)
    running = await manager.get_running_workers()
    assert [w.dataset_name for w in running] == ["ds-1"]


class TestUpdateStatus:
  async def test_updates_status_of_existing_worker(self, manager):
    await manager.create_worker("ds-1", "Regression")
    ok = await manager.update_status("ds-1", WorkerStatus.RUNNING)
    assert ok is True
    worker = await manager.get_worker("ds-1")
    assert worker.status == WorkerStatus.RUNNING

  async def test_missing_worker_returns_false(self, manager):
    ok = await manager.update_status("missing", WorkerStatus.RUNNING)
    assert ok is False

  async def test_error_message_increments_error_counters(self, manager):
    await manager.create_worker("ds-1", "Regression")
    await manager.update_status("ds-1", WorkerStatus.ERROR, error_message="boom")
    worker = await manager.get_worker("ds-1")
    assert worker.last_error == "boom"
    assert worker.error_count == 1
    assert worker.consecutive_errors == 1

  async def test_running_status_resets_consecutive_errors(self, manager):
    await manager.create_worker("ds-1", "Regression")
    await manager.update_status("ds-1", WorkerStatus.ERROR, error_message="boom")
    await manager.update_status("ds-1", WorkerStatus.RUNNING)
    worker = await manager.get_worker("ds-1")
    assert worker.consecutive_errors == 0


class TestUpdateMetrics:
  async def test_accumulates_predictions(self, manager):
    await manager.create_worker("ds-1", "Regression")
    await manager.update_metrics("ds-1", predictions_count=5)
    await manager.update_metrics("ds-1", predictions_count=3)
    worker = await manager.get_worker("ds-1")
    assert worker.total_predictions == 8

  async def test_error_accumulates_and_resets_on_success(self, manager):
    await manager.create_worker("ds-1", "Regression")
    await manager.update_metrics("ds-1", error="failed once")
    worker = await manager.get_worker("ds-1")
    assert worker.error_count == 1
    assert worker.consecutive_errors == 1

    await manager.update_metrics("ds-1", predictions_count=1)
    worker = await manager.get_worker("ds-1")
    assert worker.consecutive_errors == 0

  async def test_missing_worker_returns_false(self, manager):
    assert await manager.update_metrics("missing", predictions_count=1) is False


class TestDeleteAndClear:
  async def test_delete_existing_worker(self, manager):
    await manager.create_worker("ds-1", "Regression")
    assert await manager.delete_worker("ds-1") is True
    assert await manager.get_worker("ds-1") is None

  async def test_delete_missing_worker_returns_false(self, manager):
    assert await manager.delete_worker("missing") is False

  async def test_clear_all_returns_count_and_empties(self, manager):
    await manager.create_worker("ds-1", "Regression")
    await manager.create_worker("ds-2", "Clustering")
    count = await manager.clear_all()
    assert count == 2
    assert await manager.get_all_workers() == []


class TestGetStats:
  async def test_stats_shape_and_counts(self, manager):
    await manager.create_worker("ds-1", "Regression")
    await manager.create_worker("ds-2", "Clustering")
    await manager.update_status("ds-1", WorkerStatus.RUNNING)

    stats = manager.get_stats()
    assert stats["total_workers"] == 2
    assert stats["by_status"]["running"] == 1
    assert stats["by_status"]["idle"] == 1


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
