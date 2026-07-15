"""
Tests for app.routes.modelML (service functions) and the
model_router/worker_router HTTP endpoints (via the `client` fixture).

WorkerManager always goes through mocks here - no real pm2/subprocess
calls. Real `Logger(...)` calls (which write files under the repo's cwd,
independent of Config.dir) are replaced with the shared mock_logger.
"""

import os
import sys
import datetime
import pytest
from unittest.mock import Mock
from fastapi import HTTPException
from sqlmodel import SQLModel, create_engine
from sqlalchemy.pool import StaticPool

from app.database.orm import Dataset, ModelML
from app.database.schemas import TaskType, StatusProcess
from app.config import Config
from app.routes import modelML as modelml_service
from app.core.supervised import Supervised
from app.core.anomaly import Anomaly
from app.workers.manager import WorkerManager
from tests.core.utils import load_tagnames

# app/routes/__init__.py does `router = main_router`, shadowing the
# app.routes.router submodule as a package attribute; reach the real
# submodule via sys.modules (same trick as test_dataset_routes.py).
router_module = sys.modules["app.routes.router"]

# Real tagnames from tests/tagname.csv instead of made-up names.
_TAGNAMES = load_tagnames()
FEATURES = _TAGNAMES[0:2]  # CRAH-2DH2.1/2-SUPPLY_AIR_TEMP
TARGET = _TAGNAMES[5]  # CRAH-2DH2.1-RETURN_AIR_TEMP


@pytest.fixture
def db_engine():
  """Function-scoped override: get_stats() aggregates over *all* rows in
  the table, so it needs a database no other test in this module has
  written to (the root conftest's db_engine is module-scoped)."""
  engine = create_engine(
      "sqlite:///:memory:", connect_args={"check_same_thread": False},
      poolclass=StaticPool,
  )
  SQLModel.metadata.create_all(engine)
  yield engine
  SQLModel.metadata.drop_all(engine)


def make_dataset(db_session, name, task_type=TaskType.Regression, is_valid=True,
                  features=FEATURES, target=TARGET, top_model=None,
                  meta=None, n_models=2, status=StatusProcess.SUCCESS_PULL):
  ds = Dataset(
      name=name, task_type=task_type, description="", features=list(features),
      target=target, status=status, top_model=top_model,
      start_date="20240101", end_date="20240131", time_start="00:00:00",
      time_end="23:59:00", interval=5, preprocessing=None, is_valid=is_valid,
      meta=meta if meta is not None else {"created_by": "x"}, n_models=n_models,
  )
  db_session.add(ds)
  db_session.commit()
  db_session.refresh(ds)
  return ds


def make_model(db_session, dataset, algorithm="lr", is_active=True,
               status=StatusProcess.SUCCESS_TRAIN, path=None,
               evaluation=None, meta=None):
  model = ModelML(
      dataset_id=dataset.id, name=f"{algorithm}-model", algorithm=algorithm,
      is_active=is_active, evaluation=evaluation or {"R2": 0.9}, status=status,
      path=path or f"storages/{dataset.name}/top_model/{algorithm}",
      meta=meta if meta is not None else {"size_of": 1, "created_at": datetime.datetime.now().isoformat()},
  )
  db_session.add(model)
  db_session.commit()
  db_session.refresh(model)
  return model


@pytest.fixture(autouse=True)
def _patch_logger(monkeypatch, mock_logger):
  """app.routes.modelML.Logger writes real files under cwd - always
  replace it so these tests never touch the filesystem outside tmp_path."""
  monkeypatch.setattr(modelml_service, "Logger", lambda name: mock_logger)


# =============================================================================
# check_integrity_model
# =============================================================================


class TestCheckIntegrityModel:
  def test_none_model_raises_404(self):
    with pytest.raises(HTTPException) as exc:
      modelml_service.check_integrity_model(None)
    assert exc.value.status_code == 404

  def test_inactive_model_raises(self, db_session):
    dataset = make_dataset(db_session, "Regression-inactivemodel")
    model = make_model(db_session, dataset, is_active=False)
    with pytest.raises(HTTPException) as exc:
      modelml_service.check_integrity_model(model)
    assert exc.value.status_code == 500

  def test_not_trained_status_raises(self, db_session):
    dataset = make_dataset(db_session, "Regression-untrainedmodel")
    model = make_model(db_session, dataset, status=StatusProcess.RUNNING_TRAIN)
    with pytest.raises(HTTPException):
      modelml_service.check_integrity_model(model)

  def test_missing_file_raises(self, db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "dir", tmp_path)
    dataset = make_dataset(db_session, "Regression-missingfile")
    model = make_model(db_session, dataset, path="storages/x/top_model/lr")
    with pytest.raises(HTTPException):
      modelml_service.check_integrity_model(model)

  def test_valid_model_does_not_raise(self, db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "dir", tmp_path)
    dataset = make_dataset(db_session, "Regression-validmodel")
    model_dir = tmp_path / "storages" / dataset.name / "top_model"
    model_dir.mkdir(parents=True)
    (model_dir / "lr.pkl").write_bytes(b"x")
    model = make_model(db_session, dataset, path=str(model_dir / "lr"))
    modelml_service.check_integrity_model(model)  # should not raise


# =============================================================================
# check_dataset_pretrained
# =============================================================================


class TestCheckDatasetPretrained:
  def test_no_top_model_and_no_models_raises_404(self, db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "dir", tmp_path)
    dataset = make_dataset(db_session, "Regression-notrained")
    with pytest.raises(HTTPException) as exc:
      modelml_service.check_dataset_pretrained(dataset)
    assert exc.value.status_code == 404

  def test_passes_when_models_present_and_valid(self, db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "dir", tmp_path)
    dataset = make_dataset(db_session, "Regression-pretrained")
    model_dir = tmp_path / "storages" / dataset.name / "top_model"
    model_dir.mkdir(parents=True)
    (model_dir / "lr.pkl").write_bytes(b"x")
    make_model(db_session, dataset, path=str(model_dir / "lr"))
    db_session.refresh(dataset)

    modelml_service.check_dataset_pretrained(dataset)  # should not raise


# =============================================================================
# train_model
# =============================================================================


class TestTrainModel:
  def test_invalid_algorithm_raises(self, db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "dir", tmp_path)
    dataset = make_dataset(db_session, "Regression-badalgo")
    with pytest.raises(HTTPException) as exc:
      modelml_service.train_model(dataset, "not-a-real-algorithm", db_session)
    assert exc.value.status_code == 500

  def test_valid_algorithm_trains_and_sets_top_model(self, db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "dir", tmp_path)
    dataset = make_dataset(db_session, "Regression-traingood")

    def fake_train(ds, algorithm, logger):
      ds.models.append(
          ModelML(
              dataset_id=ds.id, name=f"{algorithm}-trained", algorithm=algorithm,
              is_active=True, evaluation={}, status=StatusProcess.SUCCESS_TRAIN,
              path="x", meta={},
          )
      )

    monkeypatch.setattr(Supervised, "train", staticmethod(fake_train))

    modelml_service.train_model(dataset, "lr", db_session, use_this=False)

    assert dataset.top_model == "lr-trained"


# =============================================================================
# change_top_model
# =============================================================================


class TestChangeTopModel:
  def test_algorithm_not_found_raises_404(self, db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "dir", tmp_path)
    dataset = make_dataset(db_session, "Regression-changetopmissing")
    model_dir = tmp_path / "storages" / dataset.name / "top_model"
    model_dir.mkdir(parents=True)
    (model_dir / "lr.pkl").write_bytes(b"x")
    make_model(db_session, dataset, algorithm="lr", path=str(model_dir / "lr"))
    db_session.refresh(dataset)

    with pytest.raises(HTTPException) as exc:
      modelml_service.change_top_model(dataset, "rf")
    assert exc.value.status_code == 404

  def test_switches_to_matching_algorithm(self, db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "dir", tmp_path)
    dataset = make_dataset(db_session, "Regression-changetopok")
    model_dir = tmp_path / "storages" / dataset.name / "top_model"
    model_dir.mkdir(parents=True)
    (model_dir / "lr.pkl").write_bytes(b"x")
    make_model(db_session, dataset, algorithm="lr", path=str(model_dir / "lr"))
    db_session.refresh(dataset)

    modelml_service.change_top_model(dataset, "lr")

    assert dataset.top_model == "lr-model"


# =============================================================================
# inference (dispatch)
# =============================================================================


class TestInferenceDispatch:
  def test_dataset_not_found_raises_404(self, db_session):
    from app.database.schemas import InferenceRequestSchema
    payload = InferenceRequestSchema(dataset_name="does-not-exist", X={FEATURES[0]: [1.0]})
    with pytest.raises(HTTPException) as exc:
      modelml_service.inference(payload, db_session)
    assert exc.value.status_code == 404

  def test_untrained_dataset_raises(self, db_session, tmp_path, monkeypatch):
    from app.database.schemas import InferenceRequestSchema
    monkeypatch.setattr(Config, "dir", tmp_path)
    dataset = make_dataset(db_session, "Regression-inferuntrained")
    payload = InferenceRequestSchema(dataset_name=dataset.name, X={FEATURES[0]: [1.0]})
    with pytest.raises(HTTPException):
      modelml_service.inference(payload, db_session)

  def test_valid_dispatches_to_core_inference(self, db_session, tmp_path, monkeypatch):
    from app.database.schemas import InferenceRequestSchema
    monkeypatch.setattr(Config, "dir", tmp_path)
    dataset = make_dataset(db_session, "Regression-infergood")
    model_dir = tmp_path / "storages" / dataset.name / "top_model"
    model_dir.mkdir(parents=True)
    (model_dir / "lr.pkl").write_bytes(b"x")
    make_model(db_session, dataset, algorithm="lr", path=str(model_dir / "lr"))
    db_session.refresh(dataset)

    captured = {}

    def fake_inference(payload, ds, logger):
      captured["X"] = payload.X
      return "sentinel"

    monkeypatch.setattr(Supervised, "inference", staticmethod(fake_inference))

    payload = InferenceRequestSchema(dataset_name=dataset.name, X={FEATURES[0]: [1.0]})
    result = modelml_service.inference(payload, db_session)

    assert result == "sentinel"
    assert captured["X"] == {FEATURES[0]: [1.0]}


# =============================================================================
# auto_anomaly
# =============================================================================


class TestAutoAnomaly:
  def test_trains_and_creates_worker_when_not_active(self, db_session, tmp_path, monkeypatch):
    from app.database.schemas import AnomalyRequestSchema
    monkeypatch.setattr(Config, "dir", tmp_path)
    dataset = make_dataset(db_session, "Anomaly-autotrain", task_type=TaskType.Anomaly)

    train_calls = []
    monkeypatch.setattr(
        Anomaly, "train", staticmethod(lambda ds, algo, frac, logger: train_calls.append((algo, frac)))
    )
    monkeypatch.setattr(
        "app.routes.modelML.WorkerManager.is_active", staticmethod(lambda name: False)
    )
    create_calls = []
    monkeypatch.setattr(
        "app.routes.modelML.WorkerManager.create",
        staticmethod(lambda name, tt: create_calls.append((name, tt))),
    )

    payload = AnomalyRequestSchema(dataset_name=dataset.name, algorithm="iforest", fraction=0.1)
    modelml_service.auto_anomaly(payload, dataset)

    assert train_calls == [("iforest", 0.1)]
    assert create_calls == [(dataset.name, TaskType.Anomaly)]

  def test_skips_create_when_already_active(self, db_session, tmp_path, monkeypatch):
    from app.database.schemas import AnomalyRequestSchema
    monkeypatch.setattr(Config, "dir", tmp_path)
    dataset = make_dataset(db_session, "Anomaly-alreadyactive", task_type=TaskType.Anomaly)

    monkeypatch.setattr(Anomaly, "train", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(
        "app.routes.modelML.WorkerManager.is_active", staticmethod(lambda name: True)
    )
    create_calls = []
    monkeypatch.setattr(
        "app.routes.modelML.WorkerManager.create",
        staticmethod(lambda name, tt: create_calls.append((name, tt))),
    )

    payload = AnomalyRequestSchema(dataset_name=dataset.name, algorithm="iforest", fraction=0.1)
    result = modelml_service.auto_anomaly(payload, dataset)

    assert create_calls == []
    assert result["detail"].endswith("is still active")


# =============================================================================
# clean_models / clean_results
# =============================================================================


class TestCleanModels:
  def test_removes_models_and_resets_dataset(self, db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "dir", tmp_path)
    dataset = make_dataset(db_session, "Regression-cleanmodels", top_model="lr-model")
    model_dir = tmp_path / "storages" / dataset.name / "top_model"
    model_dir.mkdir(parents=True)
    (model_dir / "lr.pkl").write_bytes(b"x")
    (tmp_path / "storages" / dataset.name / "results").mkdir(parents=True)
    make_model(db_session, dataset, algorithm="lr", path=str(model_dir / "lr"))
    db_session.refresh(dataset)

    count = modelml_service.clean_models(dataset, db_session)

    assert count == 1
    assert dataset.top_model == ""
    assert dataset.status == StatusProcess.SUCCESS_PULL
    assert dataset.models == []


class TestCleanResults:
  def test_recreates_empty_results_dir(self, db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "dir", tmp_path)
    dataset = make_dataset(db_session, "Regression-cleanresults")
    results_dir = tmp_path / "storages" / dataset.name / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "leftover.bin").write_bytes(b"x")

    modelml_service.clean_results(dataset)

    assert results_dir.exists()
    assert list(results_dir.iterdir()) == []


# =============================================================================
# get_stats
# =============================================================================


class TestGetStats:
  def test_shape_and_aggregation(self, db_session, monkeypatch):
    monkeypatch.setattr("app.routes.modelML.get_session", lambda: iter([db_session]))
    monkeypatch.setattr(
        "app.routes.modelML.WorkerManager.get_tasks", staticmethod(lambda: [1, 2])
    )

    now_iso = datetime.datetime.now().isoformat()
    dataset = make_dataset(
        db_session, "Regression-stats", status=StatusProcess.IDLE,
        meta={"created_by": "x", "created_at": now_iso},
    )
    make_model(
        db_session, dataset, algorithm="lr", evaluation={"MAPE": 0.2},
        meta={"size_of": 1, "created_at": now_iso},
    )
    db_session.refresh(dataset)

    stats = modelml_service.get_stats()

    assert stats["total_dataset"] == 1
    assert stats["total_model"] == 1
    assert stats["total_workers"] == 2
    assert stats["avg_accuracy"] == pytest.approx(0.8)
    assert len(stats["recent_activity"]) >= 1


# =============================================================================
# HTTP endpoints: worker_router (WorkerManager fully mocked)
# =============================================================================


class TestWorkerRouterEndpoints:
  """WorkerManager is patched directly on the shared class object (it's
  the same class instance app.routes.router imports), which sidesteps
  the app.routes.router module-shadowing issue entirely."""

  def test_get_tasks(self, client, monkeypatch):
    monkeypatch.setattr(WorkerManager, "get_tasks", staticmethod(lambda name=None: [{"name": "x"}]))
    resp = client.get("/workers/tasks")
    assert resp.status_code == 200
    assert resp.json() == [{"name": "x"}]

  def test_start_missing_task_returns_404(self, client, monkeypatch):
    monkeypatch.setattr(WorkerManager, "get_tasks", staticmethod(lambda name=None: []))
    resp = client.post("/workers/start", params={"task_name": "missing"})
    assert resp.status_code == 404

  def test_start_already_active_short_circuits(self, client, monkeypatch):
    monkeypatch.setattr(
        WorkerManager, "get_tasks", staticmethod(lambda name=None: {"status": "online"})
    )
    monkeypatch.setattr(WorkerManager, "is_active", staticmethod(lambda name: True))
    start_calls = []
    monkeypatch.setattr(
        WorkerManager, "start", staticmethod(lambda name: start_calls.append(name))
    )
    resp = client.post("/workers/start", params={"task_name": "x-Regression"})
    assert resp.status_code == 200
    assert "still Active" in resp.json()["detail"]
    assert start_calls == []

  def test_stop_missing_task_returns_404(self, client, monkeypatch):
    monkeypatch.setattr(WorkerManager, "get_tasks", staticmethod(lambda name=None: []))
    resp = client.post("/workers/stop", params={"task_name": "missing"})
    assert resp.status_code == 404

  def test_delete_missing_task_returns_404(self, client, monkeypatch):
    monkeypatch.setattr(WorkerManager, "get_tasks", staticmethod(lambda name=None: []))
    resp = client.delete("/workers/delete", params={"task_name": "missing"})
    assert resp.status_code == 404

  def test_delete_existing_task(self, client, monkeypatch):
    monkeypatch.setattr(WorkerManager, "get_tasks", staticmethod(lambda name=None: {"name": "x"}))
    delete_calls = []
    monkeypatch.setattr(
        WorkerManager, "delete", staticmethod(lambda name: delete_calls.append(name))
    )
    resp = client.delete("/workers/delete", params={"task_name": "x"})
    assert resp.status_code == 200
    assert delete_calls == ["x"]


# =============================================================================
# HTTP endpoints: model_router (subset)
# =============================================================================


class TestModelRouterEndpoints:
  def test_list_all_models_empty(self, client):
    resp = client.get("/models")
    assert resp.status_code == 200
    assert resp.json() == []

  def test_get_model_missing_returns_404(self, client):
    resp = client.get("/models/does-not-exist")
    assert resp.status_code == 404

  def test_algorithms_for_valid_task_type(self, client):
    resp = client.get("/models/algorithms/Regression")
    assert resp.status_code == 200
    assert "lr" in resp.json()

  def test_algorithms_for_invalid_task_type(self, client):
    resp = client.get("/models/algorithms/NotAType")
    assert resp.status_code == 422

  def test_train_missing_dataset_returns_404(self, client):
    resp = client.post(
        "/models/train",
        json={"dataset_name": "does-not-exist", "algorithm": "lr", "use_this": False},
    )
    assert resp.status_code == 404

  def test_history_missing_dataset_returns_404(self, client):
    # Regression guard: the endpoint's bare `except Exception` used to
    # swallow the HTTPException(404) from get_dataset_or_404 and re-raise
    # it as a 500.
    resp = client.get("/models/does-not-exist/history")
    assert resp.status_code == 404

  def test_history_queries_influx_for_non_clustering(
      self, client, monkeypatch, tmp_path
  ):
    monkeypatch.setattr(Config, "dir", tmp_path)
    created = client.post("/datasets", json={
        "description": "test", "task_type": "RegressionDummy",
        "features": list(FEATURES), "target": TARGET,
        "start_date": "20240101", "end_date": "20240131",
    })
    assert created.status_code == 200
    name = created.json()["name"]

    mock_influx = Mock()
    mock_influx.query_inference.return_value = [
        {"dt": "2024-01-01T00:00:00Z", "lr": 1.0}
    ]
    monkeypatch.setattr(router_module, "get_influx_storage", lambda: mock_influx)

    resp = client.get(f"/models/{name}/history")

    assert resp.status_code == 200
    assert resp.json() == [{"dt": "2024-01-01T00:00:00Z", "lr": 1.0}]
    assert mock_influx.query_inference.call_args.kwargs == {
        "dataset_name": name, "task_type": "RegressionDummy",
    }


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
