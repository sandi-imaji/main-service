"""
Tests for app.routes.dataset (service functions) and dataset_router
(HTTP endpoints, via the `client` fixture in tests/conftest.py).

External side effects (pm2/WorkerManager subprocess calls, InfluxDB,
background pulling) are mocked; the filesystem is redirected to tmp_path
via a monkeypatched Config.dir.
"""

import sys
import pytest
from unittest.mock import Mock
from fastapi import HTTPException

from app.database.orm import Dataset, ModelML
from app.database.schemas import DatasetRequestSchema, TaskType, StatusProcess
from app.config import Config
from app.routes import dataset as dataset_service
from tests.core.utils import load_tagnames

# app/routes/__init__.py does `router = main_router`, which shadows the
# `app.routes.router` *submodule* as a package attribute. Dotted-string
# monkeypatch targets ("app.routes.router.x") resolve through that
# attribute and hit the APIRouter instance instead of the submodule, so
# reach the real submodule via sys.modules instead.
router_module = sys.modules["app.routes.router"]

# Real tagnames from tests/tagname.csv instead of made-up names, same
# convention as tests/test_initiate.py and tests/core/test_supervised.py.
_TAGNAMES = load_tagnames()
FEATURES = _TAGNAMES[0:2]  # CRAH-2DH2.1/2-SUPPLY_AIR_TEMP
TARGET = _TAGNAMES[5]  # CRAH-2DH2.1-RETURN_AIR_TEMP


def make_dataset_row(db_session, name, features=FEATURES, target=TARGET,
                      is_valid=True, meta=None, n_models=2,
                      task_type=TaskType.Regression, status=StatusProcess.SUCCESS_PULL):
  ds = Dataset(
      name=name, task_type=task_type, description="", features=list(features),
      target=target, status=status, top_model=None,
      start_date="20240101", end_date="20240131", time_start="00:00:00",
      time_end="23:59:00", interval=5, preprocessing=None, is_valid=is_valid,
      meta=meta if meta is not None else {}, n_models=n_models,
  )
  db_session.add(ds)
  db_session.commit()
  db_session.refresh(ds)
  return ds


# =============================================================================
# create_dataset (service function)
# =============================================================================


class TestCreateDataset:
  def test_creates_dataset_and_storage_dirs(self, db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "dir", tmp_path)
    payload = DatasetRequestSchema(
        task_type="Regression", features=list(FEATURES), target=TARGET,
        start_date="20240101", end_date="20240131",
    )
    dataset = dataset_service.create_dataset(payload, db_session)

    assert dataset.name.startswith("Regression-")
    assert dataset.is_valid is False
    assert dataset.status == StatusProcess.PENDING
    assert (tmp_path / "storages" / dataset.name / "top_model").exists()
    assert (tmp_path / "storages" / dataset.name / "results").exists()


# =============================================================================
# delete_dataset / clean_all_datasets
# =============================================================================


class TestDeleteDataset:
  def test_delete_missing_dataset_raises_404(self, db_session):
    with pytest.raises(HTTPException) as exc:
      dataset_service.delete_dataset("does-not-exist", db_session)
    assert exc.value.status_code == 404

  def test_delete_existing_dataset_removes_row_and_files(
      self, db_session, tmp_path, monkeypatch
  ):
    monkeypatch.setattr(Config, "dir", tmp_path)
    dataset = make_dataset_row(db_session, "Regression-deleteme")
    storage = tmp_path / "storages" / dataset.name
    storage.mkdir(parents=True)
    (storage / "data.csv").write_text("x")

    monkeypatch.setattr(
        "app.routes.dataset.WorkerManager.delete_by_dataset", Mock()
    )
    mock_influx = Mock()
    monkeypatch.setattr(
        "app.routes.dataset.get_influx_storage", lambda: mock_influx
    )

    result = dataset_service.delete_dataset("Regression-deleteme", db_session)

    assert result["dataset_name"] == "Regression-deleteme"
    assert not storage.exists()
    assert Dataset.get_by_name("Regression-deleteme", db_session) is None
    mock_influx.delete_dataset.assert_called_once_with("Regression-deleteme")

  def test_delete_all_marker_deletes_every_dataset(
      self, db_session, tmp_path, monkeypatch
  ):
    monkeypatch.setattr(Config, "dir", tmp_path)
    make_dataset_row(db_session, "Regression-all1")
    make_dataset_row(db_session, "Regression-all2")
    monkeypatch.setattr(
        "app.routes.dataset.WorkerManager.delete_by_dataset", Mock()
    )
    monkeypatch.setattr(
        "app.routes.dataset.get_influx_storage", lambda: Mock()
    )

    result = dataset_service.delete_dataset("ALL-", db_session)

    assert "Successfully removed 2" in result["detail"]
    assert Dataset.get_all(db_session) is None


class TestCleanAllDatasets:
  def test_raises_when_no_datasets(self, db_session):
    with pytest.raises(HTTPException):
      dataset_service.clean_all_datasets([], db_session)

  def test_continues_when_worker_or_influx_cleanup_fails(
      self, db_session, tmp_path, monkeypatch
  ):
    monkeypatch.setattr(Config, "dir", tmp_path)
    dataset = make_dataset_row(db_session, "Regression-cleanup-fail")

    monkeypatch.setattr(
        "app.routes.dataset.WorkerManager.delete_by_dataset",
        Mock(side_effect=RuntimeError("pm2 down")),
    )
    monkeypatch.setattr(
        "app.routes.dataset.get_influx_storage",
        lambda: (_ for _ in ()).throw(RuntimeError("influx down")),
    )

    result = dataset_service.clean_all_datasets([dataset], db_session)

    assert "Successfully removed 1" in result["detail"]
    assert Dataset.get_by_name("Regression-cleanup-fail", db_session) is None


# =============================================================================
# get_df_sample / get_df_describe / dim_reduce
# =============================================================================


class TestDataframeHelpers:
  def _dataset_with_csv(self, db_session, tmp_path, monkeypatch, name):
    monkeypatch.setattr(Config, "dir", tmp_path)
    dataset = make_dataset_row(db_session, name, meta={"created_by": "x"})
    storage = tmp_path / "storages" / dataset.name
    storage.mkdir(parents=True)
    import pandas as pd
    pd.DataFrame({FEATURES[0]: range(20), FEATURES[1]: range(20, 40)}).to_csv(
        storage / "data.csv", index=False
    )
    return dataset

  def test_get_df_sample_returns_requested_rows(self, db_session, tmp_path, monkeypatch):
    dataset = self._dataset_with_csv(db_session, tmp_path, monkeypatch, "Regression-sample")
    sample = dataset_service.get_df_sample(dataset, 5)
    assert len(sample) == 5

  def test_get_df_sample_none_dataset_raises_404(self):
    with pytest.raises(HTTPException) as exc:
      dataset_service.get_df_sample(None, 5)
    assert exc.value.status_code == 404

  def test_get_df_describe_returns_stats_dict(self, db_session, tmp_path, monkeypatch):
    dataset = self._dataset_with_csv(db_session, tmp_path, monkeypatch, "Regression-describe")
    stats = dataset_service.get_df_describe(dataset)
    assert FEATURES[0] in stats
    assert "mean" in stats[FEATURES[0]]

  def test_dim_reduce_raises_when_pca_not_processed(self, db_session, tmp_path, monkeypatch):
    dataset = self._dataset_with_csv(db_session, tmp_path, monkeypatch, "Regression-nopca")
    with pytest.raises(HTTPException) as exc:
      dataset_service.dim_reduce(dataset)
    assert exc.value.status_code == 404

  def test_dim_reduce_returns_existing_pca_meta(self, db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "dir", tmp_path)
    dataset = make_dataset_row(
        db_session, "Regression-withpca",
        meta={"created_by": "x", "pca": {"components": [1, 2], "variance": 0.9}},
    )
    result = dataset_service.dim_reduce(dataset)
    assert result == {"components": [1, 2], "variance": 0.9}


# =============================================================================
# HTTP endpoint tests (dataset_router, via TestClient)
# =============================================================================


class TestDatasetListEndpoints:
  def test_list_datasets_empty(self, client):
    resp = client.get("/datasets")
    assert resp.status_code == 200
    body = resp.json()
    assert body["recordsTotal"] == 0
    assert body["data"] == []

  def test_list_dataset_names(self, client):
    resp = client.get("/datasets/names")
    assert resp.status_code == 200
    assert resp.json() == []

  def test_active_dataset_empty(self, client):
    resp = client.get("/datasets/active")
    assert resp.status_code == 200
    assert resp.json() == []

  def test_filter_by_invalid_task_type_returns_422(self, client):
    resp = client.get("/datasets/filter/NotARealTaskType")
    assert resp.status_code == 422


class TestDatasetDetailEndpoints:
  def test_get_missing_dataset_returns_404(self, client):
    resp = client.get("/datasets/does-not-exist")
    assert resp.status_code == 404

  def test_get_dataset_status_missing_returns_404(self, client):
    resp = client.get("/datasets/does-not-exist/status")
    assert resp.status_code == 404


class TestCreateDatasetEndpoint:
  def test_create_dummy_dataset_does_not_trigger_pulling(self, client, monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "dir", tmp_path)
    pulling_calls = []
    monkeypatch.setattr(
        router_module, "pulling", lambda **kw: pulling_calls.append(kw)
    )

    payload = {
        "description": "test",
        "task_type": "RegressionDummy",
        "features": list(FEATURES),
        "target": TARGET,
        "start_date": "20240101",
        "end_date": "20240131",
    }
    resp = client.post("/datasets", json=payload)

    assert resp.status_code == 200
    assert resp.json()["name"].startswith("RegressionDummy-")
    assert pulling_calls == []

  def test_create_real_dataset_triggers_background_pulling(
      self, client, monkeypatch, tmp_path
  ):
    monkeypatch.setattr(Config, "dir", tmp_path)
    pulling_calls = []
    monkeypatch.setattr(
        router_module, "pulling", lambda **kw: pulling_calls.append(kw)
    )

    payload = {
        "description": "test",
        "task_type": "Regression",
        "features": list(FEATURES),
        "target": TARGET,
        "start_date": "20240101",
        "end_date": "20240131",
    }
    resp = client.post("/datasets", json=payload)

    assert resp.status_code == 200
    assert len(pulling_calls) == 1
    assert pulling_calls[0]["dataset_name"] == resp.json()["name"]


class TestDeleteDatasetEndpoint:
  def test_delete_missing_dataset_returns_404(self, client):
    resp = client.delete("/datasets/does-not-exist")
    assert resp.status_code == 404

  def test_delete_by_invalid_task_type_returns_empty_result(self, client, monkeypatch):
    # get_by_task_type with a bogus string just yields no rows; the route
    # forwards straight to clean_all_datasets, which 400s on an empty list.
    resp = client.delete("/datasets/filter/NotARealTaskType")
    assert resp.status_code == 400


class TestSampleDescribePcaEndpoints:
  def test_sample_missing_dataset_returns_404(self, client):
    resp = client.get("/datasets/does-not-exist/sample")
    assert resp.status_code == 404

  def test_describe_missing_dataset_returns_404(self, client):
    resp = client.get("/datasets/does-not-exist/describe")
    assert resp.status_code == 404

  def test_pca_missing_dataset_returns_404(self, client):
    resp = client.get("/datasets/does-not-exist/pca")
    assert resp.status_code == 404


class TestUtilEndpoints:
  def test_list_task_types(self, client):
    resp = client.get("/datasets/utils/task-types")
    assert resp.status_code == 200
    print(resp.text)
    assert "Regression" in resp.json()

  def test_search_tagname_delegates_to_query_tagnames(self, client, monkeypatch):
    monkeypatch.setattr(
        router_module, "query_tagnames", lambda q: [f"tag-{q}"]
    )
    resp = client.get("/datasets/utils/tagname", params={"query": "temp"})
    assert resp.status_code == 200
    assert resp.json() == ["tag-temp"]


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
