"""
Unit tests for app.database.orm (Dataset, ModelML), using an isolated
in-memory SQLite session (see tests/conftest.py: db_engine/db_session).
"""

import datetime
import pytest
from fastapi import HTTPException
from sqlmodel import SQLModel, create_engine
from sqlalchemy.pool import StaticPool

from app.database.orm import Dataset, ModelML
from app.database.schemas import TaskType, StatusProcess
from app.config import Config
from tests.core.utils import load_tagnames

# Use real tagnames from tests/tagname.csv instead of made-up names, same
# convention as tests/test_initiate.py and tests/core/test_supervised.py.
_TAGNAMES = load_tagnames()
FEATURES = _TAGNAMES[0:2]  # CRAH-2DH2.1/2-SUPPLY_AIR_TEMP
TARGET = _TAGNAMES[5]  # CRAH-2DH2.1-RETURN_AIR_TEMP


@pytest.fixture
def db_engine():
  """Function-scoped override of the root conftest's module-scoped
  db_engine: several tests below rely on Dataset.save()/get_all() seeing
  a table with an exact, predictable row count (including empty), which
  a fixture shared across the whole test module can't guarantee since
  ORM methods here call db.commit() directly (bypassing rollback-based
  isolation)."""
  engine = create_engine(
      "sqlite:///:memory:", connect_args={"check_same_thread": False},
      poolclass=StaticPool,
  )
  SQLModel.metadata.create_all(engine)
  yield engine
  SQLModel.metadata.drop_all(engine)


_UNSET = object()


def make_dataset(db_session, name="Regression-orm-test", features=_UNSET,
                  target=TARGET, is_valid=True, meta=None, preprocessing=None,
                  n_models=2, task_type=TaskType.Regression):
  if features is _UNSET:
    features = list(FEATURES)
  dataset = Dataset(
      name=name, task_type=task_type, description="", features=features,
      target=target, status=StatusProcess.IDLE, top_model=None,
      start_date="20240101", end_date="20240131", time_start="00:00:00",
      time_end="23:59:00", interval=5, preprocessing=preprocessing,
      is_valid=is_valid, meta=meta if meta is not None else {}, n_models=n_models,
  )
  db_session.add(dataset)
  db_session.commit()
  db_session.refresh(dataset)
  return dataset


def make_model(db_session, dataset, algorithm="lr", is_active=True,
                status=StatusProcess.SUCCESS_TRAIN, path="storages/x/top_model/lr",
                evaluation=None, meta=None):
  model = ModelML(
      dataset_id=dataset.id, name=f"{algorithm}-model", algorithm=algorithm,
      is_active=is_active, evaluation=evaluation or {"R2": 0.9}, status=status,
      path=path, meta=meta if meta is not None else {"size_of": 123},
  )
  db_session.add(model)
  db_session.commit()
  db_session.refresh(model)
  return model


# =============================================================================
# Dataset.to_response / to_responses
# =============================================================================


class TestDatasetToResponse:
  def test_to_response_shape(self, db_session):
    dataset = make_dataset(db_session, name="Regression-toresponse")
    response = dataset.to_response()
    assert response.names == dataset.name
    assert response.task_type == "Regression"
    assert response.features == FEATURES
    assert response.target == TARGET
    assert response.n_models == 2
    assert response.models == []  # no models trained yet -> get_models() -> []

  def test_to_responses_counts(self, db_session):
    make_dataset(db_session, name="Regression-a")
    make_dataset(db_session, name="Regression-b")
    datas = Dataset.get_all(db_session, to_response=False)
    result = Dataset.to_responses(datas, db_session)
    assert result["recordsTotal"] == 2
    assert result["recordsFiltered"] == 2
    assert result["data"] == datas


# =============================================================================
# Dataset.open_dataframe / check_path
# =============================================================================


class TestOpenDataframe:
  def test_raises_when_missing(self, db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "dir", tmp_path)
    dataset = make_dataset(db_session, name="Regression-nofile")
    with pytest.raises(FileNotFoundError):
      dataset.open_dataframe()

  def test_reads_and_parses_dt_column(self, db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "dir", tmp_path)
    dataset = make_dataset(db_session, name="Regression-withfile")
    storage = tmp_path / "storages" / dataset.name
    storage.mkdir(parents=True)
    import pandas as pd
    pd.DataFrame({"dt": ["2024-01-01", "2024-01-02"], FEATURES[0]: [1, 2]}).to_csv(
        storage / "data.csv", index=False
    )
    df = dataset.open_dataframe()
    assert len(df) == 2
    assert str(df["dt"].dtype).startswith("datetime64")


# =============================================================================
# Dataset.check_features
# =============================================================================


class TestCheckFeatures:
  def test_matching_features_with_values_is_true(self, db_session):
    dataset = make_dataset(db_session, name="Regression-checkfeat1")
    assert dataset.check_features({FEATURES[0]: [1.0], FEATURES[1]: [2.0]}) is True

  def test_mismatched_feature_set_is_false(self, db_session):
    dataset = make_dataset(db_session, name="Regression-checkfeat2")
    assert dataset.check_features({FEATURES[0]: [1.0], "other-tag": [2.0]}) is False

  def test_empty_value_list_is_false(self, db_session):
    dataset = make_dataset(db_session, name="Regression-checkfeat3")
    assert dataset.check_features({FEATURES[0]: [], FEATURES[1]: [2.0]}) is False


# =============================================================================
# Dataset.check_integrity
# =============================================================================


class TestCheckIntegrity:
  def test_raises_when_not_valid(self, db_session):
    dataset = make_dataset(db_session, name="Regression-integrity1", is_valid=False)
    with pytest.raises(HTTPException) as exc:
      dataset.check_integrity()
    assert exc.value.status_code == 400

  def test_raises_when_no_features(self, db_session):
    dataset = make_dataset(db_session, name="Regression-integrity2", features=[], meta={"x": 1})
    with pytest.raises(HTTPException):
      dataset.check_integrity()

  def test_raises_when_no_meta(self, db_session):
    dataset = make_dataset(db_session, name="Regression-integrity3", meta={})
    with pytest.raises(HTTPException):
      dataset.check_integrity()

  def test_passes_with_valid_dataset(self, db_session):
    dataset = make_dataset(db_session, name="Regression-integrity4", meta={"created_by": "x"})
    dataset.check_integrity()  # should not raise


# =============================================================================
# Dataset.save
# =============================================================================


class TestDatasetSave:
  def test_save_new_dataset_inserts(self, db_session):
    dataset = Dataset(
        name="Regression-newsave", task_type=TaskType.Regression, description="",
        features=[FEATURES[0]], target=TARGET, status=StatusProcess.IDLE, top_model=None,
        start_date="20240101", end_date="20240131", time_start="00:00:00",
        time_end="23:59:00", interval=5, preprocessing=None, is_valid=True,
        meta={}, n_models=1,
    )
    dataset.save(db_session)
    assert dataset.id is not None
    assert Dataset.get_by_name("Regression-newsave", db_session) is not None

  def test_save_existing_dataset_updates(self, db_session):
    dataset = make_dataset(db_session, name="Regression-updatetest")
    dataset.description = "updated description"
    dataset.save(db_session)

    reloaded = Dataset.get_by_name("Regression-updatetest", db_session)
    assert reloaded.description == "updated description"


# =============================================================================
# Dataset.get_models
# =============================================================================


class TestGetModels:
  def test_empty_when_not_valid(self, db_session):
    dataset = make_dataset(db_session, is_valid=False, name="Regression-invalidmodels")
    make_model(db_session, dataset)
    assert dataset.get_models() == []

  def test_empty_when_no_models(self, db_session):
    dataset = make_dataset(db_session, name="Regression-nomodels")
    assert dataset.get_models() == []

  def test_returns_only_active_models(self, db_session):
    dataset = make_dataset(db_session, name="Regression-activemodels")
    make_model(db_session, dataset, algorithm="lr", is_active=True)
    make_model(db_session, dataset, algorithm="rf", is_active=False)
    db_session.refresh(dataset)

    models = dataset.get_models()
    assert len(models) == 1
    assert models[0]["algorithm"] == "lr"


# =============================================================================
# Dataset classmethod queries
# =============================================================================


class TestDatasetQueries:
  def test_get_by_id_and_name(self, db_session):
    dataset = make_dataset(db_session, name="Regression-lookup")
    assert Dataset.get_by_id(dataset.id, db_session).name == "Regression-lookup"
    assert Dataset.get_by_name("Regression-lookup", db_session).id == dataset.id

  def test_get_by_name_missing_returns_none(self, db_session):
    assert Dataset.get_by_name("does-not-exist", db_session) is None

  def test_get_all_returns_none_when_empty(self, db_session):
    assert Dataset.get_all(db_session) is None

  def test_get_all_to_response_true_returns_dicts(self, db_session):
    make_dataset(db_session, name="Regression-all1")
    result = Dataset.get_all(db_session, to_response=True)
    assert len(result) == 1
    from app.database.schemas import DatasetResponseSchema
    assert isinstance(result[0], DatasetResponseSchema)

  def test_delete_all_removes_everything(self, db_session):
    make_dataset(db_session, name="Regression-del1")
    make_dataset(db_session, name="Regression-del2")
    Dataset.delete_all(db_session)
    assert Dataset.get_all(db_session) is None


# =============================================================================
# Dataset.is_time_to_finetune
# =============================================================================


class TestIsTimeToFinetune:
  def test_false_when_no_preprocessing(self, db_session):
    dataset = make_dataset(db_session, name="Regression-nofinetune", preprocessing=None)
    assert dataset.is_time_to_finetune() is False

  def test_true_when_interval_elapsed(self, db_session):
    old_dt = (datetime.datetime.now() - datetime.timedelta(days=10)).isoformat()
    dataset = make_dataset(
        db_session, name="Regression-duefinetune",
        meta={"current_dt": old_dt},
        preprocessing={"interval_finetune": 5, "retention": 3,
                       "missing_handling": "NEIGHBOR_VALUE", "outlier_handling": False,
                       "scale": False, "dim_reduce": False},
    )
    assert dataset.is_time_to_finetune() is True

  def test_false_when_interval_not_elapsed(self, db_session):
    now_iso = datetime.datetime.now().isoformat()
    dataset = make_dataset(
        db_session, name="Regression-notyetfinetune",
        meta={"current_dt": now_iso},
        preprocessing={"interval_finetune": 30, "retention": 3,
                       "missing_handling": "NEIGHBOR_VALUE", "outlier_handling": False,
                       "scale": False, "dim_reduce": False},
    )
    assert dataset.is_time_to_finetune() is False


# =============================================================================
# ModelML
# =============================================================================


class TestModelML:
  def test_to_response(self, db_session):
    dataset = make_dataset(db_session, name="Regression-modeltoresp")
    model = make_model(db_session, dataset, algorithm="lr")
    response = model.to_response()
    assert response.name == model.name
    assert response.dataset_name == dataset.name
    assert response.algorithm == "lr"

  def test_check_path_false_when_missing(self, db_session):
    dataset = make_dataset(db_session, name="Regression-modelpath")
    model = make_model(db_session, dataset, path="/tmp/does-not-exist-model")
    assert model.check_path() is False

  def test_check_path_true_when_exists(self, db_session, tmp_path):
    dataset = make_dataset(db_session, name="Regression-modelpath2")
    pkl = tmp_path / "model.pkl"
    pkl.write_bytes(b"x")
    model = make_model(db_session, dataset, path=str(tmp_path / "model"))
    assert model.check_path() is True

  def test_view_model_shape(self, db_session):
    dataset = make_dataset(db_session, name="Regression-viewmodel")
    model = make_model(
        db_session, dataset, algorithm="lr",
        meta={"size_of": 456}, evaluation={"R2": 0.99},
    )
    view = model.view_model()
    assert view["name"] == model.name
    assert view["algorithm"] == "lr"
    assert view["size"] == 456
    assert view["evaluation"] == {"R2": 0.99}

  def test_get_by_id_and_name(self, db_session):
    dataset = make_dataset(db_session, name="Regression-modellookup")
    model = make_model(db_session, dataset, algorithm="rf")
    assert ModelML.get_by_id(model.id, db_session).algorithm == "rf"
    assert ModelML.get_by_name(model.name, db_session).id == model.id

  def test_save_new_model_inserts(self, db_session):
    dataset = make_dataset(db_session, name="Regression-modelsave")
    model = ModelML(
        dataset_id=dataset.id, name="new-model", algorithm="lr", is_active=True,
        evaluation={}, status=StatusProcess.SUCCESS_TRAIN, path="x", meta={},
    )
    model.save(db_session)
    assert model.id is not None
    assert ModelML.get_by_name("new-model", db_session) is not None


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
