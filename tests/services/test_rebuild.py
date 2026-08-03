"""app.services.rebuild — user-triggered Rebuild and Retrain.

What matters most here is the ordering: the worker MUST be gone before artifacts
are deleted and models retrained. Otherwise a live worker keeps inferring against
model files being removed from under it.
"""

import pytest

from app.config import Config
from app.database.schemas import StatusProcess
from app.exceptions import InvalidStateException
from app.services import rebuild as rebuild_service


class _Model:
  def __init__(self, algorithm):
    self.algorithm = algorithm
    self.name = f"{algorithm}-abc"


class _Dataset:
  def __init__(self, name="Regression-test", models=None, n_models=3,
               is_valid=True, status=StatusProcess.ACTIVE):
    self.name = name
    self.models = models if models is not None else [_Model("lr")]
    self.n_models = n_models
    self.is_valid = is_valid
    self.status = status
    self.top_model = "lr-abc"


class _DB:
  def __init__(self):
    self.deleted = []
    self.commits = 0
  def delete(self, obj): self.deleted.append(obj)
  def commit(self): self.commits += 1
  def refresh(self, obj): pass


@pytest.fixture
def storage(tmp_path, monkeypatch):
  monkeypatch.setattr(Config, "dir", tmp_path)
  root = tmp_path / "storages" / "Regression-test"
  (root / "top_model").mkdir(parents=True)
  (root / "results").mkdir(parents=True)
  (root / "top_model" / "lr.pkl").write_bytes(b"model")
  (root / "results" / "output.csv").write_text("a,b\n1,2\n")
  (root / "data.csv").write_text("dt,x\n2026-01-01,1\n")
  return root


@pytest.fixture
def spy_worker(monkeypatch):
  deleted = []
  monkeypatch.setattr(rebuild_service.WorkerManager, "delete_by_dataset",
                      staticmethod(lambda name: deleted.append(name)))
  return deleted


class TestReset:
  def test_deletes_the_dataset_workers(self, storage, spy_worker, mock_logger):
    rebuild_service.reset(_Dataset(), _DB(), mock_logger)
    assert spy_worker == ["Regression-test"]

  def test_deletes_model_rows_from_db(self, storage, spy_worker, mock_logger):
    ds, db = _Dataset(models=[_Model("lr"), _Model("rf")]), _DB()
    result = rebuild_service.reset(ds, db, mock_logger)
    assert result["models_removed"] == 2
    assert len(db.deleted) == 2

  def test_clears_top_model(self, storage, spy_worker, mock_logger):
    ds = _Dataset()
    rebuild_service.reset(ds, _DB(), mock_logger)
    assert ds.top_model == ""

  def test_removes_model_artifacts_and_results(self, storage, spy_worker, mock_logger):
    rebuild_service.reset(_Dataset(), _DB(), mock_logger)
    assert not (storage / "top_model" / "lr.pkl").exists()
    assert not (storage / "results" / "output.csv").exists()

  def test_folders_are_recreated_not_left_missing(self, storage, spy_worker, mock_logger):
    """The next training run writes here; a missing folder would fail it."""
    rebuild_service.reset(_Dataset(), _DB(), mock_logger)
    assert (storage / "top_model").is_dir() and (storage / "results").is_dir()

  def test_data_csv_is_kept(self, storage, spy_worker, mock_logger):
    """Retrain trains ON this data — deleting it would kill retrain along with
    everything else."""
    rebuild_service.reset(_Dataset(), _DB(), mock_logger)
    assert (storage / "data.csv").exists()


class TestResetToleratesBrokenState:
  """Broken state is exactly why people reach for Rebuild, so the cleanup must
  not fail alongside it. `model_service.clean_models` raises in each of these
  cases — which is why this module carries its own cleanup."""

  def test_dataset_with_no_models_at_all(self, storage, spy_worker, mock_logger):
    result = rebuild_service.reset(_Dataset(models=[]), _DB(), mock_logger)
    assert result["models_removed"] == 0

  def test_model_file_already_gone(self, storage, spy_worker, mock_logger):
    (storage / "top_model" / "lr.pkl").unlink()
    rebuild_service.reset(_Dataset(), _DB(), mock_logger)      # must not raise

  def test_storage_folder_does_not_exist_yet(self, tmp_path, monkeypatch, spy_worker,
                                             mock_logger):
    monkeypatch.setattr(Config, "dir", tmp_path)
    rebuild_service.reset(_Dataset(models=[]), _DB(), mock_logger)
    assert (tmp_path / "storages" / "Regression-test" / "results").is_dir()


class TestModelCount:
  def test_uses_the_datasets_n_models(self):
    assert rebuild_service._n_models(_Dataset(n_models=5), None) == 5

  def test_explicit_argument_wins(self):
    assert rebuild_service._n_models(_Dataset(n_models=5), 2) == 2

  def test_zero_means_use_the_datasets_value(self):
    assert rebuild_service._n_models(_Dataset(n_models=5), 0) == 5

  def test_never_fewer_than_one(self):
    """A dataset may carry n_models=0 (created deliberately without training),
    but an explicit rebuild request always means "train"."""
    assert rebuild_service._n_models(_Dataset(n_models=0), 0) == 1


class TestRebuild:
  def _patch(self, monkeypatch, dataset, order):
    monkeypatch.setattr(rebuild_service.Dataset, "get_by_name",
                        staticmethod(lambda name, db: dataset))
    monkeypatch.setattr(rebuild_service, "get_db_session",
                        lambda: __import__("contextlib").nullcontext(_DB()))
    monkeypatch.setattr(rebuild_service, "pulling",
                        lambda name, db: order.append("pull"))
    monkeypatch.setattr(rebuild_service.train_service, "find_top_models",
                        lambda ds, n, db, lg: order.append(f"train:{n}"))

  def test_order_is_drop_worker_then_pull_then_train(self, storage, monkeypatch, spy_worker):
    order = []
    ds = _Dataset(status=StatusProcess.SUCCESS_PULL)
    self._patch(monkeypatch, ds, order)
    rebuild_service.rebuild("Regression-test")
    assert order == ["pull", "train:3"]
    assert spy_worker == ["Regression-test"]     # worker died BEFORE both

  def test_failed_pull_cancels_training(self, storage, monkeypatch, spy_worker):
    """Training on data that failed to arrive yields a model built from stale or
    empty input — better to stop and leave a trace in the log."""
    order = []
    ds = _Dataset(status=StatusProcess.ERROR_PULL)
    self._patch(monkeypatch, ds, order)
    rebuild_service.rebuild("Regression-test")
    assert order == ["pull"]                     # no "train"

  def test_invalid_dataset_after_pull_cancels_training(self, storage, monkeypatch,
                                                       spy_worker):
    order = []
    ds = _Dataset(status=StatusProcess.SUCCESS_PULL, is_valid=False)
    self._patch(monkeypatch, ds, order)
    rebuild_service.rebuild("Regression-test")
    assert order == ["pull"]

  def test_explicit_n_models_is_forwarded(self, storage, monkeypatch, spy_worker):
    order = []
    ds = _Dataset(status=StatusProcess.SUCCESS_PULL)
    self._patch(monkeypatch, ds, order)
    rebuild_service.rebuild("Regression-test", n_models=1)
    assert order == ["pull", "train:1"]


class TestRetrain:
  def _patch(self, monkeypatch, dataset, order):
    monkeypatch.setattr(rebuild_service.Dataset, "get_by_name",
                        staticmethod(lambda name, db: dataset))
    monkeypatch.setattr(rebuild_service, "get_db_session",
                        lambda: __import__("contextlib").nullcontext(_DB()))
    monkeypatch.setattr(rebuild_service, "pulling",
                        lambda name, db: order.append("pull"))
    monkeypatch.setattr(rebuild_service.train_service, "find_top_models",
                        lambda ds, n, db, lg: order.append(f"train:{n}"))

  def test_does_not_pull(self, storage, monkeypatch, spy_worker):
    """The whole difference from rebuild."""
    order = []
    self._patch(monkeypatch, _Dataset(), order)
    rebuild_service.retrain("Regression-test")
    assert order == ["train:3"]
    assert "pull" not in order

  def test_worker_is_still_dropped_first(self, storage, monkeypatch, spy_worker):
    self._patch(monkeypatch, _Dataset(), [])
    rebuild_service.retrain("Regression-test")
    assert spy_worker == ["Regression-test"]

  def test_status_is_returned_to_success_pull(self, storage, monkeypatch, spy_worker):
    """`find_top_models` starts from the assumption that the data is ready but no
    models exist — exactly the state after a reset."""
    ds = _Dataset(status=StatusProcess.ACTIVE)
    self._patch(monkeypatch, ds, [])
    rebuild_service.retrain("Regression-test")
    assert ds.status == StatusProcess.SUCCESS_PULL

  def test_invalid_dataset_is_rejected(self, storage, monkeypatch, spy_worker):
    """There is no valid data to retrain on; refusing is clearer than training
    something undefined."""
    self._patch(monkeypatch, _Dataset(is_valid=False), [])
    with pytest.raises(InvalidStateException):
      rebuild_service.retrain("Regression-test")
