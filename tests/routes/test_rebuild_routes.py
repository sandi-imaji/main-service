"""Rebuild & Retrain endpoints — the contract behind the project-page buttons.

The work itself runs as a BackgroundTask, so what is locked down here is the
response and the scheduling; the rebuild behaviour itself is covered by
`tests/services/test_rebuild.py`.
"""
import pytest
from fastapi.testclient import TestClient

from app.routes import models as models_route
from app.services import tasks


class _Dataset:
  def __init__(self, name="Regression-test", is_valid=True):
    self.name = name
    self.is_valid = is_valid


@pytest.fixture
def client():
  from app.server import app
  with TestClient(app) as c:
    yield c


@pytest.fixture
def scheduled(monkeypatch):
  """Capture the background task without running real training."""
  calls = []
  monkeypatch.setattr(tasks, "rebuild_dataset",
                      lambda name, n=0: calls.append(("rebuild", name, n)))
  monkeypatch.setattr(tasks, "retrain_dataset",
                      lambda name, n=0: calls.append(("retrain", name, n)))
  return calls


@pytest.fixture
def existing_dataset(monkeypatch):
  ds = _Dataset()
  monkeypatch.setattr(models_route, "get_dataset_or_404", lambda name, db: ds)
  return ds


class TestRebuild:
  def test_schedules_the_job_and_answers_immediately(self, client, existing_dataset,
                                                     scheduled):
    r = client.post("/models/Regression-test/rebuild")
    assert r.status_code == 200
    assert r.json()["status"] == "rebuilding"
    assert scheduled == [("rebuild", "Regression-test", 0)]

  def test_n_models_is_forwarded(self, client, existing_dataset, scheduled):
    client.post("/models/Regression-test/rebuild?n_models=2")
    assert scheduled[0][2] == 2

  def test_unknown_dataset_gives_404(self, client, monkeypatch, scheduled):
    from app.exceptions import NotFoundException
    def _404(name, db): raise NotFoundException("Dataset", name)
    monkeypatch.setattr(models_route, "get_dataset_or_404", _404)
    assert client.post("/models/does-not-exist/rebuild").status_code == 404
    assert scheduled == []                       # nothing was scheduled


class TestRetrain:
  def test_schedules_the_job(self, client, existing_dataset, scheduled):
    r = client.post("/models/Regression-test/retrain")
    assert r.status_code == 200
    assert r.json()["status"] == "retraining"
    assert scheduled == [("retrain", "Regression-test", 0)]

  def test_invalid_dataset_is_rejected_before_scheduling(self, client, monkeypatch,
                                                         scheduled):
    """Retrain uses the data already on disk; if that data is not valid there is
    nothing to train. Rejected up front so the user gets a message instead of a
    background job that quietly fails."""
    monkeypatch.setattr(models_route, "get_dataset_or_404",
                        lambda name, db: _Dataset(is_valid=False))
    r = client.post("/models/Regression-test/retrain")
    assert r.status_code == 400          # InvalidStateException
    assert scheduled == []


class TestTheTwoStayDistinct:
  def test_rebuild_does_not_call_retrain_or_vice_versa(self, client, existing_dataset,
                                                       scheduled):
    """Only the pulling differs, so a mix-up would not show in the response —
    only in which job actually got scheduled."""
    client.post("/models/Regression-test/rebuild")
    client.post("/models/Regression-test/retrain")
    assert [c[0] for c in scheduled] == ["rebuild", "retrain"]
