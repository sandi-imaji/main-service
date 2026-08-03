"""app.core.unsupervised — pure clustering, PyCaret mocked.

Guards the regression where `train_one` referenced an undefined `module()`
and where `predict` must read the LAST row's label (the appended new point).
"""
import pandas as pd

from app.core.contracts import ClusterRequest
from app.core.unsupervised import Unsupervised


class FakeClusterMod:
  """Returns a fixed per-algorithm Cluster assignment on assign_model."""
  def __init__(self, assignments_by_algo):
    self._assignments = assignments_by_algo
    self._current = None
  def setup(self, *a, **k):
    return None
  def create_model(self, algo, **k):
    self._current = algo
    return f"model:{algo}"
  def assign_model(self, model, **k):
    algo = self._current
    return pd.DataFrame({"x1": range(len(self._assignments[algo])),
                         "Cluster": self._assignments[algo]})


class TestIsSameCluster:
  def test_all_agree(self):
    assert Unsupervised.is_same_cluster({"a": 1, "b": 1}) is True

  def test_disagree(self):
    assert Unsupervised.is_same_cluster({"a": 1, "b": 2}) is False

  def test_agree_with_named_match(self):
    assert Unsupervised.is_same_cluster({"a": 3, "b": 3}, name=3) is True
    assert Unsupervised.is_same_cluster({"a": 3, "b": 3}, name=9) is False


class TestPredict:
  def setup_method(self):
    Unsupervised.clear_cache()

  def test_predict_reads_last_row_and_full_assignment(self, monkeypatch, mock_logger):
    fake = FakeClusterMod({"kmeans": [0, 0, 1], "birch": [1, 1, 1]})
    monkeypatch.setattr("app.core.unsupervised.mod", fake)

    df = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6]})
    req = ClusterRequest(df=df, algorithms=["kmeans", "birch"], n_clusters=2,
                         preprocessing={}, task="Clustering")

    result = Unsupervised.predict(req, mock_logger)

    # cluster of the appended (last) row
    assert result.clusters == {"kmeans": 1, "birch": 1}
    # full per-row assignment
    assert result.assignments["kmeans"] == [0, 0, 1]
    assert result.assignments["birch"] == [1, 1, 1]
