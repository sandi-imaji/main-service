"""Konteks yang menyertai tiap hasil clustering realtime.

Label saja tidak menjawab pertanyaan operator: "kondisi sekarang wajar atau
tidak?". Tiga angka yang menjawabnya, dan diuji di sini:
  ratio    — jarak titik ke pusat cluster-nya, dibanding radius cluster itu
  share    — seberapa sering kondisi ini muncul secara historis
  duration — sudah berapa lama bertahan di cluster ini
"""
from types import SimpleNamespace

import pandas as pd
import pytest

from app.config import Config
from app.core.contracts import ClusterResult
from app.services import clustering as clustering_service


@pytest.fixture(autouse=True)
def clean_state():
  clustering_service.reset_episode_state()
  yield
  clustering_service.reset_episode_state()


class _Dataset:
  def __init__(self, name="Clustering-c"):
    self.name = name
    self.features = ["x1", "x2"]
    self.models = [SimpleNamespace(algorithm="kmeans")]
  def to_cluster_assign_request(self, features, reference=None):
    return SimpleNamespace(features=features, reference=reference)


def _reference():
  """Acuan historis: Cluster 0 muncul 3×, Cluster 1 sekali (25%)."""
  return pd.DataFrame({
      "x1": [1.0, 1.1, 1.2, 9.0],
      "x2": [1.0, 1.1, 1.2, 9.0],
      "kmeans": ["Cluster 0", "Cluster 0", "Cluster 0", "Cluster 1"],
  })


def _prep(monkeypatch, tmp_path, label, distances, with_reference=True):
  monkeypatch.setattr(Config, "dir", tmp_path)
  name = "Clustering-c"
  results = tmp_path / "storages" / name / "results"
  results.mkdir(parents=True)
  if with_reference:
    _reference().to_csv(results / "clusters.csv", index=False)
  monkeypatch.setattr(clustering_service, "get_naming_clusters", lambda n: None)
  monkeypatch.setattr(
      clustering_service.Unsupervised, "assign",
      staticmethod(lambda req, logger: ClusterResult(
          clusters={"kmeans": label}, assignments={}, distances=distances)))
  return _Dataset(name)


class TestUnusualFlag:
  def test_point_near_the_centre_is_not_flagged(self, monkeypatch, tmp_path, mock_logger):
    ds = _prep(monkeypatch, tmp_path, "Cluster 0",
               {"kmeans": {"distance": 0.5, "radius": 1.0, "ratio": 0.5}})
    ctx = clustering_service.inference(ds, {"x1": [1.0], "x2": [1.0]}, mock_logger).context
    assert ctx["kmeans"]["unusual"] is False
    assert ctx["kmeans"]["ratio"] == 0.5

  def test_far_point_is_flagged(self, monkeypatch, tmp_path, mock_logger):
    """Pada data nyata, pencilan mencapai 21× radius sambil tetap dilabeli
    cluster 'normal' — label saja tidak akan memunculkannya."""
    ds = _prep(monkeypatch, tmp_path, "Cluster 0",
               {"kmeans": {"distance": 21.0, "radius": 1.0, "ratio": 21.0}})
    ctx = clustering_service.inference(ds, {"x1": [1.0], "x2": [1.0]}, mock_logger).context
    assert ctx["kmeans"]["unusual"] is True

  def test_threshold_is_two_times_the_radius(self, monkeypatch, tmp_path, mock_logger):
    """Radius itu RATA-RATA, jadi separuh anggota normal sudah di atas 1×."""
    assert clustering_service.UNUSUAL_RATIO == 2.0
    ds = _prep(monkeypatch, tmp_path, "Cluster 0",
               {"kmeans": {"distance": 1.9, "radius": 1.0, "ratio": 1.9}})
    assert clustering_service.inference(
        ds, {"x1": [1.0], "x2": [1.0]}, mock_logger).context["kmeans"]["unusual"] is False

  def test_missing_distance_does_not_break_the_payload(self, monkeypatch, tmp_path, mock_logger):
    ds = _prep(monkeypatch, tmp_path, "Cluster 0", {})
    ctx = clustering_service.inference(ds, {"x1": [1.0], "x2": [1.0]}, mock_logger).context
    assert ctx["kmeans"]["unusual"] is False
    assert "ratio" not in ctx["kmeans"]


class TestHistoricalShare:
  def test_share_comes_from_the_reference(self, monkeypatch, tmp_path, mock_logger):
    ds = _prep(monkeypatch, tmp_path, "Cluster 1", {"kmeans": {"ratio": 0.5, "radius": 1.0}})
    ctx = clustering_service.inference(ds, {"x1": [9.0], "x2": [9.0]}, mock_logger).context
    assert ctx["kmeans"]["share"] == 0.25            # 1 dari 4 baris

  def test_common_cluster_reports_a_high_share(self, monkeypatch, tmp_path, mock_logger):
    ds = _prep(monkeypatch, tmp_path, "Cluster 0", {"kmeans": {"ratio": 0.5, "radius": 1.0}})
    ctx = clustering_service.inference(ds, {"x1": [1.0], "x2": [1.0]}, mock_logger).context
    assert ctx["kmeans"]["share"] == 0.75

  def test_no_reference_means_no_share_but_still_works(self, monkeypatch, tmp_path, mock_logger):
    ds = _prep(monkeypatch, tmp_path, "Cluster 0", {"kmeans": {"ratio": 0.5}},
               with_reference=False)
    ctx = clustering_service.inference(ds, {"x1": [1.0], "x2": [1.0]}, mock_logger).context
    assert "share" not in ctx["kmeans"]
    assert "duration_minutes" in ctx["kmeans"]       # sisanya tetap terisi


class TestEpisodeDuration:
  def _run(self, ds, mock_logger):
    return clustering_service.inference(ds, {"x1": [1.0], "x2": [1.0]}, mock_logger).context

  def test_first_observation_is_marked_inexact(self, monkeypatch, tmp_path, mock_logger):
    """Saat mulai memantau, cluster itu mungkin sudah aktif sejak lama —
    durasinya batas bawah, bukan angka pasti."""
    ds = _prep(monkeypatch, tmp_path, "Cluster 0", {"kmeans": {"ratio": 0.5}})
    ctx = self._run(ds, mock_logger)
    assert ctx["kmeans"]["duration_exact"] is False
    assert ctx["kmeans"]["duration_minutes"] >= 0

  def test_duration_grows_while_the_label_holds(self, monkeypatch, tmp_path, mock_logger):
    ds = _prep(monkeypatch, tmp_path, "Cluster 0", {"kmeans": {"ratio": 0.5}})
    first = self._run(ds, mock_logger)["kmeans"]["since"]
    second = self._run(ds, mock_logger)["kmeans"]["since"]
    assert first == second                           # episode yang sama

  def test_a_change_restarts_the_episode_and_is_exact(self, monkeypatch, tmp_path, mock_logger):
    ds = _prep(monkeypatch, tmp_path, "Cluster 0", {"kmeans": {"ratio": 0.5}})
    self._run(ds, mock_logger)

    monkeypatch.setattr(
        clustering_service.Unsupervised, "assign",
        staticmethod(lambda req, logger: ClusterResult(
            clusters={"kmeans": "Cluster 1"}, assignments={}, distances={})))
    ctx = self._run(ds, mock_logger)["kmeans"]

    # perpindahannya kita saksikan sendiri → `since` bisa dipercaya
    assert ctx["duration_exact"] is True
    assert ctx["duration_minutes"] < 1

  def test_state_is_per_dataset(self, monkeypatch, tmp_path, mock_logger):
    ds = _prep(monkeypatch, tmp_path, "Cluster 0", {"kmeans": {"ratio": 0.5}})
    self._run(ds, mock_logger)
    other = _Dataset("Clustering-lain")
    clustering_service._track_episode("Clustering-lain", "kmeans", "Cluster 9",
                                      __import__("app.helpers", fromlist=["DTEncoder"]).DTEncoder.now())
    assert clustering_service._EPISODE_STATE["Clustering-c"]["kmeans"]["label"] == "Cluster 0"
    assert clustering_service._EPISODE_STATE["Clustering-lain"]["kmeans"]["label"] == "Cluster 9"

  def test_reset_clears_one_dataset_only(self, monkeypatch, tmp_path, mock_logger):
    ds = _prep(monkeypatch, tmp_path, "Cluster 0", {"kmeans": {"ratio": 0.5}})
    self._run(ds, mock_logger)
    clustering_service._track_episode("Clustering-lain", "kmeans", "Cluster 9",
                                      __import__("app.helpers", fromlist=["DTEncoder"]).DTEncoder.now())
    clustering_service.reset_episode_state("Clustering-c")
    assert "Clustering-c" not in clustering_service._EPISODE_STATE
    assert "Clustering-lain" in clustering_service._EPISODE_STATE
