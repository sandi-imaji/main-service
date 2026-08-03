"""app.services.clustering — inference, naming, dan pembacaan hasil clustering.

Covers the storage-facing orchestration that the pure core deliberately does
NOT do: human cluster naming, and the inference path that assembles the
combined frame, calls the core, applies naming, and writes the CSVs.
"""
from types import SimpleNamespace

import pandas as pd
import pytest

from app.config import Config
from app.core.contracts import ClusterResult
from app.services import clustering as clustering_service
from app.services import clustering as cluster_helpers


# --- naming helpers --------------------------------------------------------

class TestRenameAndValidate:
  def test_rename_clusters_maps_labels(self):
    df = pd.DataFrame({"kmeans": [0, 1, 0], "x1": [1, 2, 3]})
    out = cluster_helpers.rename_clusters(df, {"kmeans": {0: "cold", 1: "hot"}})
    assert out["kmeans"].tolist() == ["cold", "hot", "cold"]
    assert df["kmeans"].tolist() == [0, 1, 0]  # original untouched (copy)

  def test_rename_ignores_unknown_columns(self):
    df = pd.DataFrame({"kmeans": [0, 1]})
    out = cluster_helpers.rename_clusters(df, {"birch": {0: "x"}})
    assert out["kmeans"].tolist() == [0, 1]

  def test_naming_clusters_valid(self, mock_logger):
    assert cluster_helpers.naming_clusters([0, 1, 0], {0: "a", 1: "b"}, mock_logger) \
        == ["a", "b", "a"]

  def test_naming_clusters_missing_name_raises(self, mock_logger):
    with pytest.raises(ValueError):
      cluster_helpers.naming_clusters([0, 1, 2], {0: "a", 1: "b"}, mock_logger)


class TestNamingPersistence:
  def _prep(self, monkeypatch, tmp_path, name="Clustering-x"):
    monkeypatch.setattr(Config, "dir", tmp_path)
    (tmp_path / "storages" / name / "results").mkdir(parents=True)
    return SimpleNamespace(name=name)

  def test_save_and_get_roundtrip(self, monkeypatch, tmp_path):
    dataset = self._prep(monkeypatch, tmp_path)
    naming = {"kmeans": {"0": "cold", "1": "hot"}}
    cluster_helpers.save_naming_clusters(dataset, naming)
    assert cluster_helpers.get_naming_clusters(dataset.name) == naming

  def test_get_missing_returns_none(self, monkeypatch, tmp_path):
    dataset = self._prep(monkeypatch, tmp_path)
    assert cluster_helpers.get_naming_clusters(dataset.name) is None

  def test_update_merges_values(self, monkeypatch, tmp_path):
    dataset = self._prep(monkeypatch, tmp_path)
    cluster_helpers.save_naming_clusters(dataset, {"kmeans": {"0": "raw0", "1": "raw1"}})
    cluster_helpers.update_naming_clusters(dataset, {"kmeans": {"raw0": "COLD"}})
    updated = cluster_helpers.get_naming_clusters(dataset.name)
    assert updated["kmeans"]["0"] == "COLD"
    assert updated["kmeans"]["1"] == "raw1"

  def test_get_cluster_unique(self, monkeypatch, tmp_path):
    dataset = self._prep(monkeypatch, tmp_path)
    results = tmp_path / "storages" / dataset.name / "results" / "clusters.csv"
    pd.DataFrame({"kmeans": [0, 0, 1, 2], "birch": [1, 1, 1, 1]}).to_csv(results, index=False)
    dataset.models = [SimpleNamespace(algorithm="kmeans"), SimpleNamespace(algorithm="birch")]
    unique = cluster_helpers.get_cluster_unique(dataset)
    assert sorted(unique["kmeans"]) == [0, 1, 2]
    assert unique["birch"] == [1]


# --- inference orchestration ----------------------------------------------

class _ClusterDataset:
  """Dataset stand-in for clustering_service.inference."""
  def __init__(self, name, df, features):
    self.name = name
    self._df = df
    self.features = features
    self.models = [SimpleNamespace(algorithm="kmeans")]
  def open_dataframe(self):
    return self._df.copy()
  def to_cluster_assign_request(self, features, reference=None):
    return SimpleNamespace(features=features, reference=reference,
                           models=[("kmeans", "/tmp/kmeans")], task="Clustering")


class TestInference:
  """Penetapan cluster memakai MODEL TERSIMPAN, bukan melatih ulang.

  Versi sebelumnya transduktif: titik baru ditempel ke data latih lalu seluruhnya
  di-cluster ulang. Terukur pada data identik — pengelompokannya sama (ARI 1.000)
  tapi penomorannya diacak tiap panggilan (label cocok 0%), sehingga warna di
  grafik realtime tidak bisa dibandingkan antar waktu.
  """

  def _prep(self, monkeypatch, tmp_path, name, clusters, naming):
    monkeypatch.setattr(Config, "dir", tmp_path)
    (tmp_path / "storages" / name / "results").mkdir(parents=True)
    train = pd.DataFrame({
        "dt": pd.to_datetime(["2026-07-01 00:00+00:00", "2026-07-01 00:05+00:00"]),
        "x1": [1.0, 2.0], "x2": [3.0, 4.0]})
    # file latih ditulis supaya bisa dipastikan inference tidak menyentuhnya
    train.to_csv(tmp_path / "storages" / name / "data.csv", index=False)
    dataset = _ClusterDataset(name, train, ["x1", "x2"])
    monkeypatch.setattr(
        clustering_service.Unsupervised, "assign",
        staticmethod(lambda req, logger: ClusterResult(clusters=clusters, assignments={})))
    monkeypatch.setattr(clustering_service, "get_naming_clusters", lambda n: naming)
    return dataset

  def test_applies_naming(self, monkeypatch, tmp_path, mock_logger):
    name = "Clustering-y"
    dataset = self._prep(monkeypatch, tmp_path, name, {"kmeans": "Cluster 1"},
                         {"kmeans": {"Cluster 1": "hot"}})
    schema = clustering_service.inference(dataset, {"x1": [9.0], "x2": [9.0]}, mock_logger)
    assert schema.is_valid is True
    assert schema.clusters == {"kmeans": "hot"}

  def test_without_naming_keeps_raw_label(self, monkeypatch, tmp_path, mock_logger):
    name = "Clustering-z"
    dataset = self._prep(monkeypatch, tmp_path, name, {"kmeans": "Cluster 0"}, None)
    schema = clustering_service.inference(dataset, {"x1": [9.0], "x2": [9.0]}, mock_logger)
    assert schema.clusters == {"kmeans": "Cluster 0"}

  def test_unknown_label_falls_back_to_raw_instead_of_crashing(self, monkeypatch, tmp_path, mock_logger):
    """Penamaan lama bisa saja tidak memuat label yang baru muncul."""
    name = "Clustering-n"
    dataset = self._prep(monkeypatch, tmp_path, name, {"kmeans": "Cluster 5"},
                         {"kmeans": {"Cluster 1": "hot"}})
    schema = clustering_service.inference(dataset, {"x1": [9.0], "x2": [9.0]}, mock_logger)
    assert schema.clusters == {"kmeans": "Cluster 5"}

  def test_leaves_every_stored_artifact_untouched(self, monkeypatch, tmp_path, mock_logger):
    """Inference sekarang murni membaca: tidak ada file yang ditulis per tick —
    live_buffer.csv pun tidak lagi dipakai."""
    name = "Clustering-s"
    dataset = self._prep(monkeypatch, tmp_path, name, {"kmeans": "Cluster 0"}, None)
    data_csv = tmp_path / "storages" / name / "data.csv"
    before = data_csv.read_text()

    clustering_service.inference(dataset, {"x1": [9.0], "x2": [9.0]}, mock_logger)

    assert data_csv.read_text() == before
    results = tmp_path / "storages" / name / "results"
    assert not (results / "clusters.csv").exists()
    assert not (results / "live_buffer.csv").exists()

  def test_passes_reference_only_when_clusters_csv_exists(self, monkeypatch, tmp_path, mock_logger):
    """Acuan centroid dibaca dari clusters.csv; kalau belum ada, jangan gagal —
    algoritma yang punya predict tetap bisa jalan."""
    name = "Clustering-r"
    captured = {}
    dataset = self._prep(monkeypatch, tmp_path, name, {"kmeans": "Cluster 0"}, None)
    def _capture(req, logger):
      captured["ref"] = req.reference          # DataFrame: jangan diuji lewat `or`
      return ClusterResult(clusters={"kmeans": "Cluster 0"}, assignments={})
    monkeypatch.setattr(clustering_service.Unsupervised, "assign", staticmethod(_capture))

    clustering_service.inference(dataset, {"x1": [9.0], "x2": [9.0]}, mock_logger)
    assert captured["ref"] is None                       # belum ada clusters.csv

    pd.DataFrame({"x1": [1.0], "x2": [2.0], "kmeans": ["Cluster 0"]}).to_csv(
        tmp_path / "storages" / name / "results" / "clusters.csv", index=False)
    captured.clear()
    clustering_service.inference(dataset, {"x1": [9.0], "x2": [9.0]}, mock_logger)
    assert captured["ref"] is not None and len(captured["ref"]) == 1


