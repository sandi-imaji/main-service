"""
Unit tests for app.core.unsupervised (Clustering).
Pycaret and filesystem interactions are mocked/isolated so these tests
run fast and don't depend on a real dataset or trained models.
"""

import json
import types
import pytest
import pandas as pd
from unittest.mock import Mock

from app.core.unsupervised import Unsupervised, module, UnsupervisedResultSchema
from app.database.schemas import TaskType
from app.config import Config


class FakeModelML:
  def __init__(self, algorithm):
    self.algorithm = algorithm


class FakeDataset:
  """Lightweight stand-in for the Dataset ORM object."""

  def __init__(self, name, features, target, models=None, preprocessing=None,
               task_type=TaskType.Clustering, df=None):
    self.name = name
    self.features = features
    self.target = target
    self.models = models or []
    self.preprocessing = preprocessing
    self.task_type = task_type
    self._df = df

  def open_dataframe(self):
    return self._df.copy()


# =============================================================================
# module() task-type dispatch
# =============================================================================


class TestModuleFunction:
  def test_module_clustering_returns_pycaret_clustering(self):
    from pycaret import clustering
    assert module(TaskType.Clustering) is clustering

  def test_module_anomaly_returns_pycaret_anomaly(self):
    from pycaret import anomaly
    assert module(TaskType.Anomaly) is anomaly

  def test_module_invalid_task_type_raises(self):
    with pytest.raises(ValueError, match="Task Type"):
      module(TaskType.Regression)


# =============================================================================
# Unsupervised.is_same_cluster
# =============================================================================


class TestIsSameCluster:
  def test_all_same_no_name(self):
    assert Unsupervised.is_same_cluster({"kmeans": "A", "birch": "A"}) is True

  def test_all_same_matching_name(self):
    assert Unsupervised.is_same_cluster({"kmeans": "A", "birch": "A"}, "A") is True

  def test_all_same_non_matching_name(self):
    assert Unsupervised.is_same_cluster({"kmeans": "A", "birch": "A"}, "B") is False

  def test_different_values(self):
    assert Unsupervised.is_same_cluster({"kmeans": "A", "birch": "B"}) is False


# =============================================================================
# Unsupervised.rename_clusters
# =============================================================================


class TestRenameClusters:
  def test_renames_mapped_column(self):
    df = pd.DataFrame({"kmeans": [0, 1, 0], "other": ["x", "y", "z"]})
    naming = {"kmeans": {0: "A", 1: "B"}}
    result = Unsupervised.rename_clusters(df, naming)
    assert result["kmeans"].tolist() == ["A", "B", "A"]
    assert result["other"].tolist() == ["x", "y", "z"]

  def test_not_inplace_by_default(self):
    df = pd.DataFrame({"kmeans": [0, 1]})
    naming = {"kmeans": {0: "A", 1: "B"}}
    result = Unsupervised.rename_clusters(df, naming)
    assert df["kmeans"].tolist() == [0, 1]
    assert result["kmeans"].tolist() == ["A", "B"]

  def test_inplace_true_mutates_original(self):
    df = pd.DataFrame({"kmeans": [0, 1]})
    naming = {"kmeans": {0: "A", 1: "B"}}
    result = Unsupervised.rename_clusters(df, naming, inplace=True)
    assert df["kmeans"].tolist() == ["A", "B"]
    assert result is df

  def test_column_not_in_naming_is_skipped(self):
    df = pd.DataFrame({"unknown_col": [0, 1]})
    naming = {"kmeans": {0: "A", 1: "B"}}
    result = Unsupervised.rename_clusters(df, naming)
    assert result["unknown_col"].tolist() == [0, 1]


# =============================================================================
# Unsupervised.naming_clusters (validation helper, static method)
# =============================================================================


class TestNamingClustersValidation:
  def test_valid_mapping_renames_all(self, mock_logger):
    clusters = pd.Series(["Cluster 0", "Cluster 1", "Cluster 0"])
    naming = {"Cluster 0": "Normal", "Cluster 1": "Anomaly"}
    result = Unsupervised.naming_clusters(clusters, naming, mock_logger)
    assert result == ["Normal", "Anomaly", "Normal"]

  def test_missing_key_raises_value_error(self, mock_logger):
    clusters = pd.Series(["Cluster 0", "Cluster 2"])
    naming = {"Cluster 0": "Normal"}
    with pytest.raises(ValueError):
      Unsupervised.naming_clusters(clusters, naming, mock_logger)
    mock_logger.error.assert_called_once()


# =============================================================================
# Naming clusters persistence (save/get/update) - filesystem, isolated via
# monkeypatched Config.dir
# =============================================================================


class TestNamingClustersPersistence:
  @pytest.fixture(autouse=True)
  def _isolate_storage(self, tmp_path, monkeypatch):
    dataset_name = "Clustering-test"
    storage = tmp_path / "storages" / dataset_name / "results"
    storage.mkdir(parents=True)
    monkeypatch.setattr(Config, "dir", tmp_path)
    self.dataset_name = dataset_name
    self.storage = storage

  def test_get_naming_clusters_missing_returns_none(self):
    assert Unsupervised.get_naming_clusters(self.dataset_name) is None

  def test_save_then_get_naming_clusters(self):
    naming = {"kmeans": {"Cluster 0": "A", "Cluster 1": "B"}}
    dataset = FakeDataset(self.dataset_name, [], "2")
    Unsupervised.save_naming_clusters(dataset, naming)

    fpath = self.storage / "naming_clusters.json"
    assert fpath.exists()
    assert Unsupervised.get_naming_clusters(self.dataset_name) == naming

  def test_update_naming_clusters_creates_when_missing(self):
    dataset = FakeDataset(self.dataset_name, [], "2")
    update = {"kmeans": {"Cluster 0": "A"}}
    Unsupervised.update_naming_clusters(dataset, update)
    assert Unsupervised.get_naming_clusters(self.dataset_name) == update

  def test_update_naming_clusters_remaps_existing(self):
    dataset = FakeDataset(self.dataset_name, [], "2")
    Unsupervised.save_naming_clusters(
        dataset, {"kmeans": {"Cluster 0": "A", "Cluster 1": "B"}}
    )
    result = Unsupervised.update_naming_clusters(
        dataset, {"kmeans": {"A": "Renamed-A"}}
    )
    assert result["kmeans"]["Cluster 0"] == "Renamed-A"
    assert result["kmeans"]["Cluster 1"] == "B"


# =============================================================================
# get_cluster_unique
# =============================================================================


class TestGetClusterUnique:
  def test_missing_clusters_csv_raises(self, tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "dir", tmp_path)
    dataset = FakeDataset("Clustering-x", [], "2", models=[FakeModelML("kmeans")])
    with pytest.raises(ValueError, match="clusters.csv"):
      Unsupervised.get_cluster_unique(dataset)

  def test_returns_unique_values_per_algorithm(self, tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "dir", tmp_path)
    dataset_name = "Clustering-x"
    results_dir = tmp_path / "storages" / dataset_name / "results"
    results_dir.mkdir(parents=True)
    pd.DataFrame({"kmeans": ["A", "A", "B"], "birch": ["X", "Y", "X"]}).to_csv(
        results_dir / "clusters.csv", index=False
    )
    dataset = FakeDataset(
        dataset_name, [], "2", models=[FakeModelML("kmeans"), FakeModelML("birch")]
    )
    data = Unsupervised.get_cluster_unique(dataset)
    assert set(data["kmeans"]) == {"A", "B"}
    assert set(data["birch"]) == {"X", "Y"}


# =============================================================================
# get_clusters - error paths
# =============================================================================


class TestGetClusters:
  def test_no_models_raises_value_error(self, mock_logger):
    dataset = FakeDataset("Clustering-x", ["f1", "f2"], "2", models=[])
    with pytest.raises(ValueError, match="no trained models"):
      Unsupervised.get_clusters(dataset, mock_logger)

  def test_missing_clusters_csv_raises_file_not_found(
      self, tmp_path, monkeypatch, mock_logger
  ):
    monkeypatch.setattr(Config, "dir", tmp_path)
    dataset = FakeDataset(
        "Clustering-x", ["f1", "f2"], "2", models=[FakeModelML("kmeans")]
    )
    with pytest.raises(FileNotFoundError):
      Unsupervised.get_clusters(dataset, mock_logger)


# =============================================================================
# Cache passthrough (BaseMLCore)
# =============================================================================


class TestUnsupervisedCache:
  def test_get_cache_returns_model_cache(self):
    from app.utils.model_cache import ModelCache
    assert isinstance(Unsupervised.get_cache(), ModelCache)

  def test_clear_cache_empties_cache(self):
    cache = Unsupervised.get_cache()
    cache.put("some/path", Mock())
    Unsupervised.clear_cache()
    assert cache.get("some/path") is None

  def test_get_cache_stats_shape(self):
    Unsupervised.clear_cache()
    stats = Unsupervised.get_cache_stats()
    assert set(stats.keys()) == {"hits", "misses", "size", "max_size", "hit_rate"}


# =============================================================================
# Unsupervised.inference (pycaret module mocked)
# =============================================================================


class TestInference:
  @pytest.fixture(autouse=True)
  def _isolate_storage(self, tmp_path, monkeypatch):
    dataset_name = "Clustering-infer"
    (tmp_path / "storages" / dataset_name / "results").mkdir(parents=True)
    monkeypatch.setattr(Config, "dir", tmp_path)
    self.dataset_name = dataset_name

  def _make_mock_module(self, cluster_value="Cluster 1"):
    mod = Mock()
    mod.setup = Mock(return_value=None)
    mod.create_model = Mock(return_value=Mock())

    assigned = pd.DataFrame({"Cluster": ["Cluster 0", cluster_value]})
    mod.assign_model = Mock(return_value=assigned)
    return mod

  def test_inference_happy_path(self, monkeypatch, mock_logger):
    train_df = pd.DataFrame(
        {"f1": [1.0, 2.0], "f2": [3.0, 4.0], "dt": ["2024-01-01", "2024-01-02"]}
    )
    dataset = FakeDataset(
        self.dataset_name,
        ["f1", "f2"],
        target="2",
        models=[FakeModelML("kmeans")],
        df=train_df,
    )

    mock_mod = self._make_mock_module(cluster_value="Cluster 1")
    monkeypatch.setattr("app.core.unsupervised.module", lambda tt: mock_mod)

    features = {"f1": [5.0], "f2": [6.0]}
    result = Unsupervised.inference(dataset, features, mock_logger)

    assert isinstance(result, UnsupervisedResultSchema)
    assert result.is_valid is True
    assert result.dataset_name == self.dataset_name
    assert result.clusters == {"kmeans": "Cluster 1"}
    mock_mod.setup.assert_called_once()

    storage = Config.dir / "storages" / self.dataset_name
    assert (storage / "data.csv").exists()
    assert (storage / "results" / "clusters.csv").exists()

  def test_inference_applies_naming_clusters(self, monkeypatch, mock_logger):
    train_df = pd.DataFrame(
        {"f1": [1.0, 2.0], "f2": [3.0, 4.0], "dt": ["2024-01-01", "2024-01-02"]}
    )
    dataset = FakeDataset(
        self.dataset_name,
        ["f1", "f2"],
        target="2",
        models=[FakeModelML("kmeans")],
        df=train_df,
    )
    Unsupervised.save_naming_clusters(
        dataset, {"kmeans": {"Cluster 0": "Normal", "Cluster 1": "Anomaly"}}
    )

    mock_mod = self._make_mock_module(cluster_value="Cluster 1")
    monkeypatch.setattr("app.core.unsupervised.module", lambda tt: mock_mod)

    result = Unsupervised.inference(dataset, {"f1": [5.0], "f2": [6.0]}, mock_logger)
    assert result.clusters == {"kmeans": "Anomaly"}


# =============================================================================
# Unsupervised.auto_inference
# =============================================================================


class TestAutoInference:
  def test_auto_inference_success_delegates_to_inference(self, monkeypatch, mock_logger):
    dataset = FakeDataset("Clustering-auto", ["f1", "f2"], "2")
    monkeypatch.setattr(
        "app.core.unsupervised.pull_realtime", lambda features, logger: [1.0, 2.0]
    )

    sentinel = object()
    captured = {}

    def fake_inference(ds, features, logger):
      captured["features"] = features
      return sentinel

    monkeypatch.setattr(Unsupervised, "inference", staticmethod(fake_inference))

    result = Unsupervised.auto_inference(dataset, mock_logger)

    assert result is sentinel
    assert captured["features"] == {"f1": [1.0], "f2": [2.0]}

  def test_auto_inference_catches_exception_and_returns_none(self, monkeypatch, mock_logger):
    dataset = FakeDataset("Clustering-auto", ["f1", "f2"], "2")

    def boom(features, logger):
      raise RuntimeError("pull failed")

    monkeypatch.setattr("app.core.unsupervised.pull_realtime", boom)

    result = Unsupervised.auto_inference(dataset, mock_logger)

    assert result is None
    mock_logger.error.assert_called_once()


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
