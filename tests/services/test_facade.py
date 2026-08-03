"""Guards the route → service/schema surface.

Two production bugs motivated these: routes memanggil sejumlah helper cluster
lewat `clustering_service`, dan `TaskType.algorithms()` sempat hilang saat
migrasi schema padahal tiga pemanggil (endpoint algorithms, validasi train,
ModelML.view_model) masih bergantung padanya.
"""
import pytest

from app.database.schemas import TaskType
from app.services import clustering as clustering_service


# Names the route layer accesses as `clustering_service.<name>` — see
# routes/models.py, routes/datasets.py, routes/utils.py.
CLUSTERING_FACADE = [
    "inference", "auto_inference",
    "get_clusters", "get_cluster_unique",
    "get_naming_clusters", "update_naming_clusters",
]


@pytest.mark.parametrize("name", CLUSTERING_FACADE)
def test_clustering_facade_exposes(name):
  assert hasattr(clustering_service, name), \
      f"clustering_service.{name} dipakai routes tapi tidak ter-ekspos"


class TestTaskTypeAlgorithms:
  @pytest.mark.parametrize("tt", list(TaskType))
  def test_returns_nonempty_dict(self, tt):
    algos = tt.algorithms()
    assert isinstance(algos, dict) and algos, f"{tt.name} tidak punya algoritma"
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in algos.items())

  def test_used_by_view_model_lookup(self):
    # ModelML.view_model does `.algorithms().get(algo, "")`
    assert TaskType.Regression.algorithms().get("lr") == "Linear Regression"

  def test_used_by_train_validation(self):
    # train.train_model does `algorithm not in .algorithms().keys()`
    assert "kmeans" in TaskType.Clustering.algorithms().keys()
