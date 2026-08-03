"""Unsupervised.assign — penetapan cluster yang STABIL.

Alasan keberadaannya, terukur: melatih ulang pada data yang identik menghasilkan
pengelompokan yang sama persis (ARI 1.000) tapi penomoran yang diacak — label
yang cocok 0%. Di stream realtime itu berarti warna titik tidak bisa
dibandingkan antar waktu dan nama cluster pemberian user menempel ke kelompok
yang salah. `assign` memakai satu model tetap, jadi nomornya tidak berubah.

PyCaret di-mock: yang diuji adalah kontrak dan pemilihan jalur, bukan
algoritmanya.
"""
import numpy as np
import pandas as pd
import pytest

from app.core.contracts import ClusterAssignRequest
from app.core.unsupervised import Unsupervised


class _FakePipeline:
  """Meniru Pipeline pycaret: langkah preprocessing + model di akhir."""
  def __init__(self, has_predict=True, scale=1.0):
    self.has_predict = has_predict
    self.scale = scale
  def __getitem__(self, key):                 # pipeline[:-1] → tahap preprocessing
    return _FakeTransformer(self.scale)


class _FakeTransformer:
  def __init__(self, scale): self.scale = scale
  def transform(self, data): return np.asarray(data, dtype=float) * self.scale


@pytest.fixture
def fake_mod(monkeypatch):
  """Ganti modul pycaret.clustering yang dipakai core."""
  calls = {"predict": 0, "load": []}

  class FakeMod:
    @staticmethod
    def load_model(path, **kw):
      calls["load"].append(path)
      return _FakePipeline(has_predict="nopredict" not in path)

    @staticmethod
    def predict_model(model, data=None, **kw):
      calls["predict"] += 1
      if not model.has_predict:
        # pycaret melempar TypeError untuk model tanpa metode predict
        raise TypeError("Model doesn't have the predict method.")
      return pd.DataFrame({"Cluster": ["Cluster 1"] * len(data)})

  monkeypatch.setattr("app.core.unsupervised.mod", FakeMod)
  Unsupervised.clear_cache()
  return calls


def _request(models, reference=None):
  return ClusterAssignRequest(features={"f1": [5.0], "f2": [6.0]},
                              models=models, task="Clustering", reference=reference)


class TestAssign:
  def test_uses_the_saved_model(self, fake_mod, mock_logger):
    out = Unsupervised.assign(_request([("kmeans", "/models/kmeans")]), mock_logger)
    assert out.clusters == {"kmeans": "Cluster 1"}
    assert fake_mod["load"] == ["/models/kmeans"]

  def test_never_refits(self, fake_mod, mock_logger, monkeypatch):
    """Tidak boleh ada setup/create_model — itu jalur transduktif yang lama."""
    def _forbidden(*a, **k):
      raise AssertionError("assign tidak boleh melatih ulang model")
    monkeypatch.setattr("app.core.unsupervised.mod.setup", _forbidden, raising=False)
    monkeypatch.setattr("app.core.unsupervised.mod.create_model", _forbidden, raising=False)
    Unsupervised.assign(_request([("kmeans", "/models/kmeans")]), mock_logger)

  def test_repeated_calls_give_the_same_label(self, fake_mod, mock_logger):
    """Inti perbaikannya: dipanggil berulang, nomornya tidak berubah."""
    req = _request([("kmeans", "/models/kmeans")])
    labels = [Unsupervised.assign(req, mock_logger).clusters["kmeans"] for _ in range(5)]
    assert len(set(labels)) == 1

  def test_handles_several_algorithms(self, fake_mod, mock_logger):
    out = Unsupervised.assign(
        _request([("kmeans", "/models/kmeans"), ("kmodes", "/models/kmodes")]), mock_logger)
    assert set(out.clusters) == {"kmeans", "kmodes"}

  def test_assignments_is_empty_now(self, fake_mod, mock_logger):
    """Penetapan per-baris hanya produk sampingan jalur transduktif; tidak ada
    pemakainya, jadi tidak lagi dihasilkan."""
    out = Unsupervised.assign(_request([("kmeans", "/models/kmeans")]), mock_logger)
    assert out.assignments == {}


class TestNoPredictFallback:
  """Spectral clustering tidak punya `predict` — transduktif secara matematis."""

  def _reference(self):
    return pd.DataFrame({
        "f1": [0.0, 0.1, 10.0, 10.1],
        "f2": [0.0, 0.1, 10.0, 10.1],
        "sc": ["Cluster 0", "Cluster 0", "Cluster 1", "Cluster 1"],
    })

  def test_falls_back_to_nearest_centroid(self, fake_mod, mock_logger):
    req = ClusterAssignRequest(features={"f1": [10.2], "f2": [10.2]},
                               models=[("sc", "/models/sc-nopredict")],
                               task="Clustering", reference=self._reference())
    out = Unsupervised.assign(req, mock_logger)
    assert out.clusters["sc"] == "Cluster 1"        # dekat kelompok yang jauh

  def test_picks_the_other_centroid_for_a_low_point(self, fake_mod, mock_logger):
    req = ClusterAssignRequest(features={"f1": [0.05], "f2": [0.05]},
                               models=[("sc", "/models/sc-nopredict")],
                               task="Clustering", reference=self._reference())
    assert Unsupervised.assign(req, mock_logger).clusters["sc"] == "Cluster 0"

  def test_is_stable_across_calls(self, fake_mod, mock_logger):
    req = ClusterAssignRequest(features={"f1": [10.2], "f2": [10.2]},
                               models=[("sc", "/models/sc-nopredict")],
                               task="Clustering", reference=self._reference())
    labels = [Unsupervised.assign(req, mock_logger).clusters["sc"] for _ in range(5)]
    assert len(set(labels)) == 1

  def test_missing_reference_raises_a_clear_error(self, fake_mod, mock_logger):
    req = ClusterAssignRequest(features={"f1": [1.0], "f2": [1.0]},
                               models=[("sc", "/models/sc-nopredict")],
                               task="Clustering", reference=None)
    with pytest.raises(ValueError, match="centroid"):
      Unsupervised.assign(req, mock_logger)

  def test_reference_without_the_label_column_raises(self, fake_mod, mock_logger):
    ref = self._reference().drop(columns=["sc"])
    req = ClusterAssignRequest(features={"f1": [1.0], "f2": [1.0]},
                               models=[("sc", "/models/sc-nopredict")],
                               task="Clustering", reference=ref)
    with pytest.raises(ValueError):
      Unsupervised.assign(req, mock_logger)

  def test_transform_pipeline_is_applied_to_both_sides(self, fake_mod, mock_logger, monkeypatch):
    """Titik baru dan data acuan harus melewati preprocessing yang SAMA —
    kalau tidak, jaraknya dihitung di dua ruang berbeda."""
    seen = []

    class _Spy(_FakePipeline):
      def __getitem__(self, key):
        t = _FakeTransformer(3.0)
        original = t.transform
        def _record(data):
          seen.append(np.asarray(data).shape)
          return original(data)
        t.transform = _record
        return t

    monkeypatch.setattr("app.core.unsupervised.mod.load_model",
                        staticmethod(lambda path, **kw: _Spy(has_predict=False)))
    Unsupervised.clear_cache()
    req = ClusterAssignRequest(features={"f1": [10.2], "f2": [10.2]},
                               models=[("sc", "/models/sc-nopredict")],
                               task="Clustering", reference=self._reference())
    Unsupervised.assign(req, mock_logger)
    assert seen[0][0] == 4 and seen[1][0] == 1     # 4 baris acuan, lalu 1 titik
