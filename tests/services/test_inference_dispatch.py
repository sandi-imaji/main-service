"""app.services.dispatch.auto_inference — satu pintu masuk lintas task.

Regression guard: fungsi ini pernah hilang sama sekali, sehingga
`/models/auto-inference` dan stream realtime anomaly/regression ikut mati. Ia
harus merutekan berdasarkan task type, dan menyerahkan TimeSeries/Clustering ke
modul service masing-masing.
"""
from app.database.schemas import TaskType
from app.services import dispatch as dispatch_service
from tests.conftest import FakeDataset


def test_dispatch_regression(monkeypatch, mock_logger):
  import app.services.supervised as supervised_service
  called = {}
  monkeypatch.setattr(supervised_service, "auto_predictions",
                      lambda ds, lg: called.setdefault("supervised", ds))
  ds = FakeDataset(task_type=TaskType.Regression)
  dispatch_service.auto_inference(ds, mock_logger)
  assert called == {"supervised": ds}


def test_dispatch_classification(monkeypatch, mock_logger):
  import app.services.supervised as supervised_service
  called = {}
  monkeypatch.setattr(supervised_service, "auto_predictions",
                      lambda ds, lg: called.setdefault("supervised", ds))
  ds = FakeDataset(task_type=TaskType.Classification)
  dispatch_service.auto_inference(ds, mock_logger)
  assert "supervised" in called


def test_dispatch_anomaly(monkeypatch, mock_logger):
  import app.services.anomaly as anomaly_service
  called = {}
  monkeypatch.setattr(anomaly_service, "auto_anomaly_detection",
                      lambda ds, lg: called.setdefault("anomaly", ds))
  ds = FakeDataset(task_type=TaskType.Anomaly)
  dispatch_service.auto_inference(ds, mock_logger)
  assert "anomaly" in called


def test_dispatch_timeseries_delegates(monkeypatch, mock_logger):
  import app.services.timeseries as ts_service
  called = {}
  monkeypatch.setattr(ts_service, "auto_inference",
                      lambda ds, lg: called.setdefault("timeseries", ds))
  ds = FakeDataset(task_type=TaskType.TimeSeries)
  dispatch_service.auto_inference(ds, mock_logger)
  assert "timeseries" in called


def test_dispatch_clustering_delegates(monkeypatch, mock_logger):
  import app.services.clustering as clustering_service
  called = {}
  monkeypatch.setattr(clustering_service, "auto_inference",
                      lambda ds, lg: called.setdefault("clustering", ds))
  ds = FakeDataset(task_type=TaskType.Clustering)
  dispatch_service.auto_inference(ds, mock_logger)
  assert "clustering" in called
