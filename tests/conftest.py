"""
Shared test fixtures.

The tests here are deliberately light: PyCaret and InfluxDB are never invoked
for real. The pure core (app.core) is exercised through its contracts with a
fake PyCaret module, and the service layer (app.services) through a lightweight
`FakeDataset` stand-in plus monkeypatched collaborators. Nothing touches the
real sqlite file, pm2, or a live InfluxDB.

Satu pengecualian yang disengaja: `tests/integration/test_pycaret_contract.py`
memanggil PyCaret SUNGGUHAN. Fake di atas hanya membuktikan kode kita konsisten
dengan tebakan kita tentang PyCaret — berkas itu yang membuktikan tebakannya
benar. Ditandai `slow`, jadi `pytest -m "not slow"` melewatinya.
"""
import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from app.database.schemas import TaskType


@pytest.fixture(scope="session", autouse=True)
def logs_off_the_repo(tmp_path_factory):
  """Arahkan seluruh log test ke direktori sementara.

  Tanpa ini, tiap test yang memanggil `Logger("Regression-x")` akan membuat
  folder sungguhan di `logs/` milik repo — bekas nama dataset palsu yang menumpuk
  dan menyulitkan membedakan log asli.
  """
  from app.config import Config
  from app.logger import LogManager

  with pytest.MonkeyPatch.context() as mp:
    mp.setattr(Config, "log_dir", tmp_path_factory.mktemp("logs"))
    LogManager.reset()          # lepas sink yang terlanjur menunjuk logs/ asli
    yield
  LogManager.reset()


@pytest.fixture
def mock_logger():
  """A logger with the four methods the code calls, all no-ops."""
  logger = Mock()
  logger.info = Mock()
  logger.debug = Mock()
  logger.error = Mock()
  logger.warning = Mock()
  return logger


class FakeDataset:
  """Minimal Dataset stand-in.

  Only the attributes/methods the code under test actually reads are provided,
  so a test declares its intent by what it passes in. The real `to_*_request`
  builders are re-used from app.database.DB via composition where needed.
  """

  def __init__(self, name="Regression-test", task_type=TaskType.Regression,
               features=None, target="1", interval=5, df=None, models=None,
               preprocessing=None, meta=None):
    self.name = name
    self.task_type = task_type
    self.features = features if features is not None else ["x1", "x2"]
    self.target = target
    self.interval = interval
    self.preprocessing = preprocessing
    self.models = models if models is not None else []
    self._df = df
    self.meta = meta or SimpleNamespace(columns=self.features, current_dt=None, n_rows=100)

  def open_dataframe(self):
    if self._df is None:
      raise FileNotFoundError("no dataframe configured for this FakeDataset")
    return self._df.copy()


@pytest.fixture
def fake_model():
  """A ModelML stand-in exposing the fields request-builders read."""
  def _make(algorithm="lr", path="storages/ds/top_model/lr"):
    return SimpleNamespace(algorithm=algorithm, path=path)
  return _make


@pytest.fixture
def sample_df():
  """A tiny dt + two-feature frame, enough for clustering/inference paths."""
  return pd.DataFrame({
      "dt": pd.to_datetime(["2026-07-01 00:00:00+00:00",
                            "2026-07-01 00:05:00+00:00",
                            "2026-07-01 00:10:00+00:00"]),
      "x1": [1.0, 2.0, 3.0],
      "x2": [4.0, 5.0, 6.0],
  })
