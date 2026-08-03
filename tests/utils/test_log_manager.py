"""app.logger.LogManager — tata letak, isolasi sink, dan mode Verbose.

Yang paling penting di sini adalah kelas TestSinkIsolation. Versi lama memanggil
`logger.remove()` di setiap `Logger()`, dan karena `logger` loguru itu singleton
global, membuat logger dataset B akan MENCABUT sink dataset A: file A jadi kosong
dan semua log berebut masuk ke file yang paling akhir dibuat. Bug itu tidak
kelihatan di worker (satu proses = satu dataset) tapi merusak seluruh log
per-dataset di proses FastAPI.
"""
import pytest

from app.config import Config, Verbose
from app.logger import LogManager


@pytest.fixture(autouse=True)
def isolated_logs(tmp_path, monkeypatch):
  """Arahkan semua log ke tmp_path dan lepas sink sesudahnya."""
  LogManager.reset()
  monkeypatch.setattr(Config, "log_dir", tmp_path)
  monkeypatch.setattr(Config, "verbose", Verbose.NORMAL)
  yield tmp_path
  LogManager.reset()


def _read(path):
  return path.read_text(encoding="utf-8") if path.exists() else ""


class TestLayout:
  @pytest.mark.parametrize("name,folder", [
    ("Regression-a1b2", "Supervised"),
    ("Classification-a1b2", "Supervised"),
    ("Clustering-a1b2", "Unsupervised"),
    ("Anomaly-a1b2", "Unsupervised"),
    ("TimeSeries-a1b2", "TimeSeries"),
  ])
  def test_dataset_goes_to_its_task_folder(self, name, folder):
    assert LogManager.task_folder(name) == folder

  def test_main_and_worker_are_separate_files(self, isolated_logs):
    main = LogManager.log_path("Regression-a1b2", "main")
    worker = LogManager.log_path("Regression-a1b2", "worker")
    assert main.parent == worker.parent          # satu folder dataset
    assert main.name == "main.log" and worker.name == "worker.log"

  def test_non_dataset_names_fall_back_to_global(self, isolated_logs):
    for name in ["global", "influx_storage", "", "system"]:
      assert LogManager.log_path(name).name == "global.log"

  def test_anomaly_addon_follows_the_host_dataset(self, isolated_logs):
    """Worker anomaly memakai nama dataset induknya, jadi lognya harus mendarat
    di folder induk — bukan di folder anomaly sendiri."""
    host = LogManager.log_path("Regression-a1b2", "worker")
    assert "Supervised" in str(host)


class TestSinkIsolation:
  """Regression guard: `logger.remove()` global di versi lama."""

  def test_two_datasets_do_not_share_a_file(self, isolated_logs):
    a = LogManager.get("Regression-aaaa")
    b = LogManager.get("Regression-bbbb")
    a.info("pesan-A")
    b.info("pesan-B")

    log_a = _read(LogManager.log_path("Regression-aaaa"))
    log_b = _read(LogManager.log_path("Regression-bbbb"))

    assert "pesan-A" in log_a and "pesan-B" not in log_a
    assert "pesan-B" in log_b and "pesan-A" not in log_b

  def test_logger_created_earlier_still_writes(self, isolated_logs):
    """Inti bug lama: A dibuat lebih dulu, lalu B dibuat — A harus tetap hidup."""
    a = LogManager.get("Regression-aaaa")
    LogManager.get("Clustering-bbbb")            # dulu ini mematikan sink A
    a.info("masih-hidup")
    assert "masih-hidup" in _read(LogManager.log_path("Regression-aaaa"))

  def test_main_and_worker_channels_do_not_bleed(self, isolated_logs):
    main = LogManager.get("Regression-aaaa", "main")
    worker = LogManager.worker("Regression-aaaa")
    main.info("dari-main")
    worker.info("dari-worker")

    assert "dari-worker" not in _read(LogManager.log_path("Regression-aaaa", "main"))
    assert "dari-main" not in _read(LogManager.log_path("Regression-aaaa", "worker"))

  def test_global_stays_free_of_dataset_noise(self, isolated_logs):
    LogManager.global_logger().info("dari-global")
    LogManager.get("Regression-aaaa").info("dari-dataset")

    global_log = _read(LogManager.log_path("global"))
    assert "dari-global" in global_log
    assert "dari-dataset" not in global_log

  def test_repeated_get_reuses_one_sink(self, isolated_logs):
    for _ in range(5):
      LogManager.get("Regression-aaaa").info("halo")
    # satu sink saja, jadi barisnya tidak terduplikasi 5x
    assert _read(LogManager.log_path("Regression-aaaa")).count("halo") == 5


class TestVerboseModes:
  def _emit(self):
    log = LogManager.get("Regression-aaaa")
    log.info("info-line")
    log.error("error-line")
    return _read(LogManager.log_path("Regression-aaaa"))

  def test_normal_writes_file_but_not_stdout(self, isolated_logs, monkeypatch, capsys):
    monkeypatch.setattr(Config, "verbose", Verbose.NORMAL)
    LogManager.reset()
    content = self._emit()
    assert "info-line" in content and "error-line" in content
    assert capsys.readouterr().out == ""          # pm2 out.log tetap bersih

  def test_debug_writes_file_and_stdout(self, isolated_logs, monkeypatch, capsys):
    monkeypatch.setattr(Config, "verbose", Verbose.DEBUG)
    LogManager.reset()
    content = self._emit()
    assert "info-line" in content
    assert "info-line" in capsys.readouterr().out

  def test_silent_keeps_errors_only_and_no_stdout(self, isolated_logs, monkeypatch, capsys):
    monkeypatch.setattr(Config, "verbose", Verbose.SILENT)
    LogManager.reset()
    content = self._emit()
    assert "info-line" not in content             # INFO dibuang
    assert "error-line" in content                # ERROR tetap punya jejak
    assert capsys.readouterr().out == ""


class TestRetention:
  def test_retention_comes_from_config(self, isolated_logs, monkeypatch):
    """Panjang retensi harus ikut Config (yang dibaca dari env), bukan konstanta."""
    monkeypatch.setattr(Config, "log_retention_days", 3)
    LogManager.reset()
    captured = {}
    import app.logger as logger_module
    real_add = logger_module._loguru.add
    def spy(sink, **kwargs):
      captured.update(kwargs)
      return real_add(sink, **kwargs)
    monkeypatch.setattr(logger_module._loguru, "add", spy)

    LogManager.get("Regression-aaaa")
    assert captured["retention"] == "3 days"
    assert captured["rotation"] == "1 day"        # rotasi harian → retensi hari bermakna
