"""Jalur log dari penulis sampai ke client.

Menyambung tiga bagian yang tadinya saling lepas dan sempat menunjuk path
berbeda: LogManager (penulis) → parse_log_line (pembaca) → read_logs / streamer
(penyaji). Yang dikunci di sini adalah bahwa ketiganya sepakat soal LOKASI dan
FORMAT — dulu `read_logs("system")` membaca `logger/main.log` padahal
LOGGER_GLOBAL menulis ke `storages/global/logs/main.log`, jadi log sistem tidak
pernah sampai ke frontend.
"""
import pytest
from fastapi.testclient import TestClient

from app.config import Config, Verbose
from app.helpers import parse_log_line, read_logs
from app.logger import LogManager


@pytest.fixture(autouse=True)
def isolated_logs(tmp_path, monkeypatch):
  LogManager.reset()
  monkeypatch.setattr(Config, "log_dir", tmp_path)
  monkeypatch.setattr(Config, "verbose", Verbose.NORMAL)
  yield tmp_path
  LogManager.reset()


class TestWriterReaderAgree:
  """Yang ditulis LogManager harus bisa dibaca kembali oleh read_logs."""

  def test_round_trip_dataset_main(self):
    LogManager.get("Regression-a1b2").info("mulai training")
    logs = read_logs("Regression-a1b2")
    assert [l["message"] for l in logs] == ["mulai training"]
    assert logs[0]["level"] == "INFO"
    assert logs[0]["dataset"] == "Regression-a1b2"

  def test_round_trip_worker_channel(self):
    LogManager.worker("Regression-a1b2").info("tick worker")
    assert read_logs("Regression-a1b2", "worker")[0]["message"] == "tick worker"
    assert read_logs("Regression-a1b2", "main") == []      # tidak bocor ke main

  def test_system_reads_the_global_file(self):
    """Regression guard: 'system' dulu membaca file yang tidak pernah ditulis."""
    LogManager.global_logger().info("peristiwa permukaan")
    assert [l["message"] for l in read_logs("system")] == ["peristiwa permukaan"]

  def test_missing_file_returns_empty_not_error(self):
    assert read_logs("Regression-belum-ada") == []


class TestParseLogLine:
  def test_parses_the_current_format(self):
    LogManager.get("Clustering-a1b2").warning("hati-hati")
    raw = LogManager.log_path("Clustering-a1b2").read_text().strip()
    parsed = parse_log_line(raw)
    assert parsed["level"] == "WARNING"
    assert parsed["dataset"] == "Clustering-a1b2"
    assert parsed["message"] == "hati-hati"
    assert isinstance(parsed["line"], int)

  def test_still_parses_old_lines_without_dataset(self):
    """File log lama tidak punya field [dataset] — jangan sampai jatuh ke
    fallback dan kehilangan level/modulnya."""
    old = "[2026-07-01 10:00:00] [INFO] app.pull:pulling:617 - Enter Pull ..."
    parsed = parse_log_line(old)
    assert parsed["level"] == "INFO"
    assert parsed["module"] == "app.pull"
    assert parsed["message"] == "Enter Pull ..."
    assert parsed["dataset"] == ""

  def test_module_level_function_name_is_parsed(self):
    """loguru menulis `<module>` untuk log di level modul; regex lama menolaknya
    dan seluruh barisnya berakhir di fallback."""
    line = "[2026-07-01 10:00:00] [INFO] [ds] app.server:<module>:12 - boot"
    parsed = parse_log_line(line)
    assert parsed["function"] == "<module>"
    assert parsed["message"] == "boot"

  def test_garbage_line_does_not_raise(self):
    assert parse_log_line("bukan format log")["message"] == "bukan format log"
    assert parse_log_line("   ") == {}


class TestRequestMiddleware:
  """Poin 1: global.log mencatat permukaan API — semua route, otomatis."""

  def _client(self):
    from app.server import app
    return TestClient(app)

  def test_successful_request_is_logged(self):
    with self._client() as client:
      client.get("/utils/task-types")
    logs = read_logs("system")
    assert any("GET /utils/task-types -> 200" in l["message"] for l in logs)

  def test_duration_is_recorded(self):
    with self._client() as client:
      client.get("/utils/task-types")
    line = [l for l in read_logs("system") if "task-types" in l["message"]][-1]
    assert "ms)" in line["message"]

  def test_not_found_is_logged_as_warning(self):
    with self._client() as client:
      client.get("/rute/yang/tidak/ada")
    logs = [l for l in read_logs("system") if "tidak/ada" in l["message"]]
    assert logs and logs[-1]["level"] == "WARNING"

  def test_dataset_logs_stay_out_of_global(self):
    LogManager.get("Regression-a1b2").info("detail proses")
    with self._client() as client:
      client.get("/utils/task-types")
    assert not any("detail proses" in l["message"] for l in read_logs("system"))
