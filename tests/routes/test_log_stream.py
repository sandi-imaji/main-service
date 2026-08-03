"""Endpoint WS /stream/realtime/logs — kontrak yang dipakai LogViewer frontend.

Frontend menyambung dengan `?dataset_name=<ds>&channel=main|worker` lalu
menunggu satu frame `init` berisi riwayat. Yang dikunci di sini adalah bahwa
channel benar-benar memilih FILE yang berbeda — kalau tidak, tombol
Process/Worker di UI cuma hiasan.
"""
import pytest
from fastapi.testclient import TestClient

from app.config import Config, Verbose
from app.logger import LogManager


@pytest.fixture(autouse=True)
def isolated_logs(tmp_path, monkeypatch):
  LogManager.reset()
  monkeypatch.setattr(Config, "log_dir", tmp_path)
  monkeypatch.setattr(Config, "verbose", Verbose.NORMAL)
  yield tmp_path
  LogManager.reset()


@pytest.fixture
def client():
  from app.server import app
  with TestClient(app) as c:
    yield c


def _seed_logs():
  LogManager.get("Regression-ws").info("baris main")
  LogManager.worker("Regression-ws").info("baris worker")


def _init_messages(ws):
  frame = ws.receive_json()
  assert frame["type"] == "init"
  return [entry["message"] for entry in frame["data"]]


class TestChannelSelectsTheFile:
  def test_main_channel_serves_main_log(self, client):
    _seed_logs()
    with client.websocket_connect(
        "/stream/realtime/logs?dataset_name=Regression-ws&channel=main") as ws:
      messages = _init_messages(ws)
    assert "baris main" in messages
    assert "baris worker" not in messages

  def test_worker_channel_serves_worker_log(self, client):
    _seed_logs()
    with client.websocket_connect(
        "/stream/realtime/logs?dataset_name=Regression-ws&channel=worker") as ws:
      messages = _init_messages(ws)
    assert "baris worker" in messages
    assert "baris main" not in messages

  def test_channel_defaults_to_main(self, client):
    """LogViewer versi lama (tanpa param channel) harus tetap dilayani."""
    _seed_logs()
    with client.websocket_connect(
        "/stream/realtime/logs?dataset_name=Regression-ws") as ws:
      messages = _init_messages(ws)
    assert "baris main" in messages

  def test_system_streams_the_global_log(self, client):
    LogManager.global_logger().info("peristiwa permukaan")
    with client.websocket_connect("/stream/realtime/logs?dataset_name=system") as ws:
      messages = _init_messages(ws)
    assert "peristiwa permukaan" in messages


class TestMissingLog:
  def test_dataset_without_logs_gets_no_logs_frame(self, client):
    """Belum ada file log bukan error — frontend cukup menampilkan kosong."""
    with client.websocket_connect(
        "/stream/realtime/logs?dataset_name=Regression-kosong") as ws:
      frame = ws.receive_json()
    assert frame["type"] == "init"
    assert frame["status"] == "no_logs"
    assert frame["data"] == []
