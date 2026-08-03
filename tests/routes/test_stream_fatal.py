"""Kegagalan permanen harus DIBERITAHUKAN, bukan sekadar menutup socket.

Sebelumnya stream cuma memanggil `_close_safe()` dengan alasan di parameter
`reason` — yang tidak pernah dibaca frontend. Akibatnya UI hanya menampilkan
"Disconnected" lalu menyambung ulang tiap 3 detik untuk sebab yang tidak akan
hilang sendiri (model anomaly belum dilatih), membanjiri log server.

Sekarang setiap sebab permanen mengirim frame `{type: "error", fatal: true}`
lebih dulu, supaya client bisa menampilkan pesannya DAN berhenti mencoba.
"""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.config import Config
from app.database.schemas import TaskType
from app.services.streamer.forecast import ForecastStreamer
from app.services.streamer.inference import InferenceStreamer


def _capture(streamer):
  sent = []
  async def _send(data):
    sent.append(data)
    return True
  async def _close(*a, **k):
    return None
  streamer._send_with_retry = _send
  streamer._close_safe = _close
  return sent


def _fatal(sent):
  return [m for m in sent if m.get("fatal")]


class TestAnomalyModelMissing:
  """Kasus dari log produksi: buka halaman anomaly padahal modelnya belum ada."""

  def _streamer(self, tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "dir", tmp_path)      # storages/ kosong
    s = InferenceStreamer(websocket=Mock(), dataset_name="Regression-fd7766e9",
                          db=Mock(), inference_func=lambda ds, lg: None,
                          model_path="anomaly.pkl")
    s.dataset = SimpleNamespace(name="Regression-fd7766e9", features=["x1"])
    return s, _capture(s)

  async def test_client_is_told_the_model_is_missing(self, tmp_path, monkeypatch):
    s, sent = self._streamer(tmp_path, monkeypatch)
    await s.run()

    fatal = _fatal(sent)
    assert len(fatal) == 1
    assert fatal[0]["type"] == "error"
    assert fatal[0]["error"] == "model_not_found"
    assert fatal[0]["dataset"] == "Regression-fd7766e9"

  async def test_message_names_the_dataset_and_the_way_out(self, tmp_path, monkeypatch):
    """Sengaja tidak memaku kalimatnya (teksnya user-facing dan boleh diubah):
    yang dijaga cuma isinya — dataset mana, dan bahwa jalan keluarnya melatih
    model lebih dulu."""
    s, sent = self._streamer(tmp_path, monkeypatch)
    await s.run()

    message = _fatal(sent)[0]["message"].lower()
    assert "regression-fd7766e9" in message
    assert "train" in message or "latih" in message

  async def test_loop_never_starts(self, tmp_path, monkeypatch):
    """Tidak boleh ada prediksi yang dikirim — modelnya memang tidak ada."""
    s, sent = self._streamer(tmp_path, monkeypatch)
    await s.run()
    assert all(m.get("type") == "error" for m in sent)
    assert s._is_running is False

  async def test_existing_model_is_not_reported_as_missing(self, tmp_path, monkeypatch):
    """Kebalikannya: kalau file-nya ada, jangan sampai stream ditolak."""
    from app.services.streamer import inference as inference_mod
    from app.services.streamer import state as state_mod

    monkeypatch.setattr(Config, "dir", tmp_path)
    (tmp_path / "storages" / "Regression-ok").mkdir(parents=True)
    (tmp_path / "storages" / "Regression-ok" / "anomaly.pkl").write_text("x")

    s = InferenceStreamer(websocket=Mock(), dataset_name="Regression-ok", db=Mock(),
                          inference_func=lambda ds, lg: None, model_path="anomaly.pkl")
    s.dataset = SimpleNamespace(name="Regression-ok", features=["x1"])
    s.db = None                                   # lewati db.close() di finally
    sent = _capture(s)

    async def _add(name, cid): return True
    async def _rm(name, cid): return True
    monkeypatch.setattr(state_mod.stream_state, "add_connection", _add)
    monkeypatch.setattr(state_mod.stream_state, "remove_connection", _rm)
    monkeypatch.setattr(state_mod.stream_state, "get_last_prediction",
                        lambda name, max_age=30.0: None)
    # loop-nya `while True`; dihentikan lewat sleep pertama (run() menyalakan
    # _is_running sendiri, jadi mematikannya sebelum run() tidak ada efeknya)
    async def _sleep(_):
      s._is_running = False
    monkeypatch.setattr(inference_mod.asyncio, "sleep", _sleep)

    await s.run()
    assert _fatal(sent) == []


class TestForecastWrongTaskType:
  async def test_non_timeseries_dataset_is_rejected_with_a_reason(self):
    s = ForecastStreamer(websocket=Mock(), dataset_name="Regression-x", db=Mock())
    s.dataset = SimpleNamespace(name="Regression-x", features=["x1"],
                                task_type=TaskType.Regression)
    sent = _capture(s)

    await s.run()

    fatal = _fatal(sent)
    assert len(fatal) == 1
    assert fatal[0]["error"] == "invalid_task_type"


class TestDatasetNotFound:
  async def test_setup_reports_missing_dataset(self, monkeypatch):
    from app.services.streamer import base as base_mod

    s = InferenceStreamer(websocket=Mock(), dataset_name="Tidak-Ada", db=Mock(),
                          inference_func=lambda ds, lg: None)
    sent = _capture(s)
    async def _accept(): return None
    s.websocket.accept = _accept
    monkeypatch.setattr(base_mod.Dataset, "get_by_name",
                        staticmethod(lambda name, db: None))

    assert await s.setup() is False
    assert _fatal(sent)[0]["error"] == "dataset_not_found"
