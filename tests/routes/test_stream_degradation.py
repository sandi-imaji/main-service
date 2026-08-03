"""Degradasi bertahap saat API Smartlink tidak bisa di-poll.

Aturannya (kebijakan ada di app.services.streamer.base, dipakai semua stream):
  * poll gagal  -> stream TIDAK diputus, client dikirimi data terakhir (`stale`)
  * gagal MAX_POLL_FAILURES kali beruntun -> satu frame error
    "API Smartlink sedang bermasalah!"
  * frame error itu dikirim SEKALI, bukan tiap tick
  * poll berhasil lagi -> hitungan reset + client diberi tahu sudah pulih
"""
from types import SimpleNamespace
from unittest.mock import Mock

from app.services.streamer import DEGRADED_RETRY_INTERVAL, MAX_POLL_FAILURES
from app.services.streamer import forecast as forecast_mod
from app.services.streamer import inference as inference_mod
from app.services.streamer import state as state_mod
from app.services.streamer.forecast import ForecastStreamer
from app.services.streamer.inference import InferenceStreamer


def _capture_sends(streamer):
  sent = []
  async def _capture(data):
    sent.append(data)
    return True
  streamer._send_with_retry = _capture
  return sent


def _inference_streamer(inference_func=lambda ds, logger: None):
  s = InferenceStreamer(websocket=Mock(), dataset_name="Regression-x", db=Mock(),
                        inference_func=inference_func)
  s._is_running = True
  s.dataset = SimpleNamespace(name="Regression-x", features=["x1", "x2"])
  return s, _capture_sends(s)


def _forecast_streamer():
  s = ForecastStreamer(websocket=Mock(), dataset_name="TimeSeries-x", db=Mock())
  s._is_running = True
  s.dataset = SimpleNamespace(name="TimeSeries-x", features=["x1", "x2"])
  return s, _capture_sends(s)


def _outages(sent):
  return [m for m in sent if m.get("error") == "smartlink_unavailable"]


class TestOutageThreshold:
  """Kebijakannya di BaseStreamer, jadi berlaku sama untuk semua jenis stream."""

  async def test_stays_quiet_below_threshold(self):
    s, sent = _inference_streamer()
    for _ in range(MAX_POLL_FAILURES - 1):
      await s._on_poll_failed("timeout")
    assert _outages(sent) == []                     # belum diumumkan

  async def test_announces_smartlink_problem_at_threshold(self):
    s, sent = _inference_streamer()
    for _ in range(MAX_POLL_FAILURES):
      await s._on_poll_failed("timeout")

    outages = _outages(sent)
    assert len(outages) == 1
    assert outages[0]["message"] == "API Smartlink sedang bermasalah!"
    assert outages[0]["type"] == "error"
    assert outages[0]["failures"] == MAX_POLL_FAILURES

  async def test_announced_only_once_while_still_down(self):
    s, sent = _inference_streamer()
    for _ in range(MAX_POLL_FAILURES + 5):
      await s._on_poll_failed("timeout")
    assert len(_outages(sent)) == 1                 # tidak membanjiri client

  async def test_recovery_resets_and_notifies(self):
    s, sent = _inference_streamer()
    for _ in range(MAX_POLL_FAILURES):
      await s._on_poll_failed("timeout")
    await s._on_poll_ok()

    assert s._poll_failures == 0
    assert [m["status"] for m in sent if m.get("type") == "status"] == ["smartlink_recovered"]

    # setelah pulih, ambangnya berlaku penuh lagi (bukan langsung teriak)
    await s._on_poll_failed("timeout")
    assert len(_outages(sent)) == 1

  async def test_success_before_threshold_clears_the_count(self):
    s, sent = _inference_streamer()
    await s._on_poll_failed("timeout")
    await s._on_poll_ok()
    await s._on_poll_failed("timeout")
    assert s._poll_failures == 1                    # bukan 2 — hitungannya beruntun
    assert _outages(sent) == []

  async def test_backs_off_only_while_degraded(self):
    s, _ = _inference_streamer()
    s.sleep_interval = 5.0
    assert s._idle_delay() == 5.0
    for _ in range(MAX_POLL_FAILURES):
      await s._on_poll_failed("timeout")
    assert s._idle_delay() == DEGRADED_RETRY_INTERVAL


class TestServesLastKnownActual:
  """get_actual: sekali sukses, kegagalan berikutnya menyajikan nilai terakhir."""

  async def test_failure_returns_previous_value_marked_stale(self, monkeypatch):
    s, sent = _forecast_streamer()
    async def ok(tag, logger=None): return 20.28
    monkeypatch.setattr(forecast_mod, "async_get_realtime", ok)
    await s._handle_client_command('{"type": "get_actual"}')

    async def boom(tag, logger=None): raise RuntimeError("SL down")
    monkeypatch.setattr(forecast_mod, "async_get_realtime", boom)
    await s._handle_client_command('{"type": "get_actual"}')

    assert sent[1]["value"] == 20.28                # nilai terakhir, bukan bolong
    assert sent[1]["stale"] is True
    assert "error" in sent[1]

  async def test_failure_without_history_still_reports_error(self, monkeypatch):
    s, sent = _forecast_streamer()
    async def boom(tag, logger=None): raise RuntimeError("SL down")
    monkeypatch.setattr(forecast_mod, "async_get_realtime", boom)

    await s._handle_client_command('{"type": "get_actual"}')

    assert sent[0]["type"] == "actual" and "error" in sent[0]
    assert "value" not in sent[0]

  async def test_third_failure_announces_smartlink_problem(self, monkeypatch):
    s, sent = _forecast_streamer()
    async def boom(tag, logger=None): raise RuntimeError("SL down")
    monkeypatch.setattr(forecast_mod, "async_get_realtime", boom)

    for _ in range(MAX_POLL_FAILURES):
      await s._handle_client_command('{"type": "get_actual"}')

    assert len(_outages(sent)) == 1

  async def test_stale_value_is_not_reused_for_a_different_tag(self, monkeypatch):
    s, sent = _forecast_streamer()
    async def ok(tag, logger=None): return 1.0
    monkeypatch.setattr(forecast_mod, "async_get_realtime", ok)
    await s._handle_client_command('{"type": "get_actual", "tag": "x1"}')

    async def boom(tag, logger=None): raise RuntimeError("SL down")
    monkeypatch.setattr(forecast_mod, "async_get_realtime", boom)
    await s._handle_client_command('{"type": "get_actual", "tag": "x2"}')

    assert "value" not in sent[1]                   # x1 bukan pengganti x2


class TestInferenceLoopFallback:
  """Loop inference (regression/anomaly/clustering): hasil None / is_valid=False
  berarti poll SL gagal → kirim prediksi terakhir, bukan menjatuhkan stream."""

  def _patch_state(self, monkeypatch, cached):
    async def _upd(name, data): return None
    monkeypatch.setattr(state_mod.stream_state, "update_prediction", _upd)
    monkeypatch.setattr(state_mod.stream_state, "get_last_prediction",
                        lambda name, max_age=30.0: cached)
    async def _add(name, cid): return True          # jadikan koneksi primary
    monkeypatch.setattr(state_mod.stream_state, "add_connection", _add)
    async def _rm(name, cid): return True
    monkeypatch.setattr(state_mod.stream_state, "remove_connection", _rm)

  async def _run_one_tick(self, s, monkeypatch):
    """Jalankan tepat satu iterasi loop lalu hentikan."""
    async def _sleep(_):
      s._is_running = False                         # berhenti setelah 1 tick
    monkeypatch.setattr(inference_mod.asyncio, "sleep", _sleep)
    monkeypatch.setattr(s, "_close_safe", lambda *a, **k: _noop())
    await s.run()

  async def test_invalid_result_sends_last_prediction_as_stale(self, monkeypatch):
    s, sent = _inference_streamer(lambda ds, logger: SimpleNamespace(is_valid=False))
    self._patch_state(monkeypatch, cached={"predictions": {"lr": 1.0}, "_timestamp": 5.0})
    await self._run_one_tick(s, monkeypatch)

    data = [m for m in sent if m.get("stale")]
    assert data and data[0]["predictions"] == {"lr": 1.0}
    assert s._poll_failures == 1

  async def test_no_history_sends_nothing_but_counts_failure(self, monkeypatch):
    s, sent = _inference_streamer(lambda ds, logger: None)
    self._patch_state(monkeypatch, cached=None)
    await self._run_one_tick(s, monkeypatch)

    assert sent == []                               # tak ada yang bisa disajikan
    assert s._poll_failures == 1


async def _noop():
  return None
