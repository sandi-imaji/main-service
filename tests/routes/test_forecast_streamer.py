"""Realtime forecast streamer (app.services.streamer.forecast.ForecastStreamer).

The stream is server-push for forecasts only; actuals are client-driven — the
client sends {"type":"get_actual"} on the same connection and the server answers
with one `actual` frame. These guard that contract:

  * the server answers get_actual (and validates the tag / survives failures),
  * a forecast is (re)sent only when there is none yet or it expired — it must
    NOT re-run the model on every idle tick,
  * expiry still notifies finetuning → finetuned (or finetune_failed),
  * a secondary connection reuses the primary's forecast without re-emitting.
"""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.services.streamer import forecast as streamer_mod
from app.services.streamer.forecast import ForecastStreamer


def _make_streamer():
  s = ForecastStreamer(websocket=Mock(), dataset_name="TimeSeries-x", db=Mock())
  s._is_running = True
  s.dataset = SimpleNamespace(features=["x1", "x2"])
  sent = []
  async def _capture(data):
    sent.append(data)
    return True
  s._send_with_retry = _capture
  return s, sent


# --- client-driven actuals --------------------------------------------------

class TestGetActual:
  async def test_returns_value_for_default_feature(self, monkeypatch):
    s, sent = _make_streamer()
    async def fake_realtime(tag, logger=None):
      return 20.28
    monkeypatch.setattr(streamer_mod, "async_get_realtime", fake_realtime)

    await s._handle_client_command('{"type": "get_actual"}')

    assert len(sent) == 1
    assert sent[0]["type"] == "actual"
    assert sent[0]["tag"] == "x1"                  # defaults to first feature
    assert sent[0]["value"] == 20.28
    assert "timestamp" in sent[0]

  async def test_explicit_tag_from_dataset_is_used(self, monkeypatch):
    s, sent = _make_streamer()
    async def fake_realtime(tag, logger=None):
      return 5.0
    monkeypatch.setattr(streamer_mod, "async_get_realtime", fake_realtime)

    await s._handle_client_command('{"type": "get_actual", "tag": "x2"}')
    assert sent[0]["tag"] == "x2" and sent[0]["value"] == 5.0

  async def test_foreign_tag_is_rejected(self, monkeypatch):
    s, sent = _make_streamer()
    called = {"n": 0}
    async def fake_realtime(tag, logger=None):
      called["n"] += 1
      return 1.0
    monkeypatch.setattr(streamer_mod, "async_get_realtime", fake_realtime)

    await s._handle_client_command('{"type": "get_actual", "tag": "SECRET.TAG"}')

    assert called["n"] == 0                        # never queried upstream
    assert "error" in sent[0]

  async def test_realtime_failure_reports_error_without_raising(self, monkeypatch):
    s, sent = _make_streamer()
    async def boom(tag, logger=None):
      raise RuntimeError("upstream down")
    monkeypatch.setattr(streamer_mod, "async_get_realtime", boom)

    await s._handle_client_command('{"type": "get_actual"}')   # must not raise

    assert sent[0]["type"] == "actual" and "error" in sent[0]

  async def test_dataset_without_features_reports_error(self, monkeypatch):
    s, sent = _make_streamer()
    s.dataset = SimpleNamespace(features=[])
    await s._handle_client_command('{"type": "get_actual"}')
    assert "error" in sent[0]

  @pytest.mark.parametrize("raw", ["ping", "not json", '"a string"', '{"type": "unknown"}'])
  async def test_non_command_messages_are_ignored(self, raw):
    s, sent = _make_streamer()
    await s._handle_client_command(raw)            # must not raise
    assert sent == []


# --- forecast is pushed only when new ---------------------------------------

class TestForecastPacing:
  def _patch(self, monkeypatch, s, expired):
    monkeypatch.setattr(s, "_refresh_dataset", lambda: None)
    ts = streamer_mod.timeseries_service
    monkeypatch.setattr(ts, "is_expired", lambda ds, logger: expired)
    async def _upd(name, data): return None
    monkeypatch.setattr(streamer_mod.stream_state, "update_prediction", _upd)
    return ts

  async def test_forecasts_once_then_idles_while_valid(self, monkeypatch):
    s, _ = _make_streamer()
    ts = self._patch(monkeypatch, s, expired=False)
    calls = {"n": 0}
    def _forecast(ds, logger):
      calls["n"] += 1
      return {"is_valid": True, "timestamps": ["t"], "forecast": {"ets": [1]}}
    monkeypatch.setattr(ts, "forecast", _forecast)

    first = await s._get_forecast_cycle(is_primary=True, logger=Mock())
    second = await s._get_forecast_cycle(is_primary=True, logger=Mock())

    assert first["is_valid"] is True
    assert second is None                          # still valid → nothing new
    assert calls["n"] == 1                         # model NOT re-run on idle tick

  async def test_expiry_triggers_a_new_forecast(self, monkeypatch):
    s, _ = _make_streamer()
    ts = self._patch(monkeypatch, s, expired=False)
    monkeypatch.setattr(ts, "forecast",
                        lambda ds, logger: {"is_valid": True, "timestamps": ["t"], "forecast": {}})
    await s._get_forecast_cycle(is_primary=True, logger=Mock())   # first forecast

    monkeypatch.setattr(ts, "is_expired", lambda ds, logger: True)
    monkeypatch.setattr(ts, "finetune", lambda name: None)
    again = await s._get_forecast_cycle(is_primary=True, logger=Mock())
    assert again is not None                       # expired → re-forecast

  async def test_invalid_forecast_is_retried_next_tick(self, monkeypatch):
    s, _ = _make_streamer()
    ts = self._patch(monkeypatch, s, expired=False)
    calls = {"n": 0}
    def _forecast(ds, logger):
      calls["n"] += 1
      return {"is_valid": False}
    monkeypatch.setattr(ts, "forecast", _forecast)

    assert await s._get_forecast_cycle(is_primary=True, logger=Mock()) is None
    assert await s._get_forecast_cycle(is_primary=True, logger=Mock()) is None
    assert calls["n"] == 2                         # not latched as "has forecast"


class TestPrimaryFinetuneNotify:
  """On expiry the primary must tell the frontend: retrain start → done (or failed)."""

  def _expire_and_patch(self, monkeypatch, s):
    monkeypatch.setattr(s, "_refresh_dataset", lambda: None)
    ts = streamer_mod.timeseries_service
    monkeypatch.setattr(ts, "is_expired", lambda ds, logger: True)
    async def _upd(name, data): return None
    monkeypatch.setattr(streamer_mod.stream_state, "update_prediction", _upd)
    return ts

  async def test_notifies_finetuning_then_finetuned(self, monkeypatch):
    s, sent = _make_streamer()
    ts = self._expire_and_patch(monkeypatch, s)
    monkeypatch.setattr(ts, "finetune", lambda name: None)
    monkeypatch.setattr(ts, "forecast",
                        lambda ds, logger: {"is_valid": True, "timestamps": ["t"], "forecast": {"ets": [1]}})

    data = await s._get_forecast_cycle(is_primary=True, logger=Mock())

    statuses = [(m["status"], m["type"]) for m in sent if m.get("type") == "status"]
    assert statuses == [("finetuning", "status"), ("finetuned", "status")]
    assert data["is_valid"] is True

  async def test_notifies_failure_without_dropping_the_stream(self, monkeypatch):
    """Finetune gagal (pull SL kosong/mati) TIDAK boleh memutus stream: client
    tetap memegang forecast terakhir, cycle-nya cuma mengembalikan None."""
    s, sent = _make_streamer()
    ts = self._expire_and_patch(monkeypatch, s)
    def _boom(name): raise RuntimeError("pull failed")
    monkeypatch.setattr(ts, "finetune", _boom)

    assert await s._get_forecast_cycle(is_primary=True, logger=Mock()) is None

    statuses = [m["status"] for m in sent if m.get("type") == "status"]
    assert statuses == ["finetuning", "finetune_failed"]


class TestSecondaryReuse:
  async def test_secondary_returns_cached_once_then_skips(self, monkeypatch):
    s, _ = _make_streamer()
    s._last_forecast_ts = 0
    cached = {"forecast": {"ets": [1, 2]}, "timestamps": ["t1"], "_timestamp": 111.0}
    monkeypatch.setattr(streamer_mod.stream_state, "get_last_prediction",
                        lambda name, max_age=0: cached)

    first = await s._get_forecast_cycle(is_primary=False, logger=Mock())
    assert first is cached                          # emits the primary's forecast
    second = await s._get_forecast_cycle(is_primary=False, logger=Mock())
    assert second is None                           # unchanged → don't re-emit
