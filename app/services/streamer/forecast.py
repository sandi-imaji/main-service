"""Stream forecast time-series.

Beda dari stream inference biasa: server hanya mendorong FORECAST (saat connect
dan tiap kali kedaluwarsa lalu model dilatih ulang). Nilai actual tidak
didorong — client yang memintanya lewat perintah `get_actual` di koneksi yang
sama, karena client yang memegang jam render-nya.
"""
import asyncio
import json
from datetime import datetime
from typing import Optional

from app.helpers import DTEncoder
from app.logger import Logger, LOGGER_GLOBAL
from app.pull import async_get_realtime
from app.services import timeseries as timeseries_service
from app.services.streamer.base import BaseStreamer
from app.services.streamer.state import stream_state


class ForecastStreamer(BaseStreamer):
  """Loop forecast untuk satu koneksi + pelayan perintah `get_actual`."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._last_forecast_ts = 0
    self._has_forecast = False        # sudah pernah kirim forecast di koneksi ini?
    self._last_actual: Optional[dict] = None

  async def run(self) -> None:
    if not self.dataset:
      await self.fail("dataset_not_initialized", "Dataset belum siap dipakai.")
      return

    logger = Logger(self.dataset.name)
    if not self.dataset.task_type.is_timeseries():
      await self.fail(
          "invalid_task_type",
          f"Dataset '{self.dataset_name}' is not a TimeSeries dataset, so it has no forecast.")
      return

    # Pendaftaran koneksi ikut MASUK try: kalau gagal di luar sini, `finally`
    # terlewat dan socket tidak pernah ditutup — client menggantung tanpa pesan
    # apa pun sampai timeout-nya sendiri.
    try:
      is_primary = await stream_state.add_connection(self.dataset_name, self.connection_id)
      self._last_forecast_ts = 0
      self._has_forecast = False
      self._is_running = True
      consecutive_errors = 0
      max_consecutive_errors = 5

      while self._is_running:
        try:
          forecast_data = await self._get_forecast_cycle(is_primary, logger)
          if not forecast_data:
            # forecast masih berlaku / belum siap: idle. Permintaan actual dari
            # client dilayani task handle_client_messages, bukan di sini.
            await asyncio.sleep(self._idle_delay())
            continue

          if not await self._send_with_retry(forecast_data):
            LOGGER_GLOBAL.error("Failed to send forecast after retries")
            break

          consecutive_errors = 0

        except asyncio.CancelledError:
          LOGGER_GLOBAL.info(f"[{self.dataset_name}] Forecast task cancelled")
          raise
        except Exception as e:
          consecutive_errors += 1
          LOGGER_GLOBAL.error(f"[{self.dataset_name}] Forecast error ({consecutive_errors}/{max_consecutive_errors}): {e}")
          if consecutive_errors >= max_consecutive_errors:
            await self._send_with_retry({"error": "Too many consecutive errors", "timestamp": datetime.now().isoformat()})
            break
          await asyncio.sleep(min(self.sleep_interval * 2, 10))

    finally:
      self._is_running = False
      await stream_state.remove_connection(self.dataset_name, self.connection_id)
      await self._close_safe()

  async def _get_forecast_cycle(self, is_primary: bool, logger) -> Optional[dict]:
    """Produce (primary) or reuse (secondary) the forecast for this cycle.

    Primary forecasts only when it has none yet or the current one has expired —
    otherwise it returns None so the caller idles. (Previously the per-timestamp
    actual streaming paced this loop; without it, forecasting on every tick would
    re-run the model every few seconds.) On expiry it sends an 'updating' notice,
    retrains via finetune (in a worker thread since finetune blocks), then
    forecasts. Returns a JSON-able dict, or None when there is nothing new.
    """
    if is_primary:
      self._refresh_dataset()
      expired = timeseries_service.is_expired(self.dataset, logger)
      if self._has_forecast and not expired:
        return None                              # forecast sekarang masih berlaku
      if expired:
        # tell the frontend a retrain is starting …
        await self._notify_status("finetuning", "Forecast kedaluwarsa, melatih ulang model…")
        try:
          # finetune is blocking (pull + pycaret) → run off the event loop.
          await asyncio.to_thread(timeseries_service.finetune, self.dataset_name)
        except Exception as e:
          # Finetune gagal hampir selalu karena pull ke SL kosong/mati. Stream
          # TIDAK diputus: client tetap memegang forecast terakhir, dan
          # gangguannya baru diumumkan setelah MAX_POLL_FAILURES kali.
          await self._notify_status("finetune_failed", f"Finetune gagal: {e}")
          await self._on_poll_failed(f"finetune: {e}")
          return None
        self._refresh_dataset()
        await self._on_poll_ok()
        # … and that the model is fresh again.
        await self._notify_status("finetuned", "Model up to date!, preparing a new forecast!")

      result = await asyncio.wait_for(
        asyncio.to_thread(timeseries_service.forecast, self.dataset, logger), timeout=60.0)
      data = result if isinstance(result, dict) else result.model_dump(mode="json")
      if not data.get("is_valid", True):
        return None                              # belum valid → coba lagi tick berikutnya
      self._has_forecast = True
      await stream_state.update_prediction(self.dataset_name, data)
      return data

    # secondary: reuse the primary's latest forecast (skip if unchanged)
    cached = stream_state.get_last_prediction(self.dataset_name, max_age=1e9)
    if not cached or cached.get("_timestamp", 0) == self._last_forecast_ts:
      return None
    self._last_forecast_ts = cached.get("_timestamp", 0)
    return cached

  # --- Perintah dari client --------------------------------------------------

  async def _handle_client_command(self, raw: str) -> None:
    """Route a client message. Non-JSON text is ignored (keep-alive chatter).

    Supported: {"type": "get_actual", "tag": "<optional>"} — the client drives
    the cadence of realtime reads, since it owns the render clock. The server no
    longer guesses when a forecast timestamp 'arrives'.
    """
    try:
      message = json.loads(raw)
    except (ValueError, TypeError):
      return
    if not isinstance(message, dict):
      return
    if message.get("type") == "get_actual":
      await self._send_actual(message.get("tag"))

  async def _send_actual(self, tag: Optional[str] = None) -> None:
    """Read one realtime value and send it back on this connection.

    `tag` defaults to the dataset's first feature; an explicit tag must belong to
    the dataset so a connection can't be used to probe arbitrary tags. Failures
    are reported as an `actual` frame carrying `error`, never by dropping the
    stream — a bad realtime read must not kill the forecast.
    """
    features = list(self.dataset.features or []) if self.dataset else []
    if not features:
      await self._send_with_retry({"type": "actual", "error": "Dataset tidak punya feature"})
      return
    if tag and tag not in features:
      await self._send_with_retry({"type": "actual", "tag": tag,
                                   "error": "Tag bukan milik dataset ini"})
      return
    tag = tag or features[0]
    try:
      value = await async_get_realtime(tag, logger=Logger(self.dataset_name))
    except Exception as e:
      # Pakai nilai terakhir yang diketahui (ditandai `stale`) supaya grafik
      # tidak bolong; kalau memang belum pernah ada, barulah frame error.
      await self._on_poll_failed(f"get_actual({tag}): {e}")
      if self._last_actual and self._last_actual["tag"] == tag:
        await self._send_with_retry({**self._last_actual, "stale": True, "error": str(e)})
      else:
        await self._send_with_retry({"type": "actual", "tag": tag, "error": str(e)})
      return
    await self._on_poll_ok()
    self._last_actual = {
        "type": "actual",
        "tag": tag,
        "value": float(value),
        "timestamp": DTEncoder.now().isoformat(),
    }
    await self._send_with_retry(self._last_actual)
