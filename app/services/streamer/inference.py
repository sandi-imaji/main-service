"""Stream inference realtime: supervised, anomaly, dan clustering.

Ketiganya memakai loop yang persis sama — hitung, kirim, tidur — dan hanya
berbeda pada fungsi inference yang dipanggil serta ada/tidaknya file model yang
harus dicek dulu. Keduanya disuntikkan lewat constructor, bukan dijadikan tiga
kelas yang isinya kembar.
"""
import asyncio
from datetime import datetime
from typing import Callable, Optional

from app.config import Config
from app.logger import Logger, LOGGER_GLOBAL
from app.services.streamer.base import BaseStreamer
from app.services.streamer.state import stream_state


class InferenceStreamer(BaseStreamer):
  """Loop inference berkala untuk satu koneksi.

  `inference_func(dataset, logger)` dijalankan di thread terpisah supaya event
  loop tidak terblokir; `model_path` (relatif ke storage dataset) diisi kalau
  stream ini butuh artefak model tertentu — mis. anomaly yang modelnya tidak
  punya baris ModelML sehingga keberadaannya harus dicek dari file.
  """

  def __init__(self, *args, inference_func: Callable, model_path: Optional[str] = None, **kwargs):
    super().__init__(*args, **kwargs)
    self.inference_func = inference_func
    self.model_path = model_path

  async def run(self) -> None:
    if not self.dataset:
      await self.fail("dataset_not_initialized", "Dataset belum siap dipakai.")
      return

    logger = Logger(self.dataset.name)

    if self.model_path:
      fpath = Config.dir / "storages" / self.dataset.name / self.model_path
      if not fpath.exists():
        await self.fail(
            "model_not_found",
            f"Model anomaly untuk '{self.dataset_name}' is not found!. "
            f"Train first via the anomaly menu, and reload.")
        return

    # Pendaftaran koneksi ikut MASUK try: kalau gagal di luar sini, `finally`
    # terlewat dan socket tidak pernah ditutup — client menggantung tanpa pesan
    # apa pun sampai timeout-nya sendiri.
    try:
      is_primary = await stream_state.add_connection(self.dataset_name, self.connection_id)
      self._is_running = True
      consecutive_errors = 0
      max_consecutive_errors = 5
      last_sent_timestamp = 0  # Track timestamp terakhir yang dikirim

      while self._is_running:
        try:
          if is_primary:
            prediction_data = await self._predict(logger)
            if prediction_data is None:
              await asyncio.sleep(self._idle_delay())
              continue
          else:
            prediction_data = await self._await_primary_prediction(last_sent_timestamp)
            if prediction_data is None:
              await asyncio.sleep(0.5)
              continue

          if not await self._send_with_retry(prediction_data):
            LOGGER_GLOBAL.error("Failed to send data after retries")
            break

          # Update timestamp untuk secondary connections
          if not is_primary and prediction_data:
            last_sent_timestamp = prediction_data.get("_timestamp", 0)

          consecutive_errors = 0
          await asyncio.sleep(self._idle_delay())

        except asyncio.CancelledError:
          LOGGER_GLOBAL.info(f"[{self.dataset_name}] Inference task cancelled")
          raise

        except Exception as e:
          consecutive_errors += 1
          LOGGER_GLOBAL.error(f"[{self.dataset_name}] Error ({consecutive_errors}/{max_consecutive_errors}): {e}")

          if consecutive_errors == max_consecutive_errors:
            await self._send_with_retry({
              "error": "Too many consecutive errors",
              "timestamp": datetime.now().isoformat()
            })
            break

          await asyncio.sleep(min(self.sleep_interval * 2, 10))

    finally:
      self._is_running = False
      await stream_state.remove_connection(self.dataset_name, self.connection_id)
      if self.db: self.db.close()
      await self._close_safe()

  async def _predict(self, logger) -> Optional[dict]:
    """Jalankan satu inference (primary). Return frame siap kirim, atau None
    kalau memang tidak ada yang bisa disajikan tick ini."""
    try:
      result = await asyncio.wait_for(
        asyncio.to_thread(self.inference_func, self.dataset, logger),
        timeout=120.0
      )
    except asyncio.TimeoutError:
      LOGGER_GLOBAL.warning(f"[{self.dataset_name}] Inference timeout after 120s")
      result = None
    except asyncio.CancelledError:
      LOGGER_GLOBAL.info(f"[{self.dataset_name}] Inference cancelled")
      raise

    # Service mengembalikan None / is_valid=False persis ketika poll ke SL tidak
    # menghasilkan nilai. Itu gangguan sumber data, bukan bug: sajikan hasil
    # terakhir (ditandai `stale`) dan catat kegagalannya.
    if result is None or not getattr(result, "is_valid", True):
      await self._on_poll_failed("inference tidak menghasilkan data valid")
      stale = stream_state.get_last_prediction(self.dataset_name, max_age=None)
      return {**stale, "stale": True} if stale else None

    await self._on_poll_ok()
    prediction_data = result.model_dump(mode="json")
    await stream_state.update_prediction(self.dataset_name, prediction_data)
    return prediction_data

  async def _await_primary_prediction(self, last_sent_timestamp: float) -> Optional[dict]:
    """Koneksi sekunder: tunggu prediksi baru dari primary (maksimal selama
    sleep_interval), lalu pakai hasilnya. None berarti belum ada yang baru."""
    cached = None
    wait_iterations = 0
    max_wait_iterations = int(self.sleep_interval / 0.1)

    while self._is_running:
      cached = stream_state.get_last_prediction(self.dataset_name, max_age=30.0)
      if cached and cached.get("_timestamp", 0) > last_sent_timestamp:
        break

      # Cek timeout untuk menghindari infinite wait
      wait_iterations += 1
      if wait_iterations >= max_wait_iterations:
        LOGGER_GLOBAL.warning(f"[{self.dataset_name}] Timeout waiting for new prediction")
        break

      await asyncio.sleep(0.1)

    if not cached or cached.get("_timestamp", 0) == last_sent_timestamp:
      return None
    return cached
