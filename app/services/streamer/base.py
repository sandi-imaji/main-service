"""Pondasi semua stream WebSocket realtime.

Yang sama untuk setiap task ada di sini: menerima koneksi, memvalidasi dataset,
mengirim frame dengan retry, menjaga heartbeat, dan menentukan apa yang terjadi
saat poll ke API Smartlink gagal. Yang berbeda hanya isi loopnya — itu urusan
subclass lewat `run()`.
"""
import asyncio
import time
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect
from sqlmodel import Session

from app.database.DB import Dataset
from app.helpers import DTEncoder
from app.logger import LOGGER_GLOBAL


# Berapa kali poll ke API Smartlink boleh gagal beruntun sebelum gangguannya
# diumumkan ke client. Di bawah ambang ini stream tetap jalan dengan data
# terakhir yang diketahui (ditandai `stale`), supaya gangguan sekejap tidak
# langsung mengosongkan grafik.
MAX_POLL_FAILURES = 3

# Jeda antar percobaan saat sedang terdegradasi. Tanpa ini stream akan menembak
# SL tiap sleep_interval (10 detik) selama gangguan — masing-masing dengan pull
# penuh.
DEGRADED_RETRY_INTERVAL = 60.0


class BaseStreamer:
  """Handler satu koneksi WebSocket. Subclass mengisi `run()`."""

  def __init__(
    self,
    websocket: WebSocket,
    dataset_name: str,
    db: Session,
    sleep_interval: float = 3.0,
    max_retries: int = 3,
    heartbeat_interval: float = 30.0
  ):
    self.websocket = websocket
    self.dataset_name = dataset_name
    self.db = db
    self.sleep_interval = sleep_interval
    self.max_retries = max_retries
    self.heartbeat_interval = heartbeat_interval
    self.dataset: Optional[Dataset] = None
    self.connection_id = f"{id(websocket)}_{time.time()}"
    self._is_running = False
    self._last_activity = time.time()
    # Dua task menulis ke socket yang sama (loop stream + balasan permintaan
    # client), jadi pengiriman diserialkan supaya frame tidak saling menyela.
    self._send_lock = asyncio.Lock()
    # Degradasi bertahap saat SL tidak bisa di-poll: sajikan data terakhir dulu,
    # umumkan gangguan hanya setelah MAX_POLL_FAILURES kegagalan beruntun.
    self._poll_failures = 0
    self._outage_announced = False

  # --- Lifecycle -------------------------------------------------------------

  async def run(self) -> None:
    """Loop utama stream. Wajib di-override subclass."""
    raise NotImplementedError

  async def setup(self) -> bool:
    """Setup koneksi WebSocket dan validasi dataset"""
    await self.websocket.accept()

    try:
      dataset = Dataset.get_by_name(name=self.dataset_name, db=self.db)

      if not dataset:
        await self.fail("dataset_not_found",
                        f"Dataset '{self.dataset_name}' was not found.")
        return False

      self.dataset = dataset
      self._last_activity = time.time()
      LOGGER_GLOBAL.info(f"WebSocket connected for dataset: {self.dataset_name}")
      return True

    except Exception as e:
      LOGGER_GLOBAL.error(f"Setup error: {str(e)}")
      await self._close_safe(1011, str(e))
      return False

  def stop(self):
    """Stop streaming loop"""
    self._is_running = False

  def _refresh_dataset(self) -> None:
    """Reload the dataset from DB so meta.current_dt / models reflect the latest
    finetune (which commits from its own session)."""
    try:
      self.db.expire_all()
      self.dataset = Dataset.get_by_name(self.dataset_name, self.db)
    except Exception as e:
      LOGGER_GLOBAL.warning(f"[{self.dataset_name}] refresh dataset failed: {e}")

  # --- Kirim / terima --------------------------------------------------------

  async def _close_safe(self, code: int = 1000, reason: str = ""):
    """Safely close WebSocket connection"""
    try:
      await self.websocket.close(code=code, reason=reason)
    except Exception:
      pass

  async def _send_with_retry(self, data: dict, max_retries: int = 3) -> bool:
    """Send data dengan exponential backoff retry"""
    for attempt in range(max_retries):
      try:
        async with self._send_lock:
          await self.websocket.send_json(data)
        self._last_activity = time.time()
        return True
      except Exception as e:
        wait_time = min(2 ** attempt, 8)
        LOGGER_GLOBAL.warning(f"Send failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
        if attempt == max_retries - 1:
          await asyncio.sleep(wait_time)
    return False

  async def fail(self, error: str, message: str, code: int = 1000) -> None:
    """Tutup stream sambil MEMBERI TAHU alasannya.

    Menutup socket begitu saja hanya menyisakan "Disconnected" di layar, dan
    client akan menyambung ulang tiap beberapa detik selamanya — untuk sebab
    yang tidak akan hilang sendiri (model belum ada, task type salah), itu
    banjir log tanpa guna. `fatal: True` adalah sinyal ke client agar berhenti
    mencoba dan menampilkan pesannya.
    """
    LOGGER_GLOBAL.error(f"[{self.dataset_name}] stream ditutup: {error} — {message}")
    await self._send_with_retry({
        "type": "error",
        "error": error,
        "message": message,
        "fatal": True,
        "dataset": self.dataset_name,
        "timestamp": DTEncoder.now().isoformat(),
    })
    await self._close_safe(code, message)

  async def _notify_status(self, status: str, message: str) -> None:
    """Push a lifecycle status frame (retraining start/done/failed) so the
    frontend can show or clear a 'retraining…' indicator. Distinguishable from
    data frames by `type == "status"`."""
    await self._send_with_retry({
        "type": "status",
        "status": status,
        "message": message,
        "dataset": self.dataset_name,
        "timestamp": DTEncoder.now().isoformat(),
    })

  async def _receive_heartbeat(self) -> bool:
    """Check apakah client masih aktif dengan menerima pesan"""
    try:
      message = await asyncio.wait_for(
        self.websocket.receive_text(),
        timeout=self.heartbeat_interval
      )
      self._last_activity = time.time()
      if message == "ping":
        async with self._send_lock:
          await self.websocket.send_text("pong")
      else:
        await self._handle_client_command(message)
      return True
    except asyncio.TimeoutError:
      return True
    except WebSocketDisconnect:
      return False
    except Exception as e:
      LOGGER_GLOBAL.warning(f"Heartbeat error: {e}")
      return False

  async def _handle_client_command(self, raw: str) -> None:
    """Pesan non-ping dari client. Default: diabaikan (hanya stream forecast
    yang menerima perintah)."""
    return

  async def handle_client_messages(self) -> None:
    """Handle pesan dari client (untuk keep-alive dan control)"""
    try:
      while self._is_running:
        if not await self._receive_heartbeat():
          break
    except WebSocketDisconnect:
      LOGGER_GLOBAL.info(f"[{self.dataset_name}] Client disconnected")
    except Exception as e:
      LOGGER_GLOBAL.warning(f"[{self.dataset_name}] Message handler error: {e}")
    finally:
      self._is_running = False

  # --- Kebijakan gangguan poll Smartlink -------------------------------------

  async def _on_poll_failed(self, reason: str) -> None:
    """Catat satu kegagalan poll ke SL.

    Di bawah MAX_POLL_FAILURES stream sengaja diam saja (pemanggil menyajikan
    data terakhir). Begitu ambangnya tercapai, gangguan diumumkan SEKALI —
    bukan tiap tick — supaya client tidak dibanjiri frame error yang sama.
    """
    self._poll_failures += 1
    LOGGER_GLOBAL.warning(
        f"[{self.dataset_name}] poll SL gagal "
        f"({self._poll_failures}/{MAX_POLL_FAILURES}): {reason}")
    if self._poll_failures < MAX_POLL_FAILURES or self._outage_announced:
      return
    self._outage_announced = True
    await self._send_with_retry({
        "type": "error",
        "error": "smartlink_unavailable",
        "message": "API Smartlink sedang bermasalah!",
        "detail": reason,
        "failures": self._poll_failures,
        "dataset": self.dataset_name,
        "timestamp": DTEncoder.now().isoformat(),
    })

  async def _on_poll_ok(self) -> None:
    """Poll berhasil: reset hitungan, dan kalau gangguan sempat diumumkan,
    beri tahu client bahwa datanya sudah normal lagi (tanpa ini indikator error
    di frontend akan menyala selamanya)."""
    if self._outage_announced:
      self._outage_announced = False
      await self._notify_status("smartlink_recovered", "Koneksi API Smartlink pulih")
    self._poll_failures = 0

  def _idle_delay(self) -> float:
    """Jeda tick berikutnya — dilonggarkan selama SL bermasalah."""
    return max(self.sleep_interval, DEGRADED_RETRY_INTERVAL) if self._outage_announced \
        else self.sleep_interval
