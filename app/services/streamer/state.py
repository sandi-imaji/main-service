"""State bersama antar koneksi WebSocket untuk satu dataset.

Beberapa client bisa membuka halaman realtime dataset yang sama. Menjalankan
inference sekali per koneksi berarti membayar komputasi yang sama berkali-kali,
jadi koneksi PERTAMA (primary) yang menghitung, sisanya membaca hasilnya dari
sini.
"""
import asyncio
import time
from typing import Dict, Optional, Set


class StreamState:
  """Registry koneksi aktif + prediksi terakhir per dataset."""

  def __init__(self):
    self._active_streams: Dict[str, Set[str]] = {}
    self._last_predictions: Dict[str, dict] = {}
    self._lock = asyncio.Lock()

  async def add_connection(self, dataset_name: str, connection_id: str) -> bool:
    """Tambahkan koneksi baru. Return True jika ini adalah koneksi pertama (harus mulai stream)"""
    async with self._lock:
      if dataset_name not in self._active_streams:
        self._active_streams[dataset_name] = set()
        self._active_streams[dataset_name].add(connection_id)
        return True
      self._active_streams[dataset_name].add(connection_id)
      return False

  async def remove_connection(self, dataset_name: str, connection_id: str) -> bool:
    """Hapus koneksi. Return True jika tidak ada koneksi lagi (harus stop stream)"""
    async with self._lock:
      if dataset_name in self._active_streams:
        self._active_streams[dataset_name].discard(connection_id)
        if not self._active_streams[dataset_name]:
          del self._active_streams[dataset_name]
          if dataset_name in self._last_predictions:
            del self._last_predictions[dataset_name]
          return True
      return False

  async def update_prediction(self, dataset_name: str, prediction: dict):
    """Update prediksi terakhir untuk dataset"""
    async with self._lock:
      timestamp = time.time()
      self._last_predictions[dataset_name] = {
        "prediction": {**prediction, "_timestamp": timestamp},
        "timestamp": timestamp
      }

  def get_last_prediction(self, dataset_name: str, max_age: Optional[float] = 30.0) -> Optional[dict]:
    """Get prediksi terakhir kalau masih dalam `max_age` detik.

    `max_age=None` berarti tanpa batas umur — dipakai jalur fallback yang memang
    sengaja menyajikan data basi saat SL sedang tidak bisa di-poll.
    """
    data = self._last_predictions.get(dataset_name)
    if not data:
      return None
    if max_age is not None and (time.time() - data["timestamp"]) > max_age:
      return None
    return data["prediction"]

  def get_connection_count(self, dataset_name: str) -> int:
    """Get jumlah koneksi aktif untuk dataset"""
    return len(self._active_streams.get(dataset_name, set()))


# Global state instance
stream_state = StreamState()
