"""Lifecycle koneksi WebSocket.

Manager tidak tahu isi stream-nya; tugasnya membangun streamer yang tepat, lalu
menjalankan dua task berdampingan — loop stream dan loop pesan client — dan
membereskan keduanya begitu salah satunya selesai.
"""
import asyncio
from typing import Callable, Optional

from fastapi import WebSocket
from sqlmodel import Session

from app.logger import LOGGER_GLOBAL
from app.services.streamer.base import BaseStreamer
from app.services.streamer.forecast import ForecastStreamer
from app.services.streamer.inference import InferenceStreamer


class BaseWebSocketManager:
  """Base manager untuk mengelola lifecycle WebSocket connections"""

  def _build_streamer(self, websocket: WebSocket, dataset_name: str, db: Session,
                      sleep_interval: float) -> BaseStreamer:
    """Bangun streamer untuk koneksi ini. Diisi subclass."""
    raise NotImplementedError

  async def handle_connection(
    self,
    websocket: WebSocket,
    dataset_name: str,
    db: Session,
    sleep_interval: float = 3.0
  ):
    """Handle WebSocket connection lifecycle"""
    streamer = self._build_streamer(websocket, dataset_name, db, sleep_interval)

    if not await streamer.setup():
      return

    stream_task = asyncio.create_task(streamer.run())
    message_task = asyncio.create_task(streamer.handle_client_messages())

    try:
      done, pending = await asyncio.wait(
        [stream_task, message_task],
        return_when=asyncio.FIRST_COMPLETED
      )

      for task in pending:
        task.cancel()
        try:
          # Beri waktu lebih lama untuk cleanup (10 detik)
          await asyncio.wait_for(task, timeout=10.0)
        except asyncio.CancelledError:
          LOGGER_GLOBAL.info(f"[{dataset_name}] Task cancelled successfully")
        except asyncio.TimeoutError:
          LOGGER_GLOBAL.warning(f"[{dataset_name}] Task cancellation timeout, forcing...")
          # Force cancel jika timeout
          if not task.done():
            task.cancel()

      if stream_task in done:
        try:
          stream_task.result()
        except asyncio.CancelledError:
          LOGGER_GLOBAL.info(f"[{dataset_name}] Stream task cancelled normally")
        except Exception as e:
          LOGGER_GLOBAL.error(f"Stream task error: {e}")

    except asyncio.CancelledError:
      LOGGER_GLOBAL.info(f"[{dataset_name}] Connection handler cancelled")
      raise

    except Exception as e:
      LOGGER_GLOBAL.error(f"Connection handler error: {e}")

    finally:
      streamer.stop()
      # Explicit cleanup database session
      if streamer.db:
        try:
          streamer.db.close()
          LOGGER_GLOBAL.info(f"[{dataset_name}] Database session closed")
        except Exception as e:
          LOGGER_GLOBAL.warning(f"[{dataset_name}] Error closing database session: {e}")
      LOGGER_GLOBAL.info(f"[{dataset_name}] Connection closed")


class InferenceWebSocketManager(BaseWebSocketManager):
  """Manager stream inference (supervised / anomaly / clustering).

  Yang membedakan ketiganya hanya `inference_func` dan `model_path`, jadi cukup
  satu kelas yang di-instansiasi tiga kali — lihat `app.services.streamer`.
  """

  def __init__(self, inference_func: Callable, model_path: Optional[str] = None):
    self.inference_func = inference_func
    self.model_path = model_path

  def _build_streamer(self, websocket, dataset_name, db, sleep_interval) -> BaseStreamer:
    return InferenceStreamer(
      websocket=websocket, dataset_name=dataset_name, db=db,
      sleep_interval=sleep_interval,
      inference_func=self.inference_func, model_path=self.model_path,
    )


class ForecastWebSocketManager(BaseWebSocketManager):
  """Manager untuk time series forecast WebSocket"""

  def _build_streamer(self, websocket, dataset_name, db, sleep_interval) -> BaseStreamer:
    return ForecastStreamer(
      websocket=websocket, dataset_name=dataset_name, db=db,
      sleep_interval=sleep_interval,
    )
