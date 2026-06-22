from fastapi import WebSocket, WebSocketDisconnect
from app.database.db import get_session
from app.database.orm import Dataset
from app.logger import Logger, LOGGER_GLOBAL
from app.config import Config
from app.helpers import read_logs, parse_log_line
from sqlmodel import Session
from app.core.anomaly import Anomaly
from app.core.supervised import Supervised
from app.core.unsupervised import Unsupervised
from app.core.time_series import TimeSeries
from app.pull import async_get_realtime
from app.helpers import DTEncoder
import asyncio
from datetime import datetime
from typing import Dict, Set, Optional, Callable, Any, List
import time
import pathlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class StreamState:
  """Shared state untuk mengelola multiple WebSocket connections"""

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

  def get_last_prediction(self, dataset_name: str, max_age: float = 30.0) -> Optional[dict]:
    """Get prediksi terakhir jika masih fresh (dalam max_age detik)"""
    data = self._last_predictions.get(dataset_name)
    if data and (time.time() - data["timestamp"]):
      return data["prediction"]
    return None

  def get_connection_count(self, dataset_name: str) -> int:
    """Get jumlah koneksi aktif untuk dataset"""
    return len(self._active_streams.get(dataset_name, set()))


# Global state instance
stream_state = StreamState()


class Streamer:
  """Handler untuk WebSocket streaming dengan optimalisasi resource"""

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

  async def setup(self) -> bool:
    """Setup koneksi WebSocket dan validasi dataset"""
    await self.websocket.accept()

    try:
      dataset = Dataset.get_by_name(name=self.dataset_name, db=self.db)

      if not dataset:
        LOGGER_GLOBAL.warning(f"Dataset {self.dataset_name} is not found!")
        await self._close_safe(1000, "Dataset is not found")
        return False

      self.dataset = dataset
      self._last_activity = time.time()
      LOGGER_GLOBAL.info(f"WebSocket connected for dataset: {self.dataset_name}")
      return True

    except Exception as e:
      LOGGER_GLOBAL.error(f"Setup error: {str(e)}")
      await self._close_safe(1011, str(e))
      return False

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
        await self.websocket.send_json(data)
        self._last_activity = time.time()
        return True
      except Exception as e:
        wait_time = min(2 ** attempt, 8)
        LOGGER_GLOBAL.warning(f"Send failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
        if attempt == max_retries - 1:
          await asyncio.sleep(wait_time)
    return False

  async def _receive_heartbeat(self) -> bool:
    """Check apakah client masih aktif dengan menerima pesan"""
    try:
      message = await asyncio.wait_for(
        self.websocket.receive_text(),
        timeout=self.heartbeat_interval
      )
      self._last_activity = time.time()
      if message == "ping":
        await self.websocket.send_text("pong")
      return True
    except asyncio.TimeoutError:
      return True
    except WebSocketDisconnect:
      return False
    except Exception as e:
      LOGGER_GLOBAL.warning(f"Heartbeat error: {e}")
      return False

  async def _run_inference_loop(
    self,
    inference_func: Callable,
    check_model_exists: bool = True,
    model_path: Optional[str] = None
  ) -> None:
    """Generic inference loop dengan optimasi"""
    if not self.dataset:
      await self._close_safe(1000, "Dataset not initialized")
      return

    logger = Logger(self.dataset.name)

    if check_model_exists and model_path:
      fpath = Config.dir / "storages" / self.dataset.name / model_path
      if not fpath.exists():
        LOGGER_GLOBAL.error(f"Model not found for {self.dataset_name} | model_path : {model_path}")
        await self._close_safe(1000, f"Anomaly Model is not found!")
        return

    is_primary = await stream_state.add_connection(self.dataset_name, self.connection_id)

    try:
      self._is_running = True
      consecutive_errors = 0
      max_consecutive_errors = 5
      last_sent_timestamp = 0  # Track timestamp terakhir yang dikirim

      while self._is_running:
        try:
          if is_primary:
            # Jalankan inference di thread terpisah agar bisa di-cancel
            try:
              result = await asyncio.wait_for(
                asyncio.to_thread(inference_func, self.dataset, logger),
                timeout=120.0  # Timeout 60 detik untuk inference
              )
              prediction_data = result.model_dump(mode="json")
            except asyncio.TimeoutError:
              LOGGER_GLOBAL.warning(f"[{self.dataset_name}] Inference timeout after 60s")
              consecutive_errors += 1
              if consecutive_errors >= max_consecutive_errors:
                break
              await asyncio.sleep(1)
              continue
            except asyncio.CancelledError:
              LOGGER_GLOBAL.info(f"[{self.dataset_name}] Inference cancelled")
              raise
            await stream_state.update_prediction(self.dataset_name, prediction_data)
          else:
            # Secondary connection: tunggu sampai ada prediksi baru
            cached = None
            wait_iterations = 0
            max_wait_iterations = int(self.sleep_interval / 0.1)  # Tunggu max sleep_interval
            
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
              # Tidak ada data baru, skip cycle ini
              await asyncio.sleep(0.5)
              continue
            
            prediction_data = cached

          success = await self._send_with_retry(prediction_data)

          if not success:
            LOGGER_GLOBAL.error("Failed to send data after retries")
            break

          # Update timestamp untuk secondary connections
          if not is_primary and prediction_data:
            last_sent_timestamp = prediction_data.get("_timestamp", 0)

          consecutive_errors = 0
          await asyncio.sleep(self.sleep_interval)

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

  async def realtime_anomaly(self) -> None:
    """Main realtime anomaly detection loop"""
    await self._run_inference_loop(
      inference_func=Anomaly.anomaly,
      check_model_exists=True,
      model_path="anomaly.pkl"
    )

  async def realtime_regression(self) -> None:
    """Main realtime regression inference loop"""
    await self._run_inference_loop(
      inference_func=Supervised.auto_inference,
      check_model_exists=False,
      # model_path="model.pkl"
    )

  async def realtime_clustering(self) -> None:
    """Main realtime clustering inference loop"""
    await self._run_inference_loop(
      inference_func=Unsupervised.auto_inference,
      check_model_exists=False
    )

  async def realtime_forecast(self) -> None:
    """Main realtime time series forecast loop dengan actual value fetching"""
    if not self.dataset:
      await self._close_safe(1000, "Dataset not initialized")
      return

    logger = Logger(self.dataset.name)

    if not self.dataset.task_type.is_timeseries():
      await self._close_safe(1000, "Dataset is not TimeSeries type")
      return

    is_primary = await stream_state.add_connection(self.dataset_name, self.connection_id)

    try:
      self._is_running = True
      consecutive_errors = 0
      max_consecutive_errors = 5
      last_sent_timestamp = 0  # Track timestamp terakhir yang dikirim

      while self._is_running:
        try:
          if TimeSeries.is_expired(self.dataset, logger):
            await self._send_with_retry({
              "status": "updating",
              "message": "Dataset expired, updating...",
              "timestamp": datetime.now().isoformat()
            })
            await asyncio.sleep(5)
            continue

          if is_primary:
            # Jalankan inference di thread terpisah agar bisa di-cancel
            try:
              result = await asyncio.wait_for(
                asyncio.to_thread(TimeSeries.inference, self.dataset, logger),
                timeout=60.0  # Timeout 60 detik untuk inference
              )
              prediction_data = result if isinstance(result, dict) else result.model_dump(mode="json")
            except asyncio.TimeoutError:
              LOGGER_GLOBAL.warning(f"[{self.dataset_name}] Forecast inference timeout after 60s")
              consecutive_errors += 1
              if consecutive_errors >= max_consecutive_errors:
                break
              await asyncio.sleep(1)
              continue
            except asyncio.CancelledError:
              LOGGER_GLOBAL.info(f"[{self.dataset_name}] Forecast inference cancelled")
              raise
            await stream_state.update_prediction(self.dataset_name, prediction_data)

            # Start actual value fetching task if timestamps exist
            if isinstance(prediction_data, dict) and "timestamps" in prediction_data:
              timestamps = prediction_data["timestamps"]
              LOGGER_GLOBAL.info(f"[{self.dataset_name}] Forecast timestamps count: {len(timestamps) if timestamps else 0}")
              if timestamps and self.dataset.features:
                feature = self.dataset.features[0]
                LOGGER_GLOBAL.info(f"[{self.dataset_name}] Starting actual value fetching for feature: {feature}")
                asyncio.create_task(
                  self._send_actual_values(feature, timestamps, logger)
                )
          else:
            # Secondary connection: tunggu sampai ada prediksi baru
            cached = None
            wait_iterations = 0
            max_wait_iterations = int(self.sleep_interval / 0.1)  # Tunggu max sleep_interval
            
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
              # Tidak ada data baru, skip cycle ini
              await asyncio.sleep(0.5)
              continue
            
            prediction_data = cached

          success = await self._send_with_retry(prediction_data)

          if not success:
            LOGGER_GLOBAL.error("Failed to send data after retries")
            break

          # Update timestamp untuk secondary connections
          if not is_primary and prediction_data:
            last_sent_timestamp = prediction_data.get("_timestamp", 0)

          consecutive_errors = 0
          await asyncio.sleep(self.sleep_interval)

        except asyncio.CancelledError:
          LOGGER_GLOBAL.info(f"[{self.dataset_name}] Forecast task cancelled")
          raise

        except Exception as e:
          consecutive_errors += 1
          LOGGER_GLOBAL.error(f"[{self.dataset_name}] Forecast error ({consecutive_errors}/{max_consecutive_errors}): {e}")

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
      await self._close_safe()

  async def _send_actual_values(self, feature: str, timestamps: List, logger) -> None:
    """Send actual values for each forecast timestamp"""
    LOGGER_GLOBAL.info(f"[{self.dataset_name}] _send_actual_values started with {len(timestamps)} timestamps, feature: {feature}")
    try:
      for idx, ts in enumerate(timestamps):
        if not self._is_running:
          LOGGER_GLOBAL.info(f"[{self.dataset_name}] _send_actual_values stopped - connection not running")
          break

        # Parse timestamp if it's a string
        if isinstance(ts, str):
          try:
            ts = DTEncoder.str_to_dt(ts)
          except ValueError as e:
            LOGGER_GLOBAL.warning(f"[{self.dataset_name}] Failed to parse timestamp {ts}: {e}")
            continue

        now = datetime.now()

        if now > ts:
          LOGGER_GLOBAL.info(f"[{self.dataset_name}] Timestamp {idx} ({ts}) already passed, skipping")
          continue

        # Calculate delay and wait
        delay = (ts - now).total_seconds()
        LOGGER_GLOBAL.info(f"[{self.dataset_name}] Waiting {delay:.2f} seconds for timestamp {idx} ({ts})")
        if delay > 0:
          await asyncio.sleep(delay)

        # Fetch actual value
        try:
          LOGGER_GLOBAL.info(f"[{self.dataset_name}] Fetching actual value for {feature} at {ts}")
          value = await async_get_realtime(feature, logger=logger)
          LOGGER_GLOBAL.info(f"[{self.dataset_name}] Got actual value: {value}")
          if value is not None:
            data = {
              "timestamp": DTEncoder.now().isoformat(),
              "actual": float(value)
            }
            LOGGER_GLOBAL.info(f"[{self.dataset_name}] Sending actual data: {data}")
            success = await self._send_with_retry(data)
            if success:
              LOGGER_GLOBAL.info(f"[{self.dataset_name}] Successfully sent actual value")
            else:
              LOGGER_GLOBAL.error(f"[{self.dataset_name}] Failed to send actual value")
          else:
            LOGGER_GLOBAL.warning(f"[{self.dataset_name}] Got None value from async_get_realtime")
        except Exception as e:
          LOGGER_GLOBAL.warning(f"[{self.dataset_name}] Failed to fetch actual value for {feature}: {e}")

    except Exception as e:
      LOGGER_GLOBAL.error(f"[{self.dataset_name}] Error in actual values loop: {e}")

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

  def stop(self):
    """Stop streaming loop"""
    self._is_running = False


class BaseWebSocketManager:
  """Base manager untuk mengelola lifecycle WebSocket connections"""

  def __init__(self, stream_method: str):
    self.stream_method = stream_method
    self._tasks: Dict[str, asyncio.Task] = {}

  async def handle_connection(
    self,
    websocket: WebSocket,
    dataset_name: str,
    db: Session,
    sleep_interval: float = 3.0
  ):
    """Handle WebSocket connection lifecycle"""
    streamer = Streamer(
      websocket=websocket,
      dataset_name=dataset_name,
      db=db,
      sleep_interval=sleep_interval
    )

    if not await streamer.setup():
      return

    stream_method = getattr(streamer, self.stream_method)
    stream_task = asyncio.create_task(stream_method())
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


class AnomalyWebSocketManager(BaseWebSocketManager):
  """Manager untuk anomaly detection WebSocket"""
  def __init__(self):
    super().__init__("realtime_anomaly")


class RegressionWebSocketManager(BaseWebSocketManager):
  """Manager untuk regression WebSocket"""
  def __init__(self):
    super().__init__("realtime_regression")


class ClusteringWebSocketManager(BaseWebSocketManager):
  """Manager untuk clustering WebSocket"""
  def __init__(self):
    super().__init__("realtime_clustering")


class ForecastWebSocketManager(BaseWebSocketManager):
  """Manager untuk time series forecast WebSocket"""
  def __init__(self):
    super().__init__("realtime_forecast")


class LogFileHandler(FileSystemEventHandler):
  """Handler untuk memantau perubahan file log menggunakan watchdog"""

  def __init__(self, callback: Callable[[str], None]):
    self.callback = callback
    self._last_position = 0

  def on_modified(self, event):
    if event.is_directory:
      return
    if event.src_path.endswith("main.log"):
      self._read_new_lines(event.src_path)

  def _read_new_lines(self, file_path: str):
    """Baca baris baru dari file sejak posisi terakhir"""
    try:
      with open(file_path, "r", encoding="utf-8") as f:
        f.seek(self._last_position)
        new_lines = f.readlines()
        self._last_position = f.tell()

        for line in new_lines:
          line = line.strip()
          if line:
            parsed = parse_log_line(line)
            if parsed:
              self.callback(parsed)
    except Exception as e:
      LOGGER_GLOBAL.warning(f"Error reading log file: {e}")

  def reset_position(self):
    """Reset posisi pembacaan ke akhir file"""
    self._last_position = 0


class LoggerState:
  """Shared state untuk mengelola multiple WebSocket connections untuk log streaming"""

  def __init__(self):
    self._active_loggers: Dict[str, Set[str]] = {}
    self._observers: Dict[str, Observer] = {}
    self._handlers: Dict[str, LogFileHandler] = {}
    self._message_queues: Dict[str, asyncio.Queue] = {}
    self._lock = asyncio.Lock()

  async def add_connection(self, dataset_name: str, connection_id: str) -> bool:
    """Tambahkan koneksi baru. Return True jika ini adalah koneksi pertama (harus mulai watchdog)"""
    async with self._lock:
      if dataset_name not in self._active_loggers:
        self._active_loggers[dataset_name] = set()
        self._message_queues[dataset_name] = asyncio.Queue()
        self._active_loggers[dataset_name].add(connection_id)
        return True
      self._active_loggers[dataset_name].add(connection_id)
      return False

  async def remove_connection(self, dataset_name: str, connection_id: str) -> bool:
    """Hapus koneksi. Return True jika tidak ada koneksi lagi (harus stop watchdog)"""
    async with self._lock:
      if dataset_name in self._active_loggers:
        self._active_loggers[dataset_name].discard(connection_id)
        if not self._active_loggers[dataset_name]:
          del self._active_loggers[dataset_name]
          if dataset_name in self._message_queues:
            del self._message_queues[dataset_name]
          return True
      return False

  def get_message_queue(self, dataset_name: str) -> Optional[asyncio.Queue]:
    """Get message queue untuk dataset"""
    return self._message_queues.get(dataset_name)

  def add_observer(self, dataset_name: str, observer: Observer, handler: LogFileHandler):
    """Simpan observer dan handler untuk dataset"""
    self._observers[dataset_name] = observer
    self._handlers[dataset_name] = handler

  def remove_observer(self, dataset_name: str):
    """Hapus observer dan handler untuk dataset"""
    if dataset_name in self._observers:
      self._observers[dataset_name].stop()
      self._observers[dataset_name].join()
      del self._observers[dataset_name]
    if dataset_name in self._handlers:
      del self._handlers[dataset_name]

  def get_connection_count(self, dataset_name: str) -> int:
    """Get jumlah koneksi aktif untuk dataset"""
    return len(self._active_loggers.get(dataset_name, set()))


# Global logger state instance
logger_state = LoggerState()


class LogStreamer:
  """Handler untuk WebSocket log streaming dengan watchdog file monitoring"""

  def __init__(
    self,
    websocket: WebSocket,
    dataset_name: str,
    require_dataset: bool = True,
    max_lines: int = 200,
    heartbeat_interval: float = 30.0
  ):
    self.websocket = websocket
    self.dataset_name = dataset_name
    self.require_dataset = require_dataset
    self.max_lines = max_lines
    self.heartbeat_interval = heartbeat_interval
    self.connection_id = f"{id(websocket)}_{time.time()}"
    self._is_running = False
    self._last_activity = time.time()
    self._log_file_path = self._get_log_file_path()

  def _get_log_file_path(self) -> Optional[pathlib.Path]:
    """Get path file log berdasarkan dataset_name"""
    if self.dataset_name == "system":
      return Config.dir / "logger" / "main.log"
    else:
      return Config.dir / "storages" / self.dataset_name / "logs" / "main.log"

  async def setup(self) -> bool:
    """Setup koneksi WebSocket dan validasi log file"""
    await self.websocket.accept()

    try:
      if self.require_dataset:
        dataset = Dataset.get_by_name(name=self.dataset_name, db=get_session())
        if not dataset:
          LOGGER_GLOBAL.warning(f"Dataset {self.dataset_name} is not found!")
          await self._close_safe(1000, "Dataset is not found")
          return False

      # Pastikan file log ada
      if not self._log_file_path or not self._log_file_path.exists():
        LOGGER_GLOBAL.warning(f"Log file not found for {self.dataset_name}")
        await self.websocket.send_json({
          "type": "init",
          "status": "no_logs",
          "message": f"No logs found for {self.dataset_name}",
          "data": []
        })
      else:
        LOGGER_GLOBAL.info(f"WebSocket connected for logs: {self.dataset_name}")

      self._last_activity = time.time()
      return True

    except Exception as e:
      LOGGER_GLOBAL.error(f"Log WebSocket setup error: {str(e)}")
      await self._close_safe(1011, str(e))
      return False

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
        await self.websocket.send_json(data)
        self._last_activity = time.time()
        return True
      except Exception as e:
        wait_time = min(2 ** attempt, 8)
        LOGGER_GLOBAL.warning(f"Log send failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
        if attempt == max_retries - 1:
          await asyncio.sleep(wait_time)
    return False

  async def _send_initial_logs(self):
    """Kirim log history (max 200 baris) saat pertama kali connect"""
    try:
      logs = read_logs(self.dataset_name)

      # Ambil max 200 baris terakhir
      if len(logs) > self.max_lines:
        logs = logs[-self.max_lines:]

      await self._send_with_retry({
        "type": "init",
        "status": "success",
        "dataset": self.dataset_name,
        "count": len(logs),
        "data": logs
      })

    except Exception as e:
      LOGGER_GLOBAL.error(f"Error sending initial logs: {e}")
      await self._send_with_retry({
        "type": "init",
        "status": "error",
        "message": str(e),
        "data": []
      })

  def _on_log_file_change(self, parsed_line: dict):
    """Callback saat file log berubah"""
    queue = logger_state.get_message_queue(self.dataset_name)
    if queue:
      try:
        # put_nowait is thread-safe, works from watchdog's thread
        queue.put_nowait(parsed_line)
      except asyncio.QueueFull:
        LOGGER_GLOBAL.warning(f"Log queue full for {self.dataset_name}, dropping message")
      except Exception as e:
        LOGGER_GLOBAL.warning(f"Error queueing log message: {e}")

  async def _start_file_watcher(self) -> bool:
    """Start watchdog observer untuk memantau file log"""
    try:
      if not self._log_file_path or not self._log_file_path.exists():
        return False

      log_dir = self._log_file_path.parent

      handler = LogFileHandler(callback=self._on_log_file_change)

      # Set posisi awal ke akhir file untuk hanya mengambil log baru
      try:
        with open(self._log_file_path, "r", encoding="utf-8") as f:
          f.seek(0, 2)  # Seek to end
          handler._last_position = f.tell()
      except Exception:
        handler._last_position = 0

      observer = Observer()
      observer.schedule(handler, str(log_dir), recursive=False)
      observer.start()

      logger_state.add_observer(self.dataset_name, observer, handler)

      LOGGER_GLOBAL.info(f"Started file watcher for {self.dataset_name}")
      return True

    except Exception as e:
      LOGGER_GLOBAL.error(f"Error starting file watcher: {e}")
      return False

  async def _receive_heartbeat(self) -> bool:
    """Check apakah client masih aktif dengan menerima pesan"""
    try:
      message = await asyncio.wait_for(
        self.websocket.receive_text(),
        timeout=self.heartbeat_interval
      )
      self._last_activity = time.time()
      if message == "ping":
        await self.websocket.send_text("pong")
      return True
    except asyncio.TimeoutError:
      return True
    except WebSocketDisconnect:
      return False
    except Exception as e:
      LOGGER_GLOBAL.warning(f"Log heartbeat error: {e}")
      return False

  async def run(self) -> None:
    """Main log streaming loop"""
    is_primary = await logger_state.add_connection(self.dataset_name, self.connection_id)

    try:
      self._is_running = True

      # Kirim initial logs terlebih dahulu
      await self._send_initial_logs()

      # Start file watcher jika ini adalah koneksi pertama
      if is_primary:
        watcher_started = await self._start_file_watcher()
        if not watcher_started and self._log_file_path and self._log_file_path.exists():
          LOGGER_GLOBAL.warning(f"Could not start file watcher for {self.dataset_name}")

      message_queue = logger_state.get_message_queue(self.dataset_name)
      consecutive_errors = 0
      max_consecutive_errors = 5

      while self._is_running:
        try:
          # Cek apakah ada log baru di queue
          log_entry = None
          if message_queue:
            try:
              log_entry = message_queue.get_nowait()
            except asyncio.QueueEmpty:
              pass

          if log_entry:
            success = await self._send_with_retry({
              "type": "log",
              "data": log_entry,
              "timestamp": datetime.now().isoformat()
            })

            if not success:
              LOGGER_GLOBAL.error("Failed to send log after retries")
              break

            consecutive_errors = 0
          else:
            # Tidak ada log baru, tunggu sebentar
            await asyncio.sleep(0.5)

        except asyncio.CancelledError:
          LOGGER_GLOBAL.info(f"[{self.dataset_name}] Log stream task cancelled")
          raise

        except Exception as e:
          consecutive_errors += 1
          LOGGER_GLOBAL.error(f"[{self.dataset_name}] Log error ({consecutive_errors}/{max_consecutive_errors}): {e}")

          if consecutive_errors >= max_consecutive_errors:
            await self._send_with_retry({
              "type": "error",
              "error": "Too many consecutive errors",
              "timestamp": datetime.now().isoformat()
            })
            break

          await asyncio.sleep(1)

    finally:
      self._is_running = False
      should_stop_watcher = await logger_state.remove_connection(self.dataset_name, self.connection_id)

      if should_stop_watcher:
        logger_state.remove_observer(self.dataset_name)
        LOGGER_GLOBAL.info(f"Stopped file watcher for {self.dataset_name}")

      await self._close_safe()

  async def handle_client_messages(self) -> None:
    """Handle pesan dari client (untuk keep-alive dan control)"""
    try:
      while self._is_running:
        if not await self._receive_heartbeat():
          break
    except WebSocketDisconnect:
      LOGGER_GLOBAL.info(f"[{self.dataset_name}] Log client disconnected")
    except Exception as e:
      LOGGER_GLOBAL.warning(f"[{self.dataset_name}] Log message handler error: {e}")
    finally:
      self._is_running = False

  def stop(self):
    """Stop streaming loop"""
    self._is_running = False


class LogWebSocketManager:
  """Manager untuk mengelola lifecycle WebSocket log connections"""

  def __init__(self):
    self._tasks: Dict[str, asyncio.Task] = {}

  async def handle_connection(
    self,
    websocket: WebSocket,
    dataset_name: str,
    require_dataset: bool = False,
    max_lines: int = 200
  ):
    """Handle WebSocket connection lifecycle untuk log streaming"""
    streamer = LogStreamer(
      websocket=websocket,
      dataset_name=dataset_name,
      require_dataset=require_dataset,
      max_lines=max_lines
    )

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
          LOGGER_GLOBAL.info(f"[{dataset_name}] Log task cancelled successfully")
        except asyncio.TimeoutError:
          LOGGER_GLOBAL.warning(f"[{dataset_name}] Log task cancellation timeout, forcing...")
          # Force cancel jika timeout
          if not task.done():
            task.cancel()

      if stream_task in done:
        try:
          stream_task.result()
        except asyncio.CancelledError:
          LOGGER_GLOBAL.info(f"[{dataset_name}] Log stream task cancelled normally")
        except Exception as e:
          LOGGER_GLOBAL.error(f"Log stream task error: {e}")

    except asyncio.CancelledError:
      LOGGER_GLOBAL.info(f"[{dataset_name}] Log connection handler cancelled")
      raise

    except Exception as e:
      LOGGER_GLOBAL.error(f"Log connection handler error: {e}")

    finally:
      streamer.stop()
      LOGGER_GLOBAL.info(f"[{dataset_name}] Log connection closed")


# Global manager instances
anomaly_manager = AnomalyWebSocketManager()
regression_manager = RegressionWebSocketManager()
clustering_manager = ClusteringWebSocketManager()
forecast_manager = ForecastWebSocketManager()
logger_manager = LogWebSocketManager()
