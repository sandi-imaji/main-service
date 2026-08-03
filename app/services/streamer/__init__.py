"""Stream realtime lewat WebSocket.

Titik rakit paket ini: menentukan fungsi inference mana yang dipakai tiap
stream, lalu mengekspos satu manager per endpoint untuk `app.routes.stream`.

    routes/stream.py  →  <task>_manager.handle_connection(...)
                              │
                              ├─ manager.py   lifecycle koneksi
                              ├─ base.py      kirim/terima + kebijakan poll SL
                              ├─ inference.py loop supervised/anomaly/clustering
                              ├─ forecast.py  loop forecast + perintah get_actual
                              └─ logs.py      stream isi file log
"""
from app.services import clustering as clustering_service
from app.services import dispatch as dispatch_service
from app.services.streamer.base import (
  BaseStreamer,
  DEGRADED_RETRY_INTERVAL,
  MAX_POLL_FAILURES,
)
from app.services.streamer.forecast import ForecastStreamer
from app.services.streamer.inference import InferenceStreamer
from app.services.streamer.logs import LogStreamer, LogWebSocketManager, logger_state
from app.services.streamer.manager import (
  BaseWebSocketManager,
  ForecastWebSocketManager,
  InferenceWebSocketManager,
)
from app.services.streamer.state import StreamState, stream_state

# Manager per endpoint. Anomaly satu-satunya yang menunjuk file model: modelnya
# tidak punya baris ModelML, jadi keberadaannya hanya bisa dicek dari disk.
anomaly_manager = InferenceWebSocketManager(
    inference_func=dispatch_service.auto_inference, model_path="anomaly.pkl")
regression_manager = InferenceWebSocketManager(
    inference_func=dispatch_service.auto_inference)
clustering_manager = InferenceWebSocketManager(
    inference_func=clustering_service.auto_inference)
forecast_manager = ForecastWebSocketManager()
logger_manager = LogWebSocketManager()

__all__ = [
  # manager siap pakai (dipakai routes/stream.py)
  "anomaly_manager", "regression_manager", "clustering_manager",
  "forecast_manager", "logger_manager",
  # kelas & state, untuk test dan komposisi lain
  "BaseStreamer", "InferenceStreamer", "ForecastStreamer", "LogStreamer",
  "BaseWebSocketManager", "InferenceWebSocketManager", "ForecastWebSocketManager",
  "LogWebSocketManager", "StreamState", "stream_state", "logger_state",
  "MAX_POLL_FAILURES", "DEGRADED_RETRY_INTERVAL",
]
