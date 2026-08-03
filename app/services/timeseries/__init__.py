"""Time-series (forecasting) — orkestrasi lengkap satu task.

Komputasi ML-nya murni di `app.core.time_series`; paket ini yang Dataset/DB/
InfluxDB-aware. Dipecah menurut apa yang dikerjakan, dengan arah impor satu
jalur supaya tidak ada lingkaran:

    horizon.py   umur forecast (dasar — tidak mengimpor sibling mana pun)
       ↑  ↑  ↑
       │  │  └── actuals.py   back-fill nilai aktual ke InfluxDB
       │  └───── retrain.py   refit: `retrain` (worker) & `finetune` (route/stream)
       └──────── forecast.py  read path + siklus auto-inference
                 worker.py    entry pm2: menyusuri horizon, tulis actual per titik

Modul ini memaparkan permukaan lama `app.services.timeseries` apa adanya, jadi
pemanggil (route, dispatch, streamer, tasks) tidak perlu tahu pembagian di atas.
"""
from app.services.timeseries.actuals import get_actual_save, get_actual_save_all
from app.services.timeseries.forecast import auto_inference, forecast
from app.services.timeseries.horizon import (
  _current_dt,
  _horizon_minutes,
  elapsed_horizon_ratio,
  is_expired,
)
from app.services.timeseries.retrain import finetune, retrain
from app.services.timeseries.worker import auto_inference_write_loop, main

__all__ = [
  "is_expired", "elapsed_horizon_ratio", "forecast", "auto_inference",
  "get_actual_save", "get_actual_save_all", "retrain", "finetune",
  "auto_inference_write_loop", "main",
  "_current_dt", "_horizon_minutes",
]
