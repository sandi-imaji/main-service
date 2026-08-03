"""Plumbing worker pm2 yang dipakai bersama tiap task.

Tiap task punya file entry-nya sendiri — pm2 menjalankan file itu langsung, dan
file-nyalah yang menentukan identitas worker (lihat `WorkerManager.get_script_path`).
Yang sama di semua task hanya kerangkanya: ambil nama dataset dari argv, buka
sesi DB, lalu jalankan loop yang tidak boleh mati gara-gara satu tick gagal.
"""
import time
import traceback
from typing import Callable, Optional

from app.database.historicalDB import get_influx_storage


def influx_write_loop(dataset, logger, infer: Callable,
                      before_tick: Optional[Callable] = None) -> None:
  """Loop worker: (before_tick) → infer → tulis ke InfluxDB → tidur.

  `before_tick(dataset, logger)` adalah pekerjaan berkala di luar inference —
  dipakai supervised untuk mengecek jadwal finetune. Task yang tidak punya
  urusan seperti itu cukup tidak mengisinya.

  Tahan banting: kegagalan satu iterasi dicatat lalu loop lanjut di tick
  berikutnya, dan setiap iterasi selalu tidur (tidak ada busy-spin).
  """
  logger.info("start auto inference write loop ...")
  while True:
    try:
      if before_tick:
        before_tick(dataset, logger)

      results = infer(dataset, logger)
      if results is not None and results.is_valid:
        logger.info("Writing to InfluxDB...")
        try:
          results.write_to_influx(get_influx_storage(), logger)
        except Exception as e:
          logger.error(f"Failed to write to InfluxDB: {e}")
          logger.error(traceback.format_exc())
    except Exception as e:
      logger.error(f"auto inference loop error: {e}")

    time.sleep(dataset.interval * 60)


def run_from_argv(write_loop: Callable) -> None:
  """Entry point worker: `python <file> <dataset_name>`.

  Sesi DB sengaja dibiarkan terbuka selama loop hidup: `dataset` adalah objek
  ORM, dan relasi seperti `dataset.models` di-load lazy saat dipakai.
  """
  import sys
  if len(sys.argv) <= 1:
    print("Dataset name is None")
    return

  # diimpor di sini (bukan di atas) supaya modul ini tetap ringan saat dipakai
  # sebagai library — DB hanya dibutuhkan untuk bootstrap worker.
  from app.database.base import get_db_session
  from app.database.DB import Dataset
  from app.logger import LogManager

  dataset_name = sys.argv[1]
  # channel `worker`: riwayat proses pm2 punya filenya sendiri, terpisah dari
  # log train/pulling milik dataset yang sama.
  logger = LogManager.worker(dataset_name)
  with get_db_session() as db:
    dataset = Dataset.get_by_name(dataset_name, db)   # raise NotFound kalau tidak ada
    write_loop(dataset, logger)
