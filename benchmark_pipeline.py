"""
Benchmark pipeline end-to-end: create dataset -> pull -> find top model.

Mengukur:
  1. Waktu pull PER POINT (per tagname/feature)   -> dari wrapper get_history
  2. Waktu pull SEMUA POINT (total)               -> meta.pulling_time + jumlah wrapper
  3. Waktu FIND TOP MODEL (training/compare)      -> diukur di sekitar find_top_model

Jalur yang dipakai sama persis dengan produksi (app.routes.modelML.auto_initialize):
  pulling(name, db) -> core.find_top_model(dataset, n_models, logger, db)

Cara pakai:
  # menembak API Schneider sungguhan (butuh jaringan):
  PYTHONPATH=. .venv/bin/python benchmark_pipeline.py --days 3 --n-models 2

  # mode sintetis (tanpa jaringan; pull disimulasikan, training tetap pycaret asli):
  PYTHONPATH=. .venv/bin/python benchmark_pipeline.py --fake-sl --fake-latency 0.05

Catatan: mode asli menembak API Schneider (butuh jaringan) dan menjalankan
pycaret compare_models sungguhan (bisa beberapa menit).
"""
import argparse, time, datetime, statistics, traceback, hashlib
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

import app.pull as pullmod
from app.pull import pulling,query_tagnames
from app.helpers import DTEncoder
from app.database.schemas import (
    DatasetRequestSchema, PreprocessingSchema, StatusProcess,
)
from app.database.db import get_session
from app.database.orm import Dataset
from app.routes.dataset import create_dataset, delete_dataset
from app.logger import Logger


# --------------------------------------------------------------------------- #
# Instrumentasi: bungkus get_history supaya tiap panggilan tercatat per-kolom.
# pull_history memanggil get_history(col, day, ...) sekali per (kolom, hari),
# jadi jumlah panggilan per kolom = jumlah hari. get_row_id (lookup row_id dari
# tagname) terjadi di dalam get_history, jadi biayanya ikut terukur - realistis.
# --------------------------------------------------------------------------- #
_orig_get_history = pullmod.get_history
_timings: dict[str, list[float]] = defaultdict(list)
_FAKE_LATENCY = 0.0


def _seeded(col: str, day: str) -> np.random.Generator:
  h = int(hashlib.md5(f"{col}".encode()).hexdigest()[:8], 16)
  return np.random.default_rng(h)


def _fake_get_history(tagname, current_date, time_start="00:00:00",
                      time_end="23:59:00", interval=1, to_dataframe=False,
                      logger=None, **kwargs):
  """Data sintetis berbentuk sama dengan get_history asli: kolom [dt, tagname],
  grid 5 menit sepanjang hari. Nilai = random-walk deterministik per kolom
  supaya target punya sinyal untuk diregresi. Latensi disimulasikan agar angka
  'waktu per point' bermakna tanpa jaringan."""
  if _FAKE_LATENCY:
    time.sleep(_FAKE_LATENCY)
  base = DTEncoder.str_to_dt(current_date)
  h0, m0, _ = (int(x) for x in time_start.split(":"))
  h1, m1, _ = (int(x) for x in time_end.split(":"))
  start = base.replace(hour=h0, minute=m0)
  end = base.replace(hour=h1, minute=m1)
  idx = pd.date_range(start, end, freq="5min", tz="UTC")
  rng = _seeded(str(tagname), current_date)
  # random walk stabil per kolom lintas hari (seed by column) + drift harian
  vals = 20 + np.cumsum(rng.normal(0, 0.15, size=len(idx))) + rng.normal(0, 0.3, size=len(idx))
  df = pd.DataFrame({"dt": idx, tagname: vals.astype(float)})
  if to_dataframe:
    return df
  return [list(df["dt"]), list(df[tagname])]

def get_tagnames(n:Optional[int] = None):
  tagnames = query_tagnames("CRAH-2DH2.1-")
  tagnames.remove("CRAH-2DH2.1-MODE")
  if n: return tagnames[:n]
  else: return tagnames

def parse_args():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--task-type", default="Regression",
                 help="Regression | Classification (supervised)")
  p.add_argument("--days", type=int, default=3, help="rentang hari ke belakang")
  p.add_argument("--n-models", type=int, default=2, help="top N model")
  p.add_argument("--features", nargs="+", default=[
      "CRAH-2DH2.1-SUPPLY_AIR_TEMP",
      "CRAH-2DH2.2-SUPPLY_AIR_TEMP",
      "CRAH-2DH2.3-SUPPLY_AIR_TEMP",
  ])
  p.add_argument("--target", default="CRAH-2DH2.1-RETURN_AIR_TEMP")
  p.add_argument("--fake-sl", action="store_true",
                 help="pakai data sintetis, tidak menembak API (untuk uji tanpa jaringan)")
  p.add_argument("--fake-latency", type=float, default=0.05,
                 help="latensi simulasi per panggilan get_history di mode --fake-sl (detik)")
  p.add_argument("--keep", action="store_true",
                 help="jangan hapus dataset setelah benchmark")
  return p.parse_args()


def fmt(sec: float) -> str:
  return f"{sec:8.3f}s" if sec < 60 else f"{sec/60:7.2f}m"


def main():
  global _FAKE_LATENCY
  args = parse_args()

  # aktifkan timer; di mode fake, ganti sumber datanya juga.
  if args.fake_sl:
    _FAKE_LATENCY = args.fake_latency
    src = _fake_get_history
  else:
    src = _orig_get_history

  def timed(tagname, *a, **k):
    t0 = time.perf_counter()
    try:
      return src(tagname, *a, **k)
    finally:
      _timings[str(tagname)].append(time.perf_counter() - t0)

  pullmod.get_history = timed

  end_date = DTEncoder.now()
  start_date = end_date - datetime.timedelta(days=args.days)

  tagnames = get_tagnames(10)

  payload = DatasetRequestSchema(
      description="benchmark",
      task_type="Clustering",
      features=tagnames,
      target=3,
      start_date=DTEncoder.dt_to_str(start_date),
      end_date=DTEncoder.dt_to_str(end_date),
      time_start="00:00:00",
      time_end="23:59:00",
      interval=0,
      preprocessing=PreprocessingSchema(),
  )

  db = next(get_session())
  dataset = create_dataset(payload, db)
  dataset.n_models = args.n_models
  dataset.save(db)
  db.commit()
  name = dataset.name

  mode = "FAKE-SL (sintetis)" if args.fake_sl else "REAL SL API"
  print("=" * 68)
  print(f" BENCHMARK PIPELINE  |  {name}  |  {mode}")
  print(f" task={args.task_type}  days={args.days}  n_models={args.n_models}")
  print(f" features={len(args.features)}  target={args.target}")
  print("=" * 68)

  # inisialisasi supaya blok finally aman meski pull gagal
  pull_ok = False
  pull_wall = 0.0
  pulling_time = 0.0
  n_rows = n_cols = "?"
  train_wall = None
  train_meta = None
  try:
    # ---------------- 1) PULL (create dataset -> data.csv) ----------------
    t0 = time.perf_counter()
    pull_ok = pulling(name, db)
    pull_wall = time.perf_counter() - t0

    dataset = Dataset.get_by_name(name, db)
    if not pull_ok or dataset.status != StatusProcess.SUCCESS_PULL:
      print(f"\n[!] PULL GAGAL - status: {dataset.status}")
      print(f"    (cek storages/{name}/logs/main.log; kemungkinan jaringan SL)")
      return

    meta = dataset.meta or {}
    n_rows = meta.get("n_rows", "?")
    n_cols = meta.get("n_cols", "?")
    pulling_time = meta.get("pulling_time", pull_wall)

    # ---------------- 2) FIND TOP MODEL (training) ------------------------
    logger = Logger(name)
    core = dataset.task_type.core()
    t0 = time.perf_counter()
    core.find_top_model(dataset, args.n_models, logger, db)
    train_wall = time.perf_counter() - t0
    dataset = Dataset.get_by_name(name, db)
    train_meta = (dataset.meta or {}).get("train_time", train_wall)

  finally:
    # ------------------------------ REPORT --------------------------------
    print("\n" + "-" * 68)
    print(" 1) WAKTU PULL PER POINT")
    print("-" * 68)
    print(f" {'tagname':<38}{'calls':>6}{'total':>11}{'avg/call':>12}")
    per_point_totals = {}
    for col, samples in _timings.items():
      tot = sum(samples)
      per_point_totals[col] = tot
      avg = statistics.mean(samples) if samples else 0.0
      print(f" {col:<38}{len(samples):>6}{fmt(tot):>11}{fmt(avg):>12}")

    print("-" * 68)
    print(" 2) WAKTU PULL SEMUA POINT")
    print("-" * 68)
    if per_point_totals:
      sum_points = sum(per_point_totals.values())
      slowest = max(per_point_totals, key=per_point_totals.get)
      fastest = min(per_point_totals, key=per_point_totals.get)
      print(f" Jumlah waktu per-point (serial)      : {fmt(sum_points)}")
      print(f" pulling_time (meta, incl. merge+prep): {fmt(pulling_time)}")
      print(f" wall-clock pulling()                 : {fmt(pull_wall)}")
      print(f" terlama : {slowest}  ({fmt(per_point_totals[slowest])})")
      print(f" tercepat: {fastest}  ({fmt(per_point_totals[fastest])})")
      print(f" rows={n_rows}  cols={n_cols}")
    else:
      print(" (tidak ada panggilan get_history tercatat)")

    print("-" * 68)
    print(" 3) WAKTU FIND TOP MODEL")
    print("-" * 68)
    if train_wall is not None:
      print(f" find_top_model wall-clock            : {fmt(train_wall)}")
      print(f" train_time (meta)                    : {fmt(train_meta)}")
      dataset = Dataset.get_by_name(name, db)
      for m in (dataset.models or []):
        print(f"   - model: {m.algorithm}")
      print(f" top_model: {dataset.top_model}  status: {dataset.status}")
    else:
      print(" (dilewati karena pull gagal)")
    print("=" * 68)

    if not args.keep:
      try:
        delete_dataset(name, db)
        db.commit()
        print(f"[cleanup] dataset {name} dihapus (pakai --keep untuk menyimpan)")
      except Exception as e:
        print(f"[cleanup] gagal hapus {name}: {e}")
    else:
      print(f"[keep] dataset {name} disimpan")
    db.close()


if __name__ == "__main__":
  try: main()
  except Exception: traceback.print_exc()
