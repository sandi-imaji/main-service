from app.config import Config
import os
import datetime
import pandas as pd
import numpy as np
import re
import requests as req
from typing import List, Optional, Union, Dict, Any
from sklearn.decomposition import PCA


def get_row_id(tagname: str,logger=None) -> str:
  url = Config.sl_url("application/api/modbus/get-point")
  data = req.post(url=url, params=dict(tagname=tagname),
                  verify=Config.sl_verify, data=dict(key=Config.sl_key),
                  timeout=Config.sl_timeout_long)
  if data.status_code != 200: return ""
  else:
    data_json = data.json()
    message = data_json.get("message","")
    if not message:
      if logger: logger.error(f"Get point no message : {data.text} | {tagname}")
      return ""
    if message != "Success":
      if logger:logger.error(f"{message} | {tagname}")
      return ""
    return data_json["data"]["row_id"]


class DTEncoder:
  """Kelas helper untuk konversi antara datetime, string, dan unix timestamp (ms)."""

  FMT_DT = "%Y%m%d"
  UTC = datetime.timezone.utc

  @staticmethod
  def time_ago(dt:datetime.datetime,now:Optional[datetime.datetime]=None):
    # pastikan dt punya timezone (recommended)
    if dt.tzinfo is None: dt = dt.replace(tzinfo=DTEncoder.UTC)
    if now is None: now = DTEncoder.now()
    diff = now - dt
    seconds = int(diff.total_seconds())

    if seconds < 60: return f"{seconds} seconds ago"
    elif seconds < 3600:
      minutes = seconds // 60
      return f"{minutes} minutes ago"
    elif seconds < 86400:
      hours = seconds // 3600
      return f"{hours} hours ago"
    else:
      days = seconds // 86400
      return f"{days} days ago"


  @staticmethod
  def get_start_datetime(last_dt: datetime.datetime, interval_minutes: int, n: int):
    """
    Menghitung start_dt dengan mundur dari last_dt sebanyak n kali interval

    Parameters:
      last_dt: datetime terakhir (waktu paling baru)
      interval_minutes: selisih waktu dalam menit (contoh: 5)
      n: berapa kali mundur

    Return: start_dt (datetime paling awal)
    """
    delta = datetime.timedelta(minutes=interval_minutes)

    # Mundur sebanyak n * interval
    start_dt = last_dt - (delta * n)

    return start_dt

  @staticmethod
  def get_end_datetime(last_dt: datetime.datetime, interval_minutes: int, n: int):
    """
    Menghitung end_datetime (datetime terakhir) dari sebuah start_dt
    dengan interval dan jumlah n

    Parameters:
        start_dt: datetime awal (paling lama)
        interval_minutes: selisih waktu dalam menit
        n: berapa kali interval (berapa langkah maju)

    Return:
        end_dt: datetime terakhir
    """
    delta = datetime.timedelta(minutes=interval_minutes)

    # Maju sebanyak n * interval
    end_dt = last_dt + (delta * n)

    return end_dt

  # --- DATETIME ⇄ STRING ---
  @staticmethod
  def dt_to_str(dt: datetime.datetime) -> str:
    """Ubah datetime → string 'YYYYMMDD'."""
    if not isinstance(dt, datetime.datetime):
      raise TypeError(f"dt harus datetime, bukan {type(dt)}")
    return dt.strftime(DTEncoder.FMT_DT)

  @staticmethod
  def str_to_dt(value: str) -> datetime.datetime:
    """
    Ubah string tanggal → datetime (UTC).
    Dukung beberapa format umum.
    """
    if not isinstance(value, str):
      raise TypeError("Input harus string.")

    patterns = [DTEncoder.FMT_DT, "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"]
    for p in patterns:
      try:
        dt = datetime.datetime.strptime(value, p)
        return dt.replace(tzinfo=DTEncoder.UTC)
      except ValueError:
        continue

    raise ValueError(f"Format tanggal tidak dikenali: {value}")

  # --- DATETIME ⇄ UNIX TIMESTAMP ---
  @staticmethod
  def dt_to_unixTS(dt: datetime.datetime) -> int:
    """
    Ubah datetime → unix timestamp dalam milidetik (int).
    """
    if not isinstance(dt, datetime.datetime):
      raise TypeError(f"dt harus datetime, bukan {type(dt)}")

    if dt.tzinfo is None:
      dt = dt.replace(tzinfo=DTEncoder.UTC)

    return int(dt.timestamp() * 1000)

  @staticmethod
  def unixTS_to_dt(unixTS) -> datetime.datetime:
    """
    Ubah unix timestamp (ms) → datetime (UTC).
    Bisa menerima int, float, atau string angka.
    """
    try: ts = int(unixTS) / 1000  # konversi milidetik ke detik
    except (ValueError, TypeError): raise TypeError("unixTS harus bisa dikonversi ke int atau float.")

    return datetime.datetime.fromtimestamp(ts, tz=DTEncoder.UTC) + datetime.timedelta(hours=7)

  @staticmethod
  def now() -> datetime.datetime: return datetime.datetime.now(tz=DTEncoder.UTC) + datetime.timedelta(hours=7)

  @staticmethod
  def now_utc() -> datetime.datetime: return datetime.datetime.now(tz=DTEncoder.UTC)
  
  @staticmethod
  def to_utc(dt:datetime.datetime): return dt - datetime.timedelta(hours=7)

  @staticmethod
  def from_utc(dt:Union[datetime.datetime,str]):
    if not isinstance(dt,datetime.datetime): dt = datetime.datetime.fromisoformat(dt)
    return dt + datetime.timedelta(hours=7)

  @staticmethod
  def now_ts() -> float:
    return DTEncoder.dt_to_unixTS(DTEncoder.now())

  @staticmethod
  def now_str() -> str:
    return DTEncoder.dt_to_str(DTEncoder.now())

  @staticmethod
  def generate_dt(
      n: int,
      interval_minutes: int,
      last_dt: Optional[datetime.datetime] = None,
      to_str: bool = False,
  ):
    if not last_dt:
      last_dt = DTEncoder.now()
    if n <= 0:
      raise ValueError("Parameter `n` harus lebih besar dari 0 ")
    delta = datetime.timedelta(minutes=interval_minutes)
    result = []
    current = last_dt
    for _ in range(n):
      current = current + delta
      if to_str:
        result.append(current.isoformat())
      else:
        result.append(current)
    return result

  @staticmethod
  def compare(dt1:datetime.datetime,dt2: datetime.datetime,tolerance:float = 1.5) -> bool:
    return abs(dt1.timestamp() - dt2.timestamp()) <= tolerance



def parse_log_line(line: str, with_module: bool = False) -> Dict[str, Any]:
  """
  Parse satu baris log menjadi dict yang sesuai dengan frontend LogViewer.
  Format log: [2025-01-01 12:00:00] [INFO] module_name:function_name:123 - message
  """
  line = line.strip()
  if not line:
    return {}

  # Format: [timestamp] [LEVEL] [dataset] module:function:line - message
  # `[dataset]` opsional supaya file log lama (tanpa field itu) tetap terbaca,
  # dan `function` boleh <module> karena log di level modul memakai nama itu.
  pattern = re.compile(
      r"\[(?P<timestamp>[^\]]+)\]\s+"
      r"\[(?P<level>[A-Z]+)\]\s+"
      r"(?:\[(?P<dataset>[^\]]*)\]\s+)?"
      r"(?P<module>[\w\.]+):(?P<function>[\w<>]+):(?P<line>\d+)\s+-\s+"
      r"(?P<message>.+)"
  )

  m = pattern.match(line)
  if not m:
    # Fallback kalau format tidak standar
    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "level": "INFO",
        "dataset": "",
        "module": "unknown",
        "function": "",
        "line": None,
        "message": line,
        "raw": line,
    }

  ts_str = m.group("timestamp").strip()
  try:
    timestamp = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").strftime(
        "%Y-%m-%d %H:%M:%S"
    )
  except ValueError:
    timestamp = ts_str  # fallback ke aslinya kalau format aneh

  level = m.group("level")
  # Map level untuk konsistensi frontend
  level_mapping = {
      "WARN": "WARNING",
      "SUCCESS": "SUCCESS",
      "DEBUG": "DEBUG",
      "ERROR": "ERROR",
      "INFO": "INFO",
      "CRITICAL": "CRITICAL",
  }
  level = level_mapping.get(level, level)

  return {
      "timestamp": timestamp,
      "level": level,
      "dataset": m.group("dataset") or "",
      "module": m.group("module"),
      "function": m.group("function"),
      "line": int(m.group("line")),
      "message": m.group("message").strip(),
      "raw": line,
  }


def read_logs(dataset_name: str, channel: str = "main") -> List[Dict[str, Any]]:
  """
  Digunakan untuk inisialisasi atau fallback kalau WebSocket belum connect.

  Lokasi filenya ditentukan LogManager (satu sumber kebenaran), jadi pembaca dan
  penulis log tidak bisa lagi menunjuk path yang berbeda.
  """
  from app.logger import LogManager
  fpath = LogManager.log_path(dataset_name, channel)

  if not fpath.exists(): return []

  logs = []
  with open(fpath, "r", encoding="utf-8") as f:
    for line in f:
      parsed = parse_log_line(line)
      if parsed: logs.append(parsed)
  return logs


def miss_val_handling_df(df: pd.DataFrame, logger, handling):
  miss = df.isna()
  miss_values = miss.sum().to_dict()
  logger.info("GET INFORMATION DATAFRAME ...")
  logger.info(f"FIX Mode is {handling}")
  logger.info(f"number of rows : {len(df)}")
  for key in miss_values:
    if miss_values[key] != 0:
      missing_dt = df.loc[df[key].isna(), "dt"]
      tagname = key if key != "dt" else "dt"
      m = miss_values[key]
      percentage = f"{round(m / len(df), 4) * 100}%"
      logger.warning(
        f"Found Missing Values in {tagname}[{key}] : {m} | dt_min : {missing_dt.min()} | dt_max : {missing_dt.max()}\
        percentage : {percentage}"
      )
      if handling == "NONE":
        logger.warning(f"Handle Missing Value : {handling}")
        return df
      if handling == "NEIGHBOR_VALUE":
        first = miss[key].iloc[0]
        if first:
          logger.warning("from the first index it is already a miss value")
          df[key] = df[key].bfill()
        df[key].ffill(inplace=True)

      if handling == "MIN":
        val = df[key].min()
      elif handling == "MAX":
        val = df[key].max()
      elif handling == "MEDIAN":
        val = df[key].median()
      elif handling == "MEAN":
        val = df[key].mean()
      else:
        val = 0
      df[key] = df[key].fillna(val)
  if df.isna().sum().sum() == 0:
    logger.success("The dataset has been successfully cleaned")
    return df


def pca(
    X: Union[pd.DataFrame, np.ndarray], n_components: int = 2, only_obj: bool = False
):

  if isinstance(X, pd.DataFrame):
    X = X.drop(columns="dt").values if "dt" in X.columns else X.values
  obj = PCA(n_components=n_components).fit(X)
  if only_obj:
    return obj
  return obj, obj.transform(X)


def init_storages(name):
  """Siapkan folder artefak sebuah dataset.

  Log TIDAK di sini lagi — semuanya di `Config.log_dir` (lihat app/logger.py),
  supaya satu perintah `rm -rf storages/<ds>` tidak ikut menghapus jejak
  kejadiannya.
  """
  path = Config.dir / "storages" / name
  os.makedirs(path, exist_ok=True)
  os.makedirs(path / "top_model", exist_ok=True)
  os.makedirs(path / "results", exist_ok=True)

