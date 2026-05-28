from app.config import Config
import os,math
import datetime
import pandas as pd
import numpy as np
import re
import requests as req
from typing import List, Optional, Union, Dict, Any
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

TAGNAME_PATH = Config.dir / "tagname.csv"
TAGNAME = pd.read_csv(
    TAGNAME_PATH) if TAGNAME_PATH.exists() else pd.DataFrame()


def get_mapping():
  data = {}
  for _, v in TAGNAME.iterrows():
    data[v["row_id"]] = v.to_dict()
  return data

def get_tagname(row_id: int):
  tagname = get_mapping().get(row_id, "")
  return tagname if not tagname else tagname.get("tag_name", "")

def get_row_id(tagname: str,logger=None) -> str:
  url = f"{Config.sl_host}application/api/modbus/get-point?tagname={tagname}"
  data = req.post(url=url, verify=False, data=dict(key=Config.sl_key))
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
    

def get_curr_value(tagname: str):
  url = f"{Config.sl_host}application/api/modbus/get-point?tagname={tagname}"
  data = req.post(url=url, verify=False, data=dict(key=Config.sl_key))
  if data.status_code != 200: return ""
  else: return data.json()["data"]["currvalue"]


def get_point_id(row_id: int):
  return get_mapping().get(row_id, "").get("point_id", "")


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



def read_bytes_to_dict(fpath: str, columns: List[str]) -> List[dict]:
  with open(fpath, "rb") as f:
    data = f.read()
  record_length = len(columns)
  arr = np.frombuffer(data, dtype=np.float64)
  num_records = len(arr) // record_length
  resultReshape = arr[: num_records * record_length].reshape(
      num_records, record_length
  )
  results = [dict(zip(columns, row)) for row in resultReshape]
  return results


def parse_log_line(line: str, with_module: bool = False) -> Dict[str, Any]:
  """
  Parse satu baris log menjadi dict yang sesuai dengan frontend LogViewer.
  Format log: [2025-01-01 12:00:00] [INFO] module_name:function_name:123 - message
  """
  line = line.strip()
  if not line:
    return {}

  # Pattern untuk format: [timestamp] [LEVEL] module:function:line - message
  pattern = re.compile(
      r"\[(?P<timestamp>[^\]]+)\]\s+"
      r"\[(?P<level>[A-Z]+)\]\s+"
      r"(?P<module>[\w\.]+):(?P<function>\w+):(?P<line>\d+)\s+-\s+"
      r"(?P<message>.+)"
  )

  m = pattern.match(line)
  if not m:
    # Fallback kalau format tidak standar
    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "level": "INFO",
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
      "module": m.group("module"),
      "function": m.group("function"),
      "line": int(m.group("line")),
      "message": m.group("message").strip(),
      "raw": line,
  }


def read_logs(dataset_name: str) -> List[Dict[str, Any]]:
  """
  Digunakan untuk inisialisasi atau fallback kalau WebSocket belum connect
  """
  if dataset_name == "system": fpath = Config.dir / "logger" / "main.log"
  else: fpath = Config.dir / "storages" / dataset_name / "logs" / "main.log"

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
      tagname = key if key != "dt" else "dt"
      m = miss_values[key]
      percentage = f"{round(m / len(df), 4) * 100}%"
      logger.warning(
          f"Found Missing Values in {tagname}[{key}] : {m} | percentage : {percentage}"
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


def outlier_handling_df(df: pd.DataFrame, multiplier=1.5):
  df = df.drop(columns=["dt"])
  result = dict()
  for key in df.columns:
    Q1 = df[key].quantile(0.25)
    Q2 = df[key].quantile(0.75)
    IQR = Q2 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q2 + multiplier * IQR
    res = df[(df[key] < lower) | (df[key] > upper)]
    result[key] = res
  return result


def write_bytes(values, fpath: str):
  if not os.path.exists(fpath):
    os.makedirs(fpath, exist_ok=True)
  result = np.asarray(values, dtype=np.float64)
  resultBytes = result.tobytes()
  with open(fpath, "ab") as f:
    f.write(resultBytes)


def pca(
    X: Union[pd.DataFrame, np.ndarray], n_components: int = 2, only_obj: bool = False
):

  if isinstance(X, pd.DataFrame):
    X = X.drop(columns="dt").values if "dt" in X.columns else X.values
  obj = PCA(n_components=n_components).fit(X)
  if only_obj:
    return obj
  return obj, obj.transform(X)


def minmax(X: Union[pd.DataFrame, np.ndarray], only_obj: bool = False):
  """
  return obj,transform values
  """

  if isinstance(X, pd.DataFrame):
    X = X.drop(columns="dt").values if "dt" in X.columns else X.values
  scaler = MinMaxScaler().fit(X)
  if only_obj:
    return scaler
  return scaler, scaler.transform(X)


def init_storages_dataset(name):
  path = Config.dir / "storages" / name
  os.makedirs(path, exist_ok=True)
  os.makedirs(path / "logs", exist_ok=True)
  os.makedirs(path / "top_model", exist_ok=True)
  os.makedirs(path / "results", exist_ok=True)
  print("Init Storages ...")


def clean_float(val:Any):
  if val is None: return None
  # Handle dict (rekursif)
  if isinstance(val, dict): return {k: clean_float(v) for k, v in val.items()}
  # Handle list (rekursif)
  if isinstance(val, list): return [clean_float(item) for item in val]

  if hasattr(val,'item'): val = val.item()
  if isinstance(val, float):
    if math.isnan(val) or math.isinf(val): return None
    return val
  return val


if __name__ == "__main__":
  data = read_logs("Regression-42c81c4e")
  print(data)
  print(len(data))

