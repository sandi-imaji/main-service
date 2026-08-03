from pydantic import BaseModel
from typing import Optional, Union, List
import warnings,time,traceback,pandas as pd,requests as req,json,asyncio,httpx
from urllib3.exceptions import HTTPError
from app.config import Config
from app.helpers import DTEncoder, miss_val_handling_df, get_row_id
from app.database.schemas import DatasetHandling,PreprocessingSchema
from datetime import datetime, timedelta
from app.logger import Logger
from functools import reduce

from app.utils.http_client import AsyncHTTPClient
from app.database.DB import Dataset
from app.database.schemas import StatusProcess, MetaDataset


from urllib3.exceptions import InsecureRequestWarning

# Selama Config.sl_verify masih False, requests akan membanjiri log dengan
# InsecureRequestWarning tiap panggilan.
if not Config.sl_verify:
  warnings.simplefilter("ignore", InsecureRequestWarning)


def _retry_budget(max_retries: int, logger=None) -> int:
  """max_retries < 1 bikin loop retry tidak pernah jalan dan fungsinya diam-diam
  mengembalikan None. Konfigurasi seperti itu dipaksa jadi minimal 1 percobaan."""
  if max_retries < 1:
    if logger: logger.warning(f"max_retries={max_retries} tidak valid, dipakai 1")
    return 1
  return max_retries

class PullDate(BaseModel):
  current_date: str
  time_start: str
  time_end: str = "23:59:00"

  def is_today(self) -> bool: return self.current_date == DTEncoder.now_str()

  @staticmethod
  def from_dt(dt: datetime):
    from app.helpers import DTEncoder
    current_date = DTEncoder.dt_to_str(dt)
    time_start = dt.strftime("%H:%M:00")
    now = DTEncoder.now()
    time_end = (
        now.strftime("%H:%M:00") if now.date() == dt.date() else "23:59:00"
    )
    return PullDate(
        current_date=current_date, time_start=time_start, time_end=time_end
    )
  @staticmethod
  def now():
    """Hari ini, dari 00:00:00 sampai jam sekarang."""
    dt = DTEncoder.now()
    return PullDate(
        current_date=DTEncoder.dt_to_str(dt),
        time_start="00:00:00",
        time_end=dt.strftime("%H:%M:00"),
    )


  @staticmethod
  def norm_time(value: Optional[str], default: str) -> str:
    """Rapikan "9:5" / "9:5:0" jadi "09:05:00" (API menolak yang tidak padded)."""
    if not value: return default
    parts = [int(p) for p in str(value).split(":")]
    while len(parts) < 3: parts.append(0)
    h, m, s = parts[:3]
    return f"{h:02d}:{m:02d}:{s:02d}"

  @staticmethod
  def from_str(dt: str, time_start: Optional[str] = None,
               time_end: Optional[str] = None):
    """Tanpa time_start/time_end, jam-nya diturunkan dari tanggalnya sendiri
    (selalu tengah malam). Isi keduanya kalau jendela waktunya sudah ditentukan,
    mis. dari kolom time_start/time_end milik Dataset."""
    parsed = DTEncoder.str_to_dt(dt)
    if time_start is None and time_end is None:
      return PullDate.from_dt(parsed)
    return PullDate(
        current_date=DTEncoder.dt_to_str(parsed),
        time_start=PullDate.norm_time(time_start, "00:00:00"),
        time_end=PullDate.norm_time(time_end, "23:59:00"),
    )

  def time_start_t(self):
    from datetime import time

    args = (int(i) for i in self.time_start.split(":"))
    return time(*args)

  def time_end_t(self):
    from datetime import time

    args = (int(i) for i in self.time_end.split(":"))
    return time(*args)

# INFO: ########################### Modules #################################
def _realtime_request_spec(tagname: str):
  """Endpoint/params/body for the realtime 'search' API (shared by sync+async).

  Realtime value is read via the tag-search endpoint (not modbus/get-point) so
  that constant points / points without modbus info are still resolvable.
  """
  return (
      Config.sl_url("application/api/tags/search.php"),
      {"search": tagname, "modbus_only": 1},
      {"authtoken": Config.sl_key},
  )


def _parse_search_currvalue(data: list, tagname: str) -> float:
  """Extract `tagname`'s value from the search-API rows.

  Each row's `text` is "TAGNAME:value". Returns the matching tag's value as a
  float, or 0.0 when it is not found / constant / unparseable.
  """
  for item in data:
    text = item.get("text", "") if isinstance(item, dict) else ""
    name, sep, raw = text.partition(":")
    if sep and name == tagname:
      try:
        return float(raw)
      except (TypeError, ValueError):
        return 0.0
  return 0.0


def get_realtime(
    tagname: str,
    logger=None,
    max_retries: int = Config.sl_retry,
    retry_delay: float = 0.5,
    timeout: int = Config.sl_timeout,
):
  """
  Get realtime value for a data point with automatic retry on timeout and connection errors.

  Uses the tag-search API (see `_realtime_request_spec`). Returns the tag's
  current value as a float, or 0.0 when the tag is not found / has no realtime
  value; transient network errors are retried, then raised.

  Args:
    tagname: The identifier for the data point
    logger: Logger instance for logging
    max_retries: Maximum number of retry attempts (default: 3)
    retry_delay: Delay between retries in seconds (default: 1.0)
    timeout: Request timeout in seconds (default: 3)

  Returns:
    Current value as float
  """
  url, params, body = _realtime_request_spec(tagname)
  last_exception = None
  max_retries = _retry_budget(max_retries, logger)

  for attempt in range(max_retries):
    try:
      if logger and attempt > 0:
        logger.info(
            f"Retry attempt {attempt + 1}/{max_retries} for tagname {tagname}"
        )
      result = req.post(
          url, data=body, verify=Config.sl_verify, timeout=timeout, params=params
      )
      if result.status_code != 200:
        error_msg = f"status_code: {result.status_code} | text: {result.text} | url: {url}"
        if logger:
          logger.error(error_msg)
        raise ValueError(error_msg)
      data = result.json().get("data", [])
      value = _parse_search_currvalue(data, tagname)

      if logger:
        logger.info(f"Point: {tagname} | Successfully retrieved value: {value}")

      return value

    except (req.exceptions.Timeout, req.exceptions.ConnectionError, HTTPError) as e:
      last_exception = e
      error_type = type(e).__name__

      if logger:
        logger.warning(
            f"{error_type} for tagname {tagname} "
            f"(attempt {attempt + 1}/{max_retries}): {str(e)}"
        )

      # If this is not the last attempt, wait before retrying
      if attempt < max_retries - 1:
        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
      else:
        error_msg = (
            f"Failed after {max_retries} attempts for tagname {tagname}: "
            f"{error_type} - {str(e)}"
        )
        if logger:
          logger.error(error_msg)
        raise ValueError(error_msg)

    except req.exceptions.RequestException as e:
      # For other request exceptions, don't retry
      error_msg = f"Request failed for Point {tagname}: {str(e)}"
      if logger:
        logger.error(error_msg)
      raise ValueError(error_msg)

    except ValueError as e:
      raise e

    except Exception as e:
      # For non-request exceptions, don't retry
      error_msg = f"Error processing data for Point {tagname}: {str(e)}"
      if logger:
        logger.error(error_msg)
      raise ValueError(error_msg)

  raise ValueError(
      f"Failed after {max_retries} attempts for tagname {tagname}: "
      f"{last_exception}")

# NOTE:  Raw Parameter
def get_history(
  tagname: Union[int,str],
  current_date: str,
  time_start: str = "00:00:00",
  time_end: str = "23:59:00",
  interval: int = 1,
  to_dataframe: bool = False,
  logger=None,
  max_retries: int = Config.sl_retry,
  retry_delay: float = 0.7,
  quality: int = 0,
  timeout: int = Config.sl_timeout_long,
  row_id: Optional[Union[int, str]] = None,
) -> Union[List,pd.DataFrame]:
  """
  Retrieve historical data with automatic retry on timeout and connection errors.

  Args:
    tagname | row_id: The identifier for the data row
    current_date: Date string in format YYYYMMDD
    time_start: Start time in HH:MM:SS format
    time_end: End time in HH:MM:SS format
    interval: Time interval in seconds
    to_dataframe: Whether to return as DataFrame or list
    logger: Logger instance for logging
    max_retries: Maximum number of retry attempts (default: 3)
    retry_delay: Delay between retries in seconds (default: 1.0)
    timeout: Request timeout in seconds (default: 30)
    row_id: kalau sudah tahu row_id-nya, isi ini supaya lookup tagname->row_id
      (network ke /get-point, ~7s) di-skip. Nama kolom output tetap `tagname`.

  column dt nya udah DATETIME!

  Returns:
    List of [datetime, values] or DataFrame with datetime index
  """
  headers = {"Content-Type": "application/x-www-form-urlencoded"}

  if row_id is not None:
    # Pemanggil sudah me-resolve row_id (mis. pull_history yang cukup sekali per
    # kolom, bukan per hari). Tetap validasi tipe tagname supaya jenis yang
    # tidak didukung tidak lolos diam-diam.
    if not isinstance(tagname, (int, str)):
      raise ValueError(f"tagname type is fail : {type(tagname)}")
    row_id = str(row_id)
  elif isinstance(tagname,int):
    row_id = str(tagname)
  elif isinstance(tagname,str):
    row_id = get_row_id(tagname)
    if not row_id: raise ValueError(f"Tagname: {tagname} is not found")
  else: raise ValueError(f"tagname type is fail : {type(tagname)}")

  packet = {
    "function": "getDBHistory",
    "key": Config.key,
    "params": {
      "row_id": row_id,
      "current_date": current_date,
      "time_start": time_start,
      "time_end": time_end,
      "interval": interval,
      "quality": quality,
    },
  }
  url = Config.sl_url("application/api/tags/get-history")
  packet = json.dumps(packet)
  payload = {"packet": packet}
  last_exception = None
  max_retries = _retry_budget(max_retries, logger)
  for attempt in range(max_retries):
    try:
      if logger and attempt > 0:
        logger.info(
            f"Retry attempt {attempt + 1}/{max_retries} for row_id {row_id}, date {current_date}"
        )
      result = req.post(
          url, data=payload, headers=headers, verify=Config.sl_verify,
          timeout=timeout
      )
      if result.status_code != 200 or not result.text.strip():
        error_msg = f"status_code is {result.status_code} | url: {url} | text: {result.text}"
        if logger: logger.error(error_msg)
        raise ValueError(error_msg)

      result_json = result.json()
      if not result_json:
        if logger:
          logger.warning(
              f"Column: {tagname} | {DTEncoder.str_to_dt(current_date)} | No data returned"
          )
        # "dt" harus tetap jadi kolom (bukan index) dan bertipe datetime,
        # supaya bentuknya sama dengan jalur normal dan tetap bisa di-merge.
        return (
            pd.DataFrame({
                "dt": pd.Series(dtype="datetime64[ns, UTC]"),
                tagname: pd.Series(dtype="float64"),
            })
            if to_dataframe
            else [[], []]
        )

      if logger:
        logger.success(
            f"Column: {tagname} | {DTEncoder.str_to_dt(current_date)} | Successfully retrieved!"
        )

      # Unpack timestamps and values
      dt, val = zip(*result_json)
      dt = [DTEncoder.unixTS_to_dt(i) for i in dt]

      if to_dataframe:
        df = pd.DataFrame({"dt": dt, tagname: val})
        df["dt"] = pd.to_datetime(df["dt"])
        return df

      return [dt, val]

    except (req.exceptions.Timeout, req.exceptions.ConnectionError, HTTPError) as e:
      last_exception = e
      error_type = type(e).__name__

      if logger:
        logger.warning(
            f"{error_type} for tagname {tagname}, date {current_date} "
            f"(attempt {attempt + 1}/{max_retries}): {str(e)}"
        )
      # If this is not the last attempt, wait before retrying
      if attempt < max_retries - 1:
        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
      else:
        error_msg = (
            f"Failed after {max_retries} attempts for tagname {tagname}: "
            f"{error_type} - {str(e)}"
        )
        if logger: logger.error(error_msg)
        raise ValueError(error_msg)

    except req.exceptions.RequestException as e:
      # For other request exceptions, don't retry
      error_msg = f"Request failed for tagname {tagname}: {str(e)}"
      if logger: logger.error(error_msg)
      raise ValueError(error_msg)

    except Exception as e:
      # For non-request exceptions, don't retry
      error_msg = f"Error processing data for tagname {tagname}: {str(e)}"
      if logger:
        logger.error(error_msg)
      raise ValueError(error_msg+f" | row id : {row_id}")

  raise ValueError(
      f"Failed after {max_retries} attempts for tagname {tagname}: "
      f"{last_exception}")

# NOTE:  Raw Parameter
def pull_history(
  columns: List, start_date: PullDate, end_date: PullDate,preprocessing:PreprocessingSchema, logger=None,
  tags: Optional[dict] = None) -> pd.DataFrame:
  """
  untuk saat ini, preprocessing pada saat pulling hanya impute miss value!

  Hari yang gagal / kosong dicatat di df.attrs["failed_days"] supaya pemanggil
  bisa tahu hasilnya berlubang, bukan cuma lewat log.

  tags: pemetaan {tagname: row_id} (dari meta dataset). Kalau row_id sebuah
    kolom sudah ada di sini, lookup get_row_id (network ke /get-point, ~7s per
    kolom) di-skip.
  """
  if logger is None: logger = Logger("")

  # Kolom duplikat (mis. target ikut masuk features) dulu menghasilkan kolom
  # "<tag>_dup" diam-diam dari merge_asof.
  seen, unique_columns = set(), []
  for col in columns:
    if col in seen:
      logger.warning(f"Kolom duplikat diabaikan: {col}")
      continue
    seen.add(col)
    unique_columns.append(col)
  columns = unique_columns
  if not columns: raise ValueError("columns kosong!")

  end_dt = DTEncoder.str_to_dt(end_date.current_date)
  start_dt = DTEncoder.str_to_dt(start_date.current_date)
  # if start_date.time_start.split(":")[0] > end_date.time_end.split(":")[0]: raise ValueError("Start (Hours) > End (Hours)")
  if start_dt.timestamp() > end_dt.timestamp(): raise ValueError("End Date < Start Date !")

  delta_days = (end_dt - start_dt).days
  if delta_days >= 1:
    days = [
        (start_dt + timedelta(days=i)).strftime(DTEncoder.FMT_DT)
        for i in range(delta_days + 1)
    ]
    times = (
        [(start_date.time_start, start_date.time_end)]
        + [("00:00:00", "23:59:00") for _ in range(len(days) - 2)]
        + [(end_date.time_start, end_date.time_end)]
    )
    if len(days) != len(times): raise ValueError(f"len :{len(days)} != {len(times)} {days} {times}")
  else:
    days = [start_dt.strftime(DTEncoder.FMT_DT)]
    # Untuk rentang satu hari, batas atasnya milik end_date - kalau dipakai
    # start_date.time_end, time_end yang diminta pemanggil terbuang.
    times = [(start_date.time_start, end_date.time_end)]

  # Pull data for each feature/target
  results = []
  failed_days = {}
  for col in columns:
    logger.info(f"Pulling feature: {col}")
    feature_data = []

    # row_id di-resolve SEKALI per kolom. Sebelumnya get_history menerjemahkan
    # tagname->row_id (network ke /get-point, ~7s) untuk SETIAP hari, jadi
    # lookup lambat itu diulang n_hari kali padahal hasilnya sama. row_id dioper
    # ke get_history via param `row_id` (nama kolom output tetap tagname).
    if isinstance(col, int):
      row_id = col
    elif tags and tags.get(col):
      # row_id sudah tersimpan di tags {tagname: row_id} saat create_dataset,
      # jadi lookup get_row_id (network ~7s per kolom) tidak perlu diulang.
      row_id = tags[col]
      logger.info(f"row_id dari tags: {col} -> {row_id}")
    else:
      rid = get_row_id(col, logger)
      if not rid: raise ValueError(f"Tagname: {col} is not found")
      row_id = rid

    for idx, day in enumerate(days):
      try:
        start_time, end_time = times[idx]
        df_day = get_history(
            col,
            day,
            time_start=start_time,
            time_end=end_time,
            interval=1,
            to_dataframe=True,
            logger=logger,
            row_id=row_id,
        )
        if df_day is None or df_day.empty:
          logger.warning(f"Empty data for {col} on {day}")
          failed_days.setdefault(col, []).append(day)
          continue
        feature_data.append(df_day)
      except Exception as e:
        logger.warning(f"Failed to get data for {col} on {day}: {str(e)}")
        failed_days.setdefault(col, []).append(day)
        continue

    if not feature_data:
      raise ValueError(f"No data retrieved for feature: {col}")

    # Concatenate all days for this feature
    feature_df = pd.concat(feature_data, axis=0)
    # feature_df = feature_df.ffill().bfill()
    feature_df.sort_values("dt", inplace=True)
    results.append(feature_df)

  # Hari yang bolong harus terlihat jelas, bukan cuma satu baris warning yang
  # tenggelam di antara ratusan baris log.
  if failed_days:
    for col, miss in failed_days.items():
      logger.error(
          f"Feature {col}: {len(miss)}/{len(days)} hari tidak ada datanya "
          f"-> {miss}"
      )

  # merge_asof memakai frame pertama sebagai tulang punggung: baris milik
  # feature lain yang tidak punya pasangan dalam toleransi akan hilang. Yang
  # paling rapat sampelnya dipakai sebagai tulang punggung supaya kehilangan
  # datanya seminimal mungkin.
  results.sort(key=len, reverse=True)
  df = reduce(
    lambda left, right: pd.merge_asof(
        left,
        right,
        on="dt",
        tolerance=pd.Timedelta("5min"),  # sesuaikan tolerance
        direction="nearest",  # bisa 'backward', 'forward', atau 'nearest'
        suffixes=("", "_dup"),
    ),
    results,
  )

  dropped = {
      frame.columns[-1]: len(frame) - len(df)
      for frame in results if len(frame) > len(df)
  }
  if dropped:
    logger.warning(
        f"Baris yang tidak dapat pasangan saat merge (toleransi 5 menit): {dropped}"
    )

  dup_cols = [c for c in df.columns if c.endswith("_dup")]
  if dup_cols:
    logger.error(f"Kolom duplikat hasil merge dibuang: {dup_cols}")
    df = df.drop(columns=dup_cols)

  # Miss Values
  if df.isna().sum().sum() and preprocessing.missing_handling != DatasetHandling.NONE: df = miss_val_handling_df(df,logger,preprocessing.missing_handling)

  n_rows, n_cols = df.shape
  logger.info(f"Number of Rows : {n_rows} | Number of Cols : {n_cols}")
  logger.info(f"Columns : {df.columns.tolist()}")
  logger.success("dataframe created successfully !")
  df = df.sort_values(by="dt")
  df.attrs["failed_days"] = failed_days
  return df

def pull_history_nday(
  n_day: int,
  columns: List[str],
  preprocessing: Optional[PreprocessingSchema] = None,
  logger=None,
  tags: Optional[dict] = None,
) -> pd.DataFrame:
  """
  Pull history untuk `n_day` hari terakhir sampai jam sekarang.

  Rentang waktunya: dari 00:00:00 pada (hari ini - n_day) sampai jam sekarang
  hari ini. Jadi n_day=0 berarti hanya hari ini, n_day=7 berarti 8 hari
  (7 hari penuh + hari ini yang masih berjalan).

  Args:
    n_day: jumlah hari ke belakang, integer >= 0
    columns: list tagname (atau row_id) yang mau di-pull
    preprocessing: default PreprocessingSchema() (impute NEIGHBOR_VALUE)
    logger: default Logger("") kalau tidak diisi
    tags: pemetaan {tagname: row_id} untuk skip lookup get_row_id (~7s/kolom)

  Returns: DataFrame dengan kolom "dt" + satu kolom per tagname, urut naik.
  """
  if isinstance(n_day, bool) or not isinstance(n_day, int):
    raise ValueError(f"n_day harus integer, bukan {type(n_day).__name__}")
  if n_day < 0: raise ValueError(f"n_day harus >= 0, dapat {n_day}")
  if not columns: raise ValueError("columns kosong!")

  if logger is None: logger = Logger("")
  if preprocessing is None: preprocessing = PreprocessingSchema()

  end_date = PullDate.now()
  start_dt = (DTEncoder.now() - timedelta(days=n_day)).replace(
    hour=0, minute=0, second=0, microsecond=0
  )
  start_date = PullDate(
    current_date=DTEncoder.dt_to_str(start_dt),
    time_start="00:00:00",
    time_end="23:59:00",
  )

  logger.info(
    f"Pull {n_day} day(s): {start_date.current_date} {start_date.time_start} "
    f"-> {end_date.current_date} {end_date.time_end} | columns: {columns}"
  )
  return pull_history(
    columns,
    start_date=start_date,
    end_date=end_date,
    logger=logger,
    preprocessing=preprocessing,
    tags=tags,
  )

def pull_realtime(columns: list, logger=None):
  """
  Pull realtime values for multiple columns/features

  Args:
      columns: List of column identifiers (row_ids or tagnames)
      logger: Logger instance for logging

  Returns:
      List of realtime values corresponding to columns
  """
  if logger:
    logger.info(f"Pull Realtime ({len(columns)}) columns ...")

  values = []
  for col in columns:
    try:
      value = get_realtime(col, logger=logger)
      values.append(value)
    except Exception as e:
      if logger:
        logger.error(f"Failed to get realtime for {col}: {str(e)}")
      values.append(None)

  if logger: logger.info(f"Pull Realtime completed: {values}")

  return values

def pulling(dataset_name: str,db) -> bool:
  """
  function ini untuk pulling dari Dataset (orm)

  Return True kalau berhasil, False kalau gagal (status dataset di-set
  ERROR_PULL). Sebelumnya errornya cuma ditelan, jadi pemanggil tidak punya
  cara tahu selain membaca status di DB.
  """

  tic = time.monotonic()
  logger = Logger(dataset_name)
  logger.info(f"Enter Pull ... {dataset_name}")

  pull_time = 0

  dataset = Dataset.get_by_name(dataset_name, db)
  if not dataset:
    raise ValueError(f"{dataset_name} not found!")
  try:
    dataset.status = StatusProcess.RUNNING_PULL
    db.commit()

    # list(...) wajib: `columns += [target]` pada dataset.features langsung akan
    # memutasi kolom JSON-nya, sehingga target ikut tersimpan sebagai feature.
    columns = list(dataset.features)
    if dataset.task_type.is_supervised():
      columns.append(dataset.target)

    logger.info(f"Pull.. {dataset_name}")
    # Jendela waktu milik dataset ikut dipakai; sebelumnya kolom time_start /
    # time_end di ORM diabaikan total karena from_str menurunkan jamnya dari
    # tanggal (yang selalu tengah malam).
    start_date = PullDate.from_str(dataset.start_date, time_start=dataset.time_start, time_end="23:59:00")
    end_date = PullDate.from_str(dataset.end_date, time_start="00:00:00", time_end=dataset.time_end)
    logger.info(f"Window: {start_date} -> {end_date}")
    if dataset.preprocessing: preprocessing = PreprocessingSchema()

    tags = dataset.meta.tags if dataset.meta else None
    df = pull_history( columns, start_date=start_date, end_date=end_date, logger=logger,preprocessing=preprocessing, tags=tags)
    pull_time += time.monotonic() - tic

    if df is None or df.empty: raise ValueError("Generated dataframe is empty")

    current_dt = df.iloc[-1]["dt"]

    # Rename columns from row_id to tagname dan update dataset.features/target
    columns_map = {}
    new_features = []
    new_target = None

    for col in df.columns:
      if col != "dt":
        tagname = col
        if tagname:
          columns_map[col] = tagname
          # Update features list - compare with original row_id or tagname
          if col in dataset.features or tagname in dataset.features:
            new_features.append(tagname)
          # Update target - compare with original row_id or tagname
          if col == dataset.target or tagname == dataset.target:
            new_target = tagname
        else:
          # Keep original if tagname not found
          if col in dataset.features or tagname in dataset.features:
            new_features.append(col)
          if col == dataset.target or tagname == dataset.target:
            new_target = col

    if columns_map:
      df = df.rename(columns=columns_map)
      # Update dataset fields to use tagnames
      dataset.features = new_features
      if new_target:
        dataset.target = new_target
      logger.info(
          f"Columns renamed to tagnames: {list(columns_map.values())}"
      )

    n_rows, n_cols = df.shape
    path = f"storages/{dataset.name}/data.csv"
    failed_days = df.attrs.get("failed_days", {})
    notes = (
        f"Hari tanpa data: {failed_days}" if failed_days else ""
    )
    dataset.meta = MetaDataset(
        # DTEncoder.now() supaya sumber waktunya seragam dengan kolom dt dan
        # current_dt; datetime.now() memakai jam lokal mesin.
        created_at=DTEncoder.now().isoformat(),
        created_by="Anonymous",
        # Pertahankan tags yang di-set saat create_dataset; meta dibangun ulang
        # di sini sehingga tanpa ini tags akan tereset ke None.
        tags=dataset.meta.tags if dataset.meta else None,
        current_dt=current_dt.isoformat(),
        size_of=df.memory_usage().sum(),
        n_rows=n_rows,
        pulling_time=pull_time,
        n_cols=n_cols,
        missing_values=df.isna().sum().sum().item(),
        is_outlier=False,
        random_seed=4,
        columns=df.columns.tolist(),
        notes=notes,
        train_size=0.8,
        path=path,
    ).model_dump()

    df.to_csv(Config.dir / path, index=False)

    interval = df["dt"].diff().median()  # Gunakan median untuk lebih robust
    dataset.is_valid = True
    dataset.status = StatusProcess.SUCCESS_PULL
    dataset.interval = (
        int(interval.total_seconds() / 60) if isinstance(interval, timedelta) else 0
    )

    logger.info("End Pulling ...")
    db.commit()
    # db.refresh(dataset)
    logger.info("Pulled successfully!")
    return True

  except Exception as e:
    # Traceback penuh, bukan cuma str(e): tanpa ini penyebab aslinya hilang
    # karena exception-nya tidak diteruskan ke pemanggil.
    logger.error(f"Dataset: {dataset_name} | Error: {str(e)}")
    logger.error(traceback.format_exc())
    # rollback dulu - kalau session-nya sudah rusak karena error di atas,
    # commit berikutnya akan gagal dan menutupi error yang sebenarnya.
    db.rollback()
    dataset.is_valid = False
    dataset.status = StatusProcess.ERROR_PULL
    db.commit()
    return False

def query_tagnames(query:str,single:bool=False, logger=None) -> List[dict]:
  """Search tagname from external API using async HTTP client"""
  url = Config.sl_url("application/api/tags/search.php")
  params = {"search": query, "modbus_only": 1}
  body = {"authtoken": Config.sl_key}
  try:
    res = req.post(url, data=body, params=params, timeout=Config.sl_timeout, verify=Config.sl_verify)
    if res.status_code != 200:
      if logger: logger.error(f"query_tagnames({query}) status {res.status_code}")
      return []
    data = res.json().get("data", [])
    res = []
    for item in data:
      tagname,value = item["text"].split(":",1)
      value = float(value)
      tagname = item['text'].split(":",1)[0]
      row_id = item['id']
      res.append(dict(tagname=tagname,row_id=row_id,currvalue=value))
    if logger:logger.info(f"Success Query '{query}'| Total Points : {len(res)}")
    return res
  except Exception as e:
    if logger: logger.error(f"query_tagnames({query}) gagal: {e}")
    return []

# =============================================================================
# Async Versions (Non-blocking)
# =============================================================================

# WARNING: Ada beberapa point yang tidak ada modbus information nya ! Function ini tidak coverage semua point!
async def async_get_realtime(
    tagname: str,
    logger=None,
    max_retries: int = Config.sl_retry,
    retry_delay: float = 0.5,
    timeout: int = Config.sl_timeout,
) -> float:
  """
  Async version of get_realtime - Get realtime value without blocking event loop.

  Uses the tag-search API (see `_realtime_request_spec`). Returns the tag's
  current value as a float, or 0.0 when the tag is not found / has no realtime
  value; transient network errors are retried, then raised.

  Args:
      tagname: The identifier for the data point
      logger: Logger instance for logging
      max_retries: Maximum number of retry attempts
      retry_delay: Delay between retries in seconds
      timeout: Request timeout in seconds

  Returns:
      Current value as float
  """

  url, params, body = _realtime_request_spec(tagname)
  last_exception = None
  max_retries = _retry_budget(max_retries, logger)

  client = await AsyncHTTPClient.get_client(verify=Config.sl_verify)

  for attempt in range(max_retries):
    try:
      if logger and attempt > 0:
        logger.info(
            f"Retry attempt {attempt + 1}/{max_retries} for tagname {tagname}"
        )

      result = await client.post(url, data=body, params=params, timeout=timeout)

      if result.status_code != 200:
        error_msg = f"status_code: {result.status_code} | text: {result.text} | url: {url}"
        if logger:
          logger.error(error_msg)
        raise ValueError(error_msg)

      data = result.json().get("data", [])
      value = _parse_search_currvalue(data, tagname)

      if logger:
        logger.info(f"Point: {tagname} | Successfully retrieved value: {value}")

      return value

    except asyncio.TimeoutError as e:
      last_exception = e
      error_type = "TimeoutError"

      if logger:
        logger.warning(
            f"{error_type} for tagname {tagname} "
            f"(attempt {attempt + 1}/{max_retries})"
        )

      if attempt < max_retries - 1:
        await asyncio.sleep(retry_delay * (attempt + 1))
      else:
        error_msg = (
            f"Failed after {max_retries} attempts for tagname {tagname}: "
            f"{error_type}"
        )
        if logger:
          logger.error(error_msg)
        raise ValueError(error_msg)

    except ValueError:
      # Error permanen (tagname tidak ada, currvalue kosong, status bukan 200):
      # retry hanya menunda kegagalan yang sama. Versi sync juga begini.
      raise

    except Exception as e:
      last_exception = e
      error_type = type(e).__name__

      if logger:
        logger.warning(
            f"{error_type} for tagname {tagname} "
            f"(attempt {attempt + 1}/{max_retries}): {str(e)}"
        )

      if attempt < max_retries - 1:
        await asyncio.sleep(retry_delay * (attempt + 1))
      else:
        error_msg = (
            f"Failed after {max_retries} attempts for tagname {tagname}: "
            f"{error_type} - {str(e)}"
        )
        if logger:
          logger.error(error_msg)
        raise ValueError(error_msg)

  raise ValueError(
      f"Failed after {max_retries} attempts for tagname {tagname}: "
      f"{last_exception}")

async def async_get_history(
    tagname: str,
    current_date: str,
    time_start: str = "00:00:00",
    time_end: str = "23:59:00",
    interval: int = 5,
    to_dataframe: bool = False,
    logger=None,
    max_retries: int = Config.sl_retry,
    retry_delay: float = 0.7,
    quality: int = 0,
    timeout: int = Config.sl_timeout_long,
):
  """
  Async version of get_history - Retrieve historical data without blocking.

  Args:
      tagname: The tagname for the data point
      current_date: Date string in format YYYYMMDD
      time_start: Start time in HH:MM:SS format
      time_end: End time in HH:MM:SS format
      interval: Time interval in seconds
      to_dataframe: Whether to return as DataFrame or list
      logger: Logger instance for logging
      max_retries: Maximum number of retry attempts
      retry_delay: Delay between retries in seconds
      quality: Data quality filter
      timeout: Request timeout in seconds

  Returns: List of [datetime, values] or DataFrame with datetime index
  """

  headers = {"Content-Type": "application/x-www-form-urlencoded"}

  # Versi sync menolak tagname yang tidak ketemu; di sini dulu row_id kosong
  # tetap dikirim ke API dan gagalnya jadi tidak jelas.
  if isinstance(tagname, int):
    row_id = str(tagname)
  elif isinstance(tagname, str):
    row_id = get_row_id(tagname, logger)
    if not row_id: raise ValueError(f"Tagname: {tagname} is not found")
  else: raise ValueError(f"tagname type is fail : {type(tagname)}")

  packet = {
      "function": "getDBHistory",
      "key": Config.key,
      "params": {
          "row_id": row_id,
          "current_date": current_date,
          "time_start": time_start,
          "time_end": time_end,
          "interval": interval,
          "quality": quality,
      },
  }
  url = Config.sl_url("application/api/tags/get-history")
  packet_json = json.dumps(packet)
  payload = {"packet": packet_json}
  last_exception = None
  max_retries = _retry_budget(max_retries, logger)

  client = await AsyncHTTPClient.get_client(verify=Config.sl_verify)

  for attempt in range(max_retries):
    try:
      if logger and attempt > 0:
        logger.info(
            f"Retry attempt {attempt + 1}/{max_retries} for row_id {row_id}, date {current_date}"
        )

      result = await client.post(
          url, data=payload, headers=headers, timeout=timeout
      )

      if result.status_code != 200:
        error_msg = f"status_code is {result.status_code} | url: {url}"
        if logger:
          logger.error(error_msg)
        raise ValueError(error_msg)

      result_json = result.json()

      if not result_json:
        if logger:
          logger.warning(
              f"Column: {tagname} | {DTEncoder.str_to_dt(current_date)} | No data returned"
          )
        # "dt" harus tetap jadi kolom (bukan index) dan bertipe datetime,
        # supaya bentuknya sama dengan jalur normal dan tetap bisa di-merge.
        return (
            pd.DataFrame({
                "dt": pd.Series(dtype="datetime64[ns, UTC]"),
                tagname: pd.Series(dtype="float64"),
            })
            if to_dataframe
            else [[], []]
        )

      if logger:
        logger.success(
            f"Column: {tagname} | {DTEncoder.str_to_dt(current_date)} | Successfully retrieved!"
        )

      # Unpack timestamps and values
      dt, val = zip(*result_json)
      dt = [DTEncoder.unixTS_to_dt(i) for i in dt]

      if to_dataframe:
        df = pd.DataFrame({"dt": dt, tagname: val})
        df["dt"] = pd.to_datetime(df["dt"])
        return df

      return [dt, val]

    except asyncio.TimeoutError as e:
      last_exception = e
      error_type = "TimeoutError"

      if logger:
        logger.warning(
            f"{error_type} for tagname {tagname}, date {current_date} "
            f"(attempt {attempt + 1}/{max_retries})"
        )

      if attempt < max_retries - 1: await asyncio.sleep(retry_delay * (attempt + 1))
      else:
        error_msg = (
            f"Failed after {max_retries} attempts for tagname {tagname}: "
            f"{error_type}"
        )
        if logger: logger.error(error_msg)
        raise ValueError(error_msg)

    except ValueError:
      # Error permanen: tagname tidak ada / status bukan 200. Jangan di-retry.
      raise

    except Exception as e:
      last_exception = e
      error_type = type(e).__name__

      if logger:
        logger.warning(
            f"{error_type} for tagname {tagname}, date {current_date} "
            f"(attempt {attempt + 1}/{max_retries}): {str(e)}"
        )

      if attempt < max_retries - 1:
        await asyncio.sleep(retry_delay * (attempt + 1))
      else:
        error_msg = (
            f"Failed after {max_retries} attempts for tagname {tagname}: "
            f"{error_type} - {str(e)}"
        )
        if logger:
          logger.error(error_msg)
        raise ValueError(error_msg)

  raise ValueError(
      f"Failed after {max_retries} attempts for tagname {tagname}: "
      f"{last_exception}")

async def async_pull_realtime(columns:List[str]) -> dict:
  # NOTE: Get Realtime nya menggunakan Query Tagname, karena ada beberapa point constant atau point yang tidak ada modbus information
  async def fetch_currValue(client,id):
    url = Config.sl_url("application/api/tags/search.php")
    params = {"search": id, "modbus_only": 1}
    body = {"authtoken": Config.sl_key}
    response = await client.post(url,data=body,params=params,timeout=Config.sl_timeout)
    if  response.status_code != 200:
      return {}
    data = response.json().get("data",[])
    for item in data:
      tagname,value = item["text"].split(":",1)
      if tagname == id: value = float(value)
      else: value = 0.0
      return [tagname,[value]]

  async with httpx.AsyncClient(verify=False) as client:
    tasks = [fetch_currValue(client,col) for col in columns]
    results = await asyncio.gather(*tasks)
    return dict(results)
    if results: return dict(results)

if __name__ == "__main__":
  from app.database.base import get_session
  import time
  q =  Dataset.get_by_name(name="Clustering-10de2b03",db=next(get_session()))
  features = q.features
  tic = time.monotonic()
  data = asyncio.run(async_pull_realtime(features))
  toc = time.monotonic()
  print(toc-tic)


