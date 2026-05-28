from pydantic import BaseModel
from typing import Optional, Union, List, Dict
import warnings,time,pandas as pd,requests as req,certifi,json,asyncio
from urllib3.exceptions import HTTPError
from app.config import Config
from app.helpers import DTEncoder, get_tagname, miss_val_handling_df, get_row_id,outlier_handling_df
from typing import Optional
from app.database.schemas import DatasetHandling,PreprocessingSchema
from datetime import datetime, timedelta
from app.logger import Logger
from functools import reduce

from app.utils.http_client import AsyncHTTPClient
from app.database.db import get_session
from app.database.orm import Dataset
from app.database.schemas import StatusProcess, MetaDataset


from urllib3.exceptions import InsecureRequestWarning

cafile = certifi.where()

warnings.simplefilter("ignore", InsecureRequestWarning)

class PullDate(BaseModel):
  current_date: str
  time_start: str
  time_end: str = "23:59:00"

  def is_today(self) -> bool: return self.current_date == DTEncoder.now_str()

  @staticmethod
  def from_dt(dt: datetime):
    from app.helpers import DTEncoder
    current_date = DTEncoder.dt_to_str(dt)
    time_start = f"{dt.hour}:{dt.minute}:00"
    now = DTEncoder.now()
    time_end = (
        f"{now.hour}:{now.minute}:00" if DTEncoder.now().date() == dt.date() else "23:59:00"
    )
    return PullDate(
        current_date=current_date, time_start=time_start, time_end=time_end
    )
  @staticmethod
  def now():
    dt = DTEncoder.now()
    current_dt = DTEncoder.dt_to_str(dt)
    time_start = "00:00:00"
    time_end = ( f"{dt.hour}:{dt.minute}:00" if dt.date() == dt.date() else "23:59:00")
    return PullDate(current_date = current_dt,time_start=time_start,time_end=time_end)


  @staticmethod
  def from_str(dt: str):
    from app.helpers import DTEncoder
    return PullDate.from_dt(DTEncoder.str_to_dt(dt))

  def time_start_t(self):
    from datetime import time

    args = (int(i) for i in self.time_start.split(":"))
    return time(*args)

  def time_end_t(self):
    from datetime import time

    args = (int(i) for i in self.time_end.split(":"))
    return time(*args)

# INFO: ########################### Modules #################################
def get_realtime(
    tagname: str,
    logger=None,
    max_retries: int = 2,
    retry_delay: float = 0.5,
    timeout: int = 5,
):
  """
  Get realtime value for a data point with automatic retry on timeout and connection errors.

  Args:
    tagname: The identifier for the data point
    logger: Logger instance for logging
    max_retries: Maximum number of retry attempts (default: 3)
    retry_delay: Delay between retries in seconds (default: 1.0)
    timeout: Request timeout in seconds (default: 3)

  Returns:
    Current value as float
  """
  url = f"{Config.sl_host}/application/api/modbus/get-point"
  params = dict(tagname=tagname)
  body = {"key": Config.key}
  last_exception = None

  for attempt in range(max_retries):
    try:
      if logger and attempt > 0:
        logger.info(
            f"Retry attempt {attempt + 1}/{max_retries} for tagname {tagname}"
        )
      result = req.post(
          url, data=body, verify=False, timeout=timeout, params=params
      )
      if result.status_code != 200:
        error_msg = f"status_code: {result.status_code} | text: {result.text} | url: {url}"
        if logger:
          logger.error(error_msg)
        raise ValueError(error_msg)
      result_json = result.json()

      if not result_json or result_json.get("message") == "Tagname not found":
        error_msg = f"Get Realtime ({tagname}) returned None | Tagname not found"
        if logger:
          logger.error(error_msg)
        raise ValueError(error_msg)

      result_json = result_json["data"]

      currvalue = result_json.get("currvalue", None)

      if currvalue is None:
        error_msg = (
            f"Get Realtime ({tagname}) - 'currvalue' field is missing or None"
        )
        if logger:
          logger.error(error_msg)
        raise ValueError(error_msg)

      if logger:
        logger.info(
            f"Point: {tagname} | Successfully retrieved value: {currvalue}"
        )

      return float(currvalue)

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

  # This should not be reached, but just in case
  if last_exception:
    raise ValueError(
        f"Failed after {max_retries} attempts: {str(last_exception)}")

# NOTE:  Raw Parameter
def get_history(
  tagname: Union[int,str],
  current_date: str,
  time_start: str = "00:00:00",
  time_end: str = "23:59:00",
  interval: int = 1,
  to_dataframe: bool = False,
  logger=None,
  max_retries: int = 3,
  retry_delay: float = 0.7,
  quality: int = 0,
  timeout: int = 3,
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

  column dt nya udah DATETIME!

  Returns:
    List of [datetime, values] or DataFrame with datetime index
  """
  headers = {"Content-Type": "application/x-www-form-urlencoded"}

  if isinstance(tagname,int):
    row_id = str(tagname)
  elif isinstance(tagname,str):
    row_id = get_row_id(tagname)
    if not row_id: raise ValueError(f"Tagname: {tagname} is not found")
  else: ValueError(f"tagname type is fail : {type(tagname)}")

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
  url = f"{Config.sl_host}application/api/tags/get-history"
  packet = json.dumps(packet)
  payload = {"packet": packet}
  last_exception = None
  for attempt in range(max_retries):
    try:
      if logger and attempt > 0:
        logger.info(
            f"Retry attempt {attempt + 1}/{max_retries} for row_id {row_id}, date {current_date}"
        )
      result = req.post(
          url, data=payload, headers=headers, verify=False, timeout=timeout
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
        return (
            pd.DataFrame(columns=["dt", tagname]).set_index("dt")
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
      raise ValueError(error_msg+f"row id : {row_id}")

  # This should not be reached, but just in case
  if last_exception:
    raise ValueError(
        f"Failed after {max_retries} attempts: {str(last_exception)}")

# NOTE:  Raw Parameter
def pull_history(
  columns: List, start_date: PullDate, end_date: PullDate, logger,preprocessing:PreprocessingSchema) -> pd.DataFrame:
  start_dt = DTEncoder.str_to_dt(start_date.current_date)
  end_dt = DTEncoder.str_to_dt(end_date.current_date)
  if start_date.time_start.split(":")[0] > end_date.time_end.split(":")[0]: raise ValueError("Start (Hours) > End (Hours)")
  if start_dt.timestamp() > end_dt.timestamp(): raise ValueError("End Date < Start Date !")

  delta_days = (end_dt - start_dt).days
  if delta_days > 1:
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
    times = [(start_date.time_start, start_date.time_end)]

  # Pull data for each feature/target
  results = []
  for col in columns:
    if logger: logger.info(f"Pulling feature: {col}")
    feature_data = []

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
        )
        feature_data.append(df_day)
      except Exception as e:
        if logger:
          logger.warning(f"Failed to get data for {col} on {day}: {str(e)}")
        continue

    if not feature_data:
      raise ValueError(f"No data retrieved for feature: {col}")

    # Concatenate all days for this feature
    feature_df = pd.concat(feature_data, axis=0)
    # feature_df = feature_df.ffill().bfill()
    feature_df.sort_values("dt", inplace=True)
    results.append(feature_df)

  # Combine all features
  # df = pd.concat(results, axis=1)
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
  # Miss Values
  print(preprocessing.missing_handling)
  if df.isna().sum().sum() and preprocessing.missing_handling != DatasetHandling.NONE: df = miss_val_handling_df(df,logger,preprocessing.missing_handling)

  # Outlier
  if preprocessing.outlier_handling != DatasetHandling.NONE: raise NotImplementedError("Outlier belum implementasi!")

  n_rows, n_cols = df.shape
  logger.info(f"Number of Rows : {n_rows} | Number of Cols : {n_cols}")
  logger.info(f"Columns : {df.columns.tolist()}")
  logger.success("dataframe created successfully !")
  return df.sort_values(by="dt")

# TODO: 
def pull_history_nday(n_day:int,columns:List[str],preprocessing:Optional[PreprocessingSchema]=None,logger=None):
  now = DTEncoder.now()
  start_date = now - timedelta(days=n_day)
  start_date = start_date.replace(hour=0,minute=0,second=0,microsecond=0)
  start_date = PullDate.from_dt(start_date)
  end_date = PullDate.from_dt(now)
  if not preprocessing:
    preprocessing = PreprocessingSchema()
  print(pull_history(columns,start_date=start_date,end_date=end_date,logger=logger,preprocessing=preprocessing))


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

  if logger:
    logger.info(f"Pull Realtime completed: {values}")

  return values

def pulling(dataset_name: str):
  """
  function ini untuk pulling dari Dataset (orm)
  """

  tic = time.monotonic()
  logger = Logger(dataset_name)
  logger.info(f"Enter Pull ... {dataset_name}")

  pull_time = 0

  db = next(get_session())
  dataset = Dataset.get_by_name(dataset_name, db)
  if not dataset:
    raise ValueError(f"{dataset_name} not found!")
  try:
    dataset.status = StatusProcess.RUNNING_PULL
    db.commit()

    columns = dataset.features
    if dataset.task_type.is_supervised():
      columns += [dataset.target]

    logger.info(f"Pull.. {dataset_name}")
    start_date = PullDate.from_str(dataset.start_date)
    end_date = PullDate.from_str(dataset.end_date)
    if not dataset.preprocessing: preprocessing = PreprocessingSchema()
    else: preprocessing = PreprocessingSchema(**dataset.preprocessing)

    df = pull_history( columns, start_date=start_date, end_date=end_date, logger=logger,preprocessing=preprocessing)
    pull_time += time.monotonic() - tic

    if df is None or df.empty: raise ValueError("Generated dataframe is empty")

    current_dt = df.iloc[-1]["dt"]

    # Rename columns from row_id to tagname dan update dataset.features/target
    columns_map = {}
    new_features = []
    new_target = None

    for col in df.columns:
      if col != "dt":
        try:
          # Convert row_id to tagname
          tagname = get_tagname(int(col))
        except (ValueError, TypeError):
          # If already a tagname, use as-is
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
    dataset.meta = MetaDataset(
        created_at=datetime.now().isoformat(),
        created_by="Anonymous",
        current_dt=str(current_dt),
        size_of=df.memory_usage().sum(),
        n_rows=n_rows,
        pulling_time=pull_time,
        n_cols=n_cols,
        missing_values=df.isna().sum().sum().item(),
        is_outlier=False,
        random_seed=4,
        columns=df.columns.tolist(),
        notes="",
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
    db.refresh(dataset)
    logger.info("Pulled successfully!")

  except Exception as e:
    logger.error(f"Dataset: {dataset_name} | Error: {str(e)}")
    dataset.is_valid = False

# =============================================================================
# Async Versions (Non-blocking)
# =============================================================================

async def async_get_realtime(
    tagname: str,
    logger=None,
    max_retries: int = 2,
    retry_delay: float = 0.5,
    timeout: int = 5,
) -> float:
  """
  Async version of get_realtime - Get realtime value without blocking event loop.

  Args:
      tagname: The identifier for the data point
      logger: Logger instance for logging
      max_retries: Maximum number of retry attempts
      retry_delay: Delay between retries in seconds
      timeout: Request timeout in seconds

  Returns:
      Current value as float
  """

  url = f"{Config.sl_host}application/api/modbus/get-point"
  params = dict(tagname=tagname)
  body = {"key": Config.key}
  last_exception = None

  client = await AsyncHTTPClient.get_client(verify=False)

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

      result_json = result.json()

      if not result_json:
        error_msg = f"Get Realtime ({tagname}) returned None"
        if logger: logger.error(error_msg)
        raise ValueError(error_msg)
      if not result_json.get("data", ""):
        raise ValueError(result_json)

      result_json = result_json["data"]
      currvalue = result_json.get("currvalue", None)

      if currvalue is None:
        error_msg = (
            f"Get Realtime ({tagname}) - 'currvalue' field is missing or None"
        )
        if logger:
          logger.error(error_msg)
        raise ValueError(error_msg)

      if logger:
        logger.info(
            f"Point: {tagname} | Successfully retrieved value: {currvalue}"
        )

      return float(currvalue)

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

  if last_exception:
    raise ValueError(
        f"Failed after {max_retries} attempts: {str(last_exception)}")

async def async_pull_realtime(columns: list, logger=None):
  """
  Async version of pull_realtime - Pull realtime values for multiple columns.

  Args:
      columns: List of column identifiers (tagnames)
      logger: Logger instance for logging

  Returns:
      List of realtime values corresponding to columns
  """
  if logger: logger.info(f"Pull Realtime ({len(columns)}) columns ...")

  # Create tasks untuk concurrent execution
  tasks = [async_get_realtime(col, logger=logger) for col in columns]

  # Execute all tasks concurrently
  results = await asyncio.gather(*tasks, return_exceptions=True)

  # Process results
  values = []
  for i, result in enumerate(results):
    if isinstance(result, Exception):
      if logger:
        logger.error(f"Failed to get realtime for {columns[i]}: {str(result)}")
      values.append(None)
    else:
      values.append(result)

  if logger:
    logger.info(f"Pull Realtime completed: {values}")

  return values

async def async_get_history(
    tagname: str,
    current_date: str,
    time_start: str = "00:00:00",
    time_end: str = "23:59:00",
    interval: int = 5,
    to_dataframe: bool = False,
    logger=None,
    max_retries: int = 3,
    retry_delay: float = 0.7,
    quality: int = 0,
    timeout: int = 10,
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

  row_id = get_row_id(tagname,logger)
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
  url = f"{Config.sl_host}application/api/tags/get-history"
  packet_json = json.dumps(packet)
  payload = {"packet": packet_json}
  last_exception = None

  client = await AsyncHTTPClient.get_client(verify=False)

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
        return (
            pd.DataFrame(columns=["dt", tagname]).set_index("dt")
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

  if last_exception: raise ValueError( f"Failed after {max_retries} attempts: {str(last_exception)}")

if __name__ == "__main__":
  from app.pull import pulling

  logger = Logger("test")
  tagnames = [
      "CRAH-2DH2.7-SUPPLY_AIR_TEMP",
      "CRAH-2DH2.100-SUPPLY_AIR_TEMP",
  ]
  tagname_e2 = "CRAH-2DH2.5-SUPPLY_AIR_TEMP"
  # print(asyncio.run(async_get_realtime(tagnames[0])))
  # start_date = PullDate(current_date="20260213",time_start="00:00:00",time_end="23:59:00")
  # end_date = PullDate(current_date="20260220",time_start="00:00:00",time_end="23:59:00")
  # preprocessing = PreprocessingSchema(missing_handling=DatasetHandling.NEIGHBOR_VALUE)
  # data = pull_history(columns=tagnames,start_date=start_date,end_date=end_date,logger=logger,preprocessing=preprocessing)
  now = PullDate.now()

  # data = pull_history(columns=tagnames,
  #                     start_date=PullDate(current_date="20260501",time_start="00:00:00"),
  #                     end_date=now,
  #                     logger=logger,preprocessing=PreprocessingSchema())
  data = get_history(tagname_e2,current_date="20260525",to_dataframe=True,interval=0,quality=4)
  print(data)
  # print(Config().display_connections_sl())
  # print(get_row_id(tagnames[1]))

