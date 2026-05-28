import os
import pathlib
from datetime import timedelta, timezone
from dataclasses import dataclass
from dotenv import load_dotenv
from enum import Enum,auto

DELAY = int(os.environ.get("DELAY", 0))
WORKER = os.environ.get("WORKER") != "0"

dir_app = pathlib.Path(__file__).parent
dir = dir_app.parent

load_dotenv(dir.parent / ".env")

class Verbose(Enum):
  SILENT = 0
  NORMAL = auto()
  DEBUG = auto()

  def log_level(self) -> str :
    return {
    Verbose.SILENT: "ERROR",      # Hanya ERROR dan CRITICAL
    Verbose.NORMAL: "INFO",       # INFO, WARNING, ERROR, CRITICAL
    Verbose.DEBUG: "DEBUG",     # Semua level (DEBUG + INFO + ...)
    }[self]

  @property
  def pycaret(self) -> bool: return self != Verbose.SILENT



@dataclass
class Config:
  dir_app = dir_app
  dir = dir
  sl_host: str = os.environ.get("SL_HOST", "http://127.0.0.1")
  sl_key: str = os.environ.get("SL_KEY", "")
  sl_token: str = os.environ.get("SL_TOKEN","")
  url: str = "https://10.3.13.1/beta2/application/api"
  key: str = os.environ.get("SL_KEY", "")
  verbose = Verbose(int(os.environ.get("VERBOSE", 1)))
  debug_mode = bool(os.environ.get("DEBUGMODE", False))
  host = os.environ.get("HOST_MAIN", "")
  port = int(os.environ.get("PORT_MAIN", 8080))
  utc = timezone(timedelta(hours=7))
  window_size: int = 1_000

  # InfluxDB Configuration
  # Support multiple token environment variable names for compatibility
  influxdb_token: str = (
      os.environ.get("INFLUXDB_TOKEN")
      or os.environ.get("INFLUXDB3_AUTH_TOKEN")
      or os.environ.get("INFLUXDB3_AUTH_TOKEN_")
      or ""
  )

  influxdb_headers = {
      "Authorization": f"Token {influxdb_token}",
      "Content-Type": "application/json",
  }
  influxdb_url: str = os.environ.get("INFLUXDB_URL", "http://localhost:8086")
  influxdb_org: str = os.environ.get("INFLUXDB_ORG", "tech")
  influxdb_bucket: str = os.environ.get("INFLUXDB_BUCKET", "ml-buckets")
  influxdb_retention_days: int = int(
      os.environ.get("INFLUXDB_RETENTION_DAYS", 30))
  prediction_batch_size: int = int(
      os.environ.get("PREDICTION_BATCH_SIZE", 100))
  prediction_batch_timeout: int = int(
      os.environ.get("PREDICTION_BATCH_TIMEOUT", 60))


  def connection_str(self) -> str: return f"SL_HOST : {self.sl_host} | HOST : {self.host} | PORT : {self.port}"
  
  def prediction_database_str(self) -> str:
    return f"INLFUX URL : {self.influxdb_url} | INFLUX_ORG : {self.influxdb_org} | BUCKET{self.influxdb_bucket}"

  def dir_str(self) -> str: return f"DIR : {self.dir} | DIR_APP : {self.dir_app}"

  def debug_str(self) -> str: return f"DEBUG_MODE : {self.debug_mode} | VERBOSE : {self.verbose}"

  def display_connections_sl(self):
    print(f"IP : {self.sl_host}")
    print(f"TOKEN : {self.sl_token}")
    print(f"KEY : {self.sl_key}")


COLOR_MAP: dict = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[41m",  # Red background
    "RESET": "\033[0m",
}



if __name__ == "__main__":
  print(Config.influxdb_token)
