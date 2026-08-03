import os
import pathlib
from datetime import timedelta, timezone
from dataclasses import dataclass
from dotenv import load_dotenv,set_key
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
  sl_retry:int = int(os.environ.get("SL_RETRY",2))
  sl_timeout: int = int(os.environ.get("SL_TIMEOUT",5))
  # Timeout untuk panggilan yang memang berat / lambat: query history satu hari
  # penuh dan lookup row_id. Server SL terukur butuh 5-7 detik, jadi memakai
  # SL_TIMEOUT (5 detik) di sini membuat sebagian besar request mati sia-sia.
  sl_timeout_long: int = int(os.environ.get("SL_TIMEOUT_LONG",30))
  # Server SL memakai sertifikat self-signed, jadi defaultnya tidak diverifikasi.
  # Set SL_VERIFY=1 kalau sertifikatnya sudah benar.
  sl_verify: bool = os.environ.get("SL_VERIFY", "0") not in ("0", "", "false", "False")
  # TTL (detik) cache nilai realtime per-tag yang dilayani proses API (endpoint
  # /utils/realtime, /utils/actual). Banyak client yang meminta tag yang sama
  # dalam jendela ini dilayani dari memori -> maksimal 1 hit upstream per TTL.
  realtime_ttl: float = float(os.environ.get("REALTIME_TTL", 5))
  # Berapa bagian horizon forecast yang boleh lewat sebelum model di-refit ke
  # data terbaru. Model meramal `fh` langkah dari UJUNG data latihnya, jadi
  # makin lama anchor itu dibiarkan makin banyak titik forecast yang jatuh ke
  # masa lalu. 1.0 = tunggu seluruh horizon lewat (rata-rata separuh grafik
  # sudah jadi masa lalu); 0.5 = refit di tengah horizon. Makin kecil makin
  # segar, tapi makin sering finetune (pull + pycaret).
  forecast_refresh_ratio: float = float(os.environ.get("FORECAST_REFRESH_RATIO", 0.5))

  # Logging: satu akar untuk semua log aplikasi (lihat app/logger.py).
  # `logs/` sengaja dipilih karena sudah ada di ignore_watch ecosystem dev —
  # menulis log tidak akan memicu restart-loop pm2.
  log_dir = dir / "logs"
  log_retention_days: int = int(os.environ.get("LOG_RETENTION_DAYS", 7))

  url: str = "https://10.3.13.1/beta2/application/api"
  key: str = os.environ.get("SL_KEY", "")
  verbose = Verbose(int(os.environ.get("MAIN_VERBOSE", 1)))
  debug_mode = bool(os.environ.get("DEBUGMODE", False))
  host = os.environ.get("NEXT_PUBLIC_HOST", "127.0.0.1")
  port = int(os.environ.get("PORT_MAIN", 8004))
  utc = timezone(timedelta(hours=7))
  window_size: int = 1_000

  # InfluxDB Configuration
  # Support multiple token environment variable names for compatibility
  influxdb_token: str = (
      os.environ.get("INFLUXDB_TOKEN")
      or os.environ.get("INFLUX_TOKEN")
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
  influxdb_retention: int = int(os.environ.get("INFLUXDB_RETENTION", 7))


  @classmethod
  def sl_url(cls, path: str) -> str:
    """Gabungkan sl_host dengan path endpoint.

    Dipakai supaya tidak ada lagi campuran f"{sl_host}/application/..." dan
    f"{sl_host}application/..." yang salah satunya selalu keliru tergantung
    ada tidaknya trailing slash di SL_HOST.
    """
    return f"{cls.sl_host.rstrip('/')}/{path.lstrip('/')}"

  def connection_str(self) -> str: return f"SL_HOST : {self.sl_host} | HOST : {self.host} | PORT : {self.port}"
  
  def prediction_database_str(self) -> str:
    return f"INLFUX URL : {self.influxdb_url} | INFLUX_ORG : {self.influxdb_org} | BUCKET{self.influxdb_bucket}"

  def dir_str(self) -> str: return f"DIR : {self.dir} | DIR_APP : {self.dir_app}"

  def debug_str(self) -> str: return f"DEBUG_MODE : {self.debug_mode} | VERBOSE : {self.verbose}"

  def display_connections_sl(self):
    print(f"IP : {self.sl_host}")
    print(f"TOKEN : {self.sl_token}")
    print(f"KEY : {self.sl_key}")
  
  @staticmethod
  def export() -> dict:
    return dict(sl_host=Config.sl_host,
                sl_key=Config.sl_key,
                sl_token=Config.sl_token,
                sl_retry=Config.sl_retry,
                sl_timeout=Config.sl_timeout,
                influxdb_token=Config.influxdb_token,
                influxdb_org=Config.influxdb_org,
                influxdb_bucket=Config.influxdb_bucket)
  
  @staticmethod
  def import_data(data:dict):
    data = data["data"]
    Config.sl_host = data['sl_host']
    Config.sl_key = data['sl_key']
    Config.sl_token = data['sl_token']
    Config.sl_retry = data['sl_retry']
    Config.sl_timeout = data['sl_timeout']
    Config.influxdb_token = data['influxdb_token']
    Config.influxdb_org = data['influxdb_org']
    Config.influxdb_bucket = data['influxdb_bucket']

  @staticmethod
  def update_config(data:dict):
    data = data["data"]
    for key in data.keys():
      if "sl" in key: set_key(dir.parent / ".env",key.upper(),data[key])

COLOR_MAP: dict = {
  "DEBUG": "\033[36m",  # Cyan
  "INFO": "\033[32m",  # Green
  "WARNING": "\033[33m",  # Yellow
  "ERROR": "\033[31m",  # Red
  "CRITICAL": "\033[41m",  # Red background
  "RESET": "\033[0m",
}

if __name__ == "__main__":
  print(Verbose(0))

  # import pprint
  # pprint.pprint(Config.sl_retry)
  # data = {
  #   "data":
  #     {
  #       "sl_retry": "10"
  #     }
  # }
  # Config.update_config(data)
  # pprint.pprint(Config.sl_retry)

