"""Manajemen log terpusat.

Satu akar (`Config.log_dir`), tiga tujuan:

    logs/global.log                              permukaan API + lifecycle worker
    logs/<TaskFolder>/<dataset>/main.log         streaming/inference/train/pulling
    logs/<TaskFolder>/<dataset>/worker.log       proses pm2 milik dataset itu

Folder task diturunkan dari NAMA dataset (`Regression-a1b2` → `Regression` →
`Supervised`), karena dataset memang dinamai `{task_type}-{uuid8}`. Efek
sampingnya pas: worker anomaly memakai nama dataset induknya, jadi lognya
otomatis mendarat di folder induk tanpa perlu cabang khusus.

Kenapa sink didaftarkan sekali lalu disaring, bukan dipasang ulang tiap
panggilan: `logger` loguru itu singleton global — `logger.remove()` menghapus
handler SEMUA logger, bukan cuma milik pemanggil. Versi lama memanggilnya di
tiap `Logger()`, sehingga membuat logger dataset B mematikan file dataset A dan
seluruh log berebut masuk ke file yang terakhir dibuat.
"""
import sys
from pathlib import Path
from typing import Dict, Optional

from loguru import logger as _loguru

from app.config import Config, Verbose


# Prefix nama dataset → folder task. Anomaly ikut Unsupervised: sebagai add-on
# ia memakai nama dataset induknya (jadi tidak pernah sampai ke sini), dan
# sebagai dataset primer ia memang unsupervised learning.
_TASK_FOLDER = {
  "Classification": "Supervised",
  "Regression": "Supervised",
  "Clustering": "Unsupervised",
  "Anomaly": "Unsupervised",
  "TimeSeries": "TimeSeries",
}

# Level per Verbose. SILENT tetap menulis ERROR ke file: kalau tidak, worker
# yang mati tengah malam tidak meninggalkan jejak apa pun.
_FILE_LEVEL = {
  Verbose.SILENT: "ERROR",
  Verbose.NORMAL: "INFO",
  Verbose.DEBUG: "DEBUG",
}

# Nama dataset ikut ditulis walau tiap dataset sudah punya filenya sendiri:
# global.log memuat banyak dataset, dan baris yang tersalin ke tiket/chat tetap
# bisa dilacak asalnya.
_FILE_FORMAT = ("[{time:YYYY-MM-DD HH:mm:ss}] [{level}] [{extra[dataset]}] "
                "{name}:{function}:{line} - {message}")
_STDOUT_FORMAT = (
  "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
  "<cyan>{extra[dataset]}</cyan> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:"
  "<cyan>{line}</cyan> - <level>{message}</level>"
)

MAIN = "main"
WORKER = "worker"


class LogManager:
  """Pemilik semua sink loguru di proses ini."""

  _file_sinks: Dict[str, int] = {}      # path log → handler id
  _stdout_sink: Optional[int] = None
  _initialised = False

  # --- Path --------------------------------------------------------------

  @staticmethod
  def task_folder(dataset_name: str) -> Optional[str]:
    """Folder task untuk sebuah nama dataset, atau None kalau bukan dataset."""
    prefix = str(dataset_name).split("-")[0]
    return _TASK_FOLDER.get(prefix)

  @classmethod
  def scope(cls, dataset_name: str, channel: str = MAIN) -> str:
    """Identitas tujuan log, RELATIF terhadap `Config.log_dir`.

    Sengaja relatif: inilah yang ditempelkan ke tiap record sebagai penanda
    tujuan. Kalau yang dipakai path absolut, logger yang sudah ter-bind akan
    menunjuk sink lama begitu `Config.log_dir` berubah (yang terjadi di test).
    """
    folder = cls.task_folder(dataset_name)
    if not folder:
      return "global"
    return f"{folder}/{dataset_name}/{channel}"

  @classmethod
  def log_path(cls, dataset_name: str, channel: str = MAIN) -> Path:
    """File tujuan. Nama yang bukan dataset (mis. "global", "influx_storage")
    jatuh ke global.log — jadi tidak ada log yang hilang tanpa jejak."""
    return Config.log_dir / f"{cls.scope(dataset_name, channel)}.log"

  # --- Sink --------------------------------------------------------------

  @classmethod
  def _init(cls) -> None:
    """Buang handler bawaan loguru — SEKALI seumur proses."""
    if cls._initialised:
      return
    _loguru.remove()
    cls._initialised = True

  @classmethod
  def _ensure_file_sink(cls, scope: str, path: Path) -> None:
    """Daftarkan sink file untuk `scope` kalau belum ada.

    Tiap sink hanya menerima record yang scope-nya cocok, sehingga file dataset
    A tidak mungkin kemasukan log dataset B.
    """
    if scope in cls._file_sinks:
      return

    path.parent.mkdir(parents=True, exist_ok=True)
    cls._file_sinks[scope] = _loguru.add(
        path,
        format=_FILE_FORMAT,
        level=_FILE_LEVEL[Config.verbose],
        rotation="1 day",
        retention=f"{Config.log_retention_days} days",
        filter=lambda record, s=scope: record["extra"].get("scope") == s,
        colorize=False,
        buffering=1,          # line-buffered: WS log tail langsung melihat baris baru
        encoding="utf-8",
        delay=True,           # file baru dibuat saat ada yang ditulis — di SILENT
                              # tidak ada pohon folder kosong yang menyesatkan
    )

  @classmethod
  def _ensure_stdout_sink(cls) -> None:
    """stdout HANYA saat DEBUG. Inilah yang membuat log pm2 worker terkendali:
    di NORMAL/SILENT `~/.pm2/logs/*-out.log` tetap kosong, riwayatnya ada di
    worker.log yang punya rotasi dan retensi."""
    if Config.verbose != Verbose.DEBUG or cls._stdout_sink is not None:
      return
    cls._stdout_sink = _loguru.add(
        sys.stdout, level="DEBUG", format=_STDOUT_FORMAT, colorize=True)

  # --- API ---------------------------------------------------------------

  @classmethod
  def get(cls, dataset_name: str = "", channel: str = MAIN):
    """Logger untuk satu dataset+channel. Aman dipanggil berkali-kali."""
    cls._init()
    scope = cls.scope(dataset_name, channel)
    cls._ensure_file_sink(scope, cls.log_path(dataset_name, channel))
    cls._ensure_stdout_sink()
    return _loguru.bind(scope=scope, dataset=str(dataset_name) or "global")

  @classmethod
  def global_logger(cls):
    """Logger permukaan: route + lifecycle worker."""
    return cls.get("global")

  @classmethod
  def worker(cls, dataset_name: str):
    """Logger proses pm2 milik sebuah dataset."""
    return cls.get(dataset_name, WORKER)

  @classmethod
  def reset(cls) -> None:
    """Lepas semua sink. Dipakai test yang mengubah Config (verbose/log_dir)."""
    _loguru.remove()
    cls._file_sinks.clear()
    cls._stdout_sink = None
    cls._initialised = False


class LoggerNone:
  """Logger yang menelan semuanya. Dipakai jalur yang sengaja tidak mencatat."""
  def __init__(self, dataset_name: str = ""): pass
  def info(self, message: str): pass
  def error(self, message: str): pass
  def warning(self, message: str): pass
  def debug(self, message: str): pass
  def success(self, message: str): pass


def Logger(dataset_name: str = "", channel: str = MAIN):
  """Bentuk pemanggilan lama — seluruh service/route masih memakai ini."""
  return LogManager.get(dataset_name, channel)


class _GlobalLogger:
  """Proxy tipis ke logger global.

  Sengaja TIDAK menyimpan objek logger hasil bind: `LOGGER_GLOBAL` dibuat sekali
  saat import dan dipakai seumur proses, sementara sink-nya bisa didaftarkan
  ulang (test yang memindah `Config.log_dir`, atau `reset()`). Menyimpan objek
  lamanya berarti log diam-diam hilang setelah pendaftaran ulang; menyelesaikan
  logger-nya saat dipakai membuat itu tidak mungkin terjadi.
  """

  def __getattr__(self, name):
    return getattr(LogManager.global_logger(), name)


LOGGER_GLOBAL = _GlobalLogger()
