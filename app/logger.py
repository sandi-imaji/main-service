from re import VERBOSE
import sys
import asyncio
import aiofiles
import logging
from pathlib import Path
from typing import Optional
from loguru import logger
from app.config import Config,Verbose
from watchdog.events import FileSystemEventHandler

# Contoh DIR, sesuaikan
DIR = Path(".")

class LoggerNone:
  def __init__(self,dataset_name:str):pass
  def info(self,message:str):pass
  def error(self,message:str):pass
  def warning(self,message:str):pass
  def debug(self,message:str):pass

def Logger(dataset_name: str):
  """
  Mengembalikan logger unik untuk tiap dataset_name.
  Jika VERBOSE=1 → tampil juga di terminal (dengan warna default loguru).
  Jika tidak, hanya menulis ke file.
  """
  if dataset_name == "":
    log_dir = DIR / "logger"
  else:
    log_dir = DIR / "storages" / dataset_name / "logs"

  log_dir.mkdir(parents=True, exist_ok=True)
  log_path = log_dir / "main.log"

  # Buat logger baru dengan bind untuk identifikasi
  log = logger.bind(dataset=dataset_name)

  # Hapus semua handler default loguru
  log.remove()
  
  # === Console handler jika VERBOSE=1 ===
  log.add(
      sys.stdout,
      level=Config.verbose.log_level(),
      format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
      colorize=True,
  )

  if not Config.verbose == Verbose.SILENT:
    # === File handler ===
    log.add(
        log_path,
        format="[{time:YYYY-MM-DD HH:mm:ss}] [{level}] {name}:{function}:{line} - {message}",
        rotation="5 MB",
        retention=3,
        level="INFO",
        colorize=False,
        buffering=1,
    )

  return log


LOGGER_GLOBAL = Logger("global")
