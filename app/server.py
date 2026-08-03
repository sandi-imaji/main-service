"""
Smart AI Main Service - FastAPI Application
"""

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.config import Config, Verbose
from app.database.base import init_db
from app.logger import LOGGER_GLOBAL
from app.routes import router

# Create FastAPI app
app = FastAPI(title="Smart AI")

# CORS middleware - MUST be added before routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
  """Catat setiap request ke global.log.

  Sengaja middleware, bukan dekorator per route: dekorator harus dipasang satu
  per satu (dan dulu cuma terpasang di 2 dari ~35 route), sementara di sini
  setiap endpoint tercatat otomatis — termasuk yang ditambahkan besok.

  Levelnya mengikuti status supaya masih berguna saat VERBOSE=SILENT: di mode
  itu hanya 5xx yang tersimpan, yaitu justru yang perlu dilihat.
  """
  start = time.perf_counter()
  endpoint = f"{request.method} {request.url.path}"
  try:
    response = await call_next(request)
  except Exception as e:
    elapsed = (time.perf_counter() - start) * 1000
    LOGGER_GLOBAL.error(f"{endpoint} -> unhandled ({elapsed:.0f}ms): {e}")
    raise

  elapsed = (time.perf_counter() - start) * 1000
  message = f"{endpoint} -> {response.status_code} ({elapsed:.0f}ms)"
  if response.status_code >= 500:
    LOGGER_GLOBAL.error(message)
  elif response.status_code >= 400:
    LOGGER_GLOBAL.warning(message)
  else:
    LOGGER_GLOBAL.info(message)
  return response


# Include all routes
app.include_router(router)

if __name__ == "__main__":
  rtdb_path = Config.dir / "storages/rtdb/data.db"
  if rtdb_path.exists():
    init_db()

  log_level = "critical" if Config.verbose == Verbose.SILENT else "info"
  # access_log uvicorn dimatikan: middleware di atas sudah mencatat hal yang
  # sama ke global.log — yang punya rotasi, retensi, dan ikut aturan Verbose.
  uvicorn.run("server:app", host=Config.host, port=Config.port, reload=True,
              log_level=log_level, access_log=False)
