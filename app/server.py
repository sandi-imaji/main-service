"""
Smart AI Main Service - FastAPI Application
"""

from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routes import router
from app.database.orm import Dataset
from app.database.db import get_session
from app.database.schemas import StatusProcess
from app.config import Config,Verbose
from app.logger import LOGGER_GLOBAL
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

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

# Include all routes
app.include_router(router)


if __name__ == "__main__":
  # LOGGER_GLOBAL.debug(Config().connection_str())
  # LOGGER_GLOBAL.debug(Config().debug_str())

  if Config.verbose == Verbose.SILENT:
    log_level = "critical"
    access_log = False
  else:
    log_level = 'info'
    access_log = True

  uvicorn.run("server:app", host="0.0.0.0", port=Config.port, reload=True,log_level=log_level,access_log=access_log)

