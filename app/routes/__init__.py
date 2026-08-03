"""
Routes initialization - aggregates all route modules
"""

from fastapi import APIRouter
from app.routes.datasets import router as dataset_router
from app.routes.models import router as model_router
from app.routes.stream import router as stream_router
from app.routes.workers import router as worker_router
from app.routes.utils import router as utils_router

# Main router that combines all sub-routers
main_router = APIRouter()

# Include all route modules with their prefixes
main_router.include_router(dataset_router, prefix="/datasets")
main_router.include_router(model_router, prefix="/models")
main_router.include_router(stream_router, prefix="/stream")
main_router.include_router(worker_router, prefix="/workers")
main_router.include_router(utils_router, prefix="/utils")

# Backward compatibility - router is the main router
router = main_router
