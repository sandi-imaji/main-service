"""
Routes initialization - aggregates all route modules
"""

from fastapi import APIRouter
from app.routes.router import dataset_router,model_router,stream_router,worker_router,utils_router

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
