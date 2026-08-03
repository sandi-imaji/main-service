"""
In-Memory Worker State Management
No SQLite, no persistence - pure in-memory with Dict storage
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import asyncio

# Logger internal, BUKAN `logging` bawaan Python. Versi sebelumnya memakai
# `logging.getLogger(__name__)`: root logger tidak punya handler dan tidak ada
# jembatan stdlib→loguru, sehingga `info`/`debug` hilang tanpa jejak dan
# `warning` bocor ke stderr lewat `logging.lastResort` — tidak pernah sampai ke
# berkas di `Config.log_dir`. Siklus hidup worker termasuk yang harus tercatat
# logger global.
from app.logger import LOGGER_GLOBAL as logger


class WorkerStatus(str, Enum):
    """Worker lifecycle states."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"
    TRAINING = "training"
    RELOAD_REQUIRED = "reload_required"


@dataclass
class WorkerState:
    """In-memory worker state (no database persistence)."""

    dataset_name: str
    task_type: str
    status: WorkerStatus = WorkerStatus.IDLE
    interval_seconds: int = 300

    # Timing
    started_at: Optional[datetime] = None
    last_run_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=datetime.now)

    # Metrics
    total_predictions: int = 0
    error_count: int = 0
    consecutive_errors: int = 0
    last_error: Optional[str] = None

    # Model versioning
    model_version: int = 1
    latest_model_version: int = 1

    # Extra config
    extra_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.started_at is None:
            self.started_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "dataset_name": self.dataset_name,
            "status": self.status.value,
            "task_type": self.task_type,
            "interval_seconds": self.interval_seconds,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_run_at": self.last_run_at.isoformat() if self.last_run_at else None,
            "updated_at": self.updated_at.isoformat(),
            "total_predictions": self.total_predictions,
            "error_count": self.error_count,
            "consecutive_errors": self.consecutive_errors,
            "last_error": self.last_error,
            "model_version": self.model_version,
            "latest_model_version": self.latest_model_version,
        }

    @property
    def is_running(self) -> bool:
        return self.status == WorkerStatus.RUNNING

    @property
    def needs_reload(self) -> bool:
        return self.model_version < self.latest_model_version


class WorkerStateManager:
    """
    In-memory worker state manager using Dict.
    No persistence - all data lost on restart (by design).
    """

    def __init__(self):
        self._states: Dict[str, WorkerState] = {}
        self._lock = asyncio.Lock()
        # Sengaja TIDAK mencatat apa pun di sini: instance singleton di bawah
        # dibuat saat modul diimpor, dan menyentuh logger di titik itu membuat
        # `Config.log_dir` terbentuk sebagai efek samping impor — termasuk saat
        # pytest mengumpulkan test, sebelum conftest sempat mengalihkannya.

    async def create_worker(
        self,
        dataset_name: str,
        task_type: str,
        interval_seconds: int = 300,
    ) -> Optional[WorkerState]:
        """Create new worker state."""
        async with self._lock:
            if dataset_name in self._states:
                logger.debug(f"Worker state already exists: {dataset_name}")
                return self._states[dataset_name]

            worker = WorkerState(
                dataset_name=dataset_name,
                task_type=task_type,
                interval_seconds=interval_seconds,
                started_at=datetime.now(),
            )
            self._states[dataset_name] = worker
            logger.info(f"Worker state created: {dataset_name}")
            return worker

    async def get_worker(self, dataset_name: str) -> Optional[WorkerState]:
        """Get worker state."""
        async with self._lock:
            return self._states.get(dataset_name)

    async def get_all_workers(
        self,
        status: Optional[WorkerStatus] = None,
    ) -> List[WorkerState]:
        """Get all worker states."""
        async with self._lock:
            workers = list(self._states.values())
            if status:
                workers = [w for w in workers if w.status == status]
            return workers

    async def get_running_workers(self) -> List[WorkerState]:
        """Get all running workers."""
        return await self.get_all_workers(status=WorkerStatus.RUNNING)

    async def update_status(
        self,
        dataset_name: str,
        status: WorkerStatus,
        error_message: Optional[str] = None,
    ) -> bool:
        """Update worker status."""
        async with self._lock:
            worker = self._states.get(dataset_name)
            if not worker:
                logger.warning(f"Worker not found for status update: {dataset_name}")
                return False

            worker.status = status
            worker.updated_at = datetime.now()

            if error_message:
                worker.last_error = error_message
                worker.error_count += 1
                if status == WorkerStatus.ERROR:
                    worker.consecutive_errors += 1
            else:
                if status == WorkerStatus.RUNNING:
                    worker.consecutive_errors = 0

            logger.debug(f"Worker status updated: {dataset_name} -> {status.value}")
            return True

    async def update_metrics(
        self,
        dataset_name: str,
        predictions_count: int = 0,
        error: Optional[str] = None,
    ) -> bool:
        """Update worker metrics after prediction run."""
        async with self._lock:
            worker = self._states.get(dataset_name)
            if not worker:
                return False

            worker.last_run_at = datetime.now()
            worker.total_predictions += predictions_count
            worker.updated_at = datetime.now()

            if error:
                worker.error_count += 1
                worker.consecutive_errors += 1
                worker.last_error = error
            else:
                worker.consecutive_errors = 0

            return True

    async def delete_worker(self, dataset_name: str) -> bool:
        """Delete worker state."""
        async with self._lock:
            if dataset_name in self._states:
                del self._states[dataset_name]
                logger.info(f"Worker state deleted: {dataset_name}")
                return True
            return False

    async def clear_all(self) -> int:
        """Clear all worker states."""
        async with self._lock:
            count = len(self._states)
            self._states.clear()
            logger.info(f"Cleared {count} worker states")
            return count

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            "total_workers": len(self._states),
            "by_status": {
                status.value: len(
                    [w for w in self._states.values() if w.status == status]
                )
                for status in WorkerStatus
            },
        }


# Global instance (singleton)
state_manager = WorkerStateManager()
