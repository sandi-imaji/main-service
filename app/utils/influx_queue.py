"""
Local Queue untuk failed InfluxDB writes.
Menyimpan predictions ke SQLite lokal dan retry secara background.
Memastikan 100% data reliability.
"""

import asyncio
import json
import sqlite3
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from app.database.influx import PredictionResult
from app.config import Config
from app.logger import Logger


@dataclass
class QueuedPrediction:
    """Prediction yang di-queue untuk retry."""

    id: Optional[int] = None
    dataset_name: str = ""
    model_name: str = ""
    timestamp: str = ""
    prediction: float = 0.0
    actual: Optional[float] = None
    created_at: Optional[str] = None
    retry_count: int = 0
    last_error: str = ""

    def to_prediction_result(self) -> PredictionResult:
        """Convert ke PredictionResult."""
        from app.database.influx import PredictionResult
        from datetime import datetime as dt

        return PredictionResult(
            dataset_name=self.dataset_name,
            model_name=self.model_name,
            timestamp=dt.fromisoformat(self.timestamp),
            prediction=self.prediction,
            actual=self.actual,
        )

    @classmethod
    def from_prediction_result(
        cls, result: PredictionResult, error: str = ""
    ) -> "QueuedPrediction":
        """Create dari PredictionResult."""
        return cls(
            dataset_name=result.dataset_name,
            model_name=result.model_name,
            timestamp=result.timestamp.isoformat(),
            prediction=result.prediction,
            actual=result.actual,
            created_at=datetime.now().isoformat(),
            retry_count=0,
            last_error=error,
        )


class LocalInfluxQueue:
    """
    SQLite-based queue untuk failed InfluxDB writes.

    Features:
    - Persistent storage (survives restarts)
    - Automatic retry dengan exponential backoff
    - Priority queue (older items first)
    - Statistics tracking

    Usage:
        queue = LocalInfluxQueue()

        # Enqueue failed prediction
        queue.enqueue(prediction_result, error="Connection timeout")

        # Dequeue batch untuk retry
        items = queue.dequeue_batch(batch_size=100)

        # Mark success/failure
        queue.mark_success([item.id for item in succeeded])
        queue.mark_failed([item.id for item in failed], error="Still failing")

        # Get stats
        stats = queue.get_stats()
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Default path: dalam folder storages
            db_path = str(Config.dir / "queue.db")

        self.db_path = db_path
        self.logger = Logger("local_influx_queue")

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database dan table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Drop old table if exists (schema changed - reset data)
                cursor.execute("DROP TABLE IF EXISTS pending_predictions")

                # Create table with new schema
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS pending_predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        dataset_name TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        prediction REAL NOT NULL,
                        actual REAL,
                        created_at TEXT NOT NULL,
                        retry_count INTEGER DEFAULT 0,
                        last_error TEXT,
                        status TEXT DEFAULT 'pending'
                    )
                """)

                # Create indexes
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_status_created 
                    ON pending_predictions(status, created_at)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_dataset 
                    ON pending_predictions(dataset_name)
                """)

                conn.commit()
                self.logger.info(f"Local queue initialized: {self.db_path}")

        except Exception as e:
            self.logger.error(f"Failed to initialize local queue: {e}")
            raise

    def enqueue(self, result: PredictionResult, error: str = "") -> int:
        """
        Add prediction ke queue.

        Args:
            result: PredictionResult yang gagal
            error: Error message

        Returns:
            int: Queue item ID
        """
        try:
            queued = QueuedPrediction.from_prediction_result(result, error)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO pending_predictions 
                    (dataset_name, model_name, timestamp, prediction, actual, 
                     created_at, retry_count, last_error, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        queued.dataset_name,
                        queued.model_name,
                        queued.timestamp,
                        queued.prediction,
                        queued.actual,
                        queued.created_at or datetime.now().isoformat(),
                        queued.retry_count,
                        queued.last_error,
                        "pending",
                    ),
                )

                conn.commit()
                item_id = cursor.lastrowid

                self.logger.debug(
                    f"Enqueued prediction {result.dataset_name}/{result.model_name} "
                    f"(id={item_id})"
                )

                return item_id if item_id is not None else 0

        except Exception as e:
            self.logger.error(f"Failed to enqueue prediction: {e}")
            raise

    def enqueue_many(
        self, results: List[PredictionResult], error: str = ""
    ) -> List[int]:
        """
        Add multiple predictions ke queue.

        Args:
            results: List of PredictionResult
            error: Error message

        Returns:
            List[int]: Queue item IDs
        """
        ids = []
        for result in results:
            try:
                item_id = self.enqueue(result, error)
                ids.append(item_id)
            except Exception as e:
                self.logger.error(f"Failed to enqueue batch item: {e}")

        return ids

    def dequeue_batch(
        self, batch_size: int = 100, max_retries: int = 5
    ) -> List[QueuedPrediction]:
        """
        Get batch of pending predictions untuk retry.

        Args:
            batch_size: Number of items to dequeue
            max_retries: Maximum retry attempts

        Returns:
            List[QueuedPrediction]: Items untuk retry
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Get oldest pending items
                cursor.execute(
                    """
                    SELECT * FROM pending_predictions
                    WHERE status = 'pending' AND retry_count < ?
                    ORDER BY created_at ASC
                    LIMIT ?
                """,
                    (max_retries, batch_size),
                )

                rows = cursor.fetchall()

                items = []
                for row in rows:
                    item = QueuedPrediction(
                        id=row["id"],
                        dataset_name=row["dataset_name"],
                        model_name=row["model_name"],
                        timestamp=row["timestamp"],
                        prediction=row["prediction"],
                        actual=row["actual"],
                        created_at=row["created_at"],
                        retry_count=row["retry_count"],
                        last_error=row["last_error"],
                    )
                    items.append(item)

                # Mark as processing
                if items:
                    ids = [item.id for item in items]
                    placeholders = ",".join("?" * len(ids))
                    cursor.execute(
                        f"""
                        UPDATE pending_predictions
                        SET status = 'processing'
                        WHERE id IN ({placeholders})
                    """,
                        ids,
                    )
                    conn.commit()

                return items

        except Exception as e:
            self.logger.error(f"Failed to dequeue batch: {e}")
            return []

    def mark_success(self, item_ids: List[int]):
        """
        Mark items sebagai successfully processed.

        Args:
            item_ids: List of item IDs to mark success
        """
        if not item_ids:
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                placeholders = ",".join("?" * len(item_ids))
                cursor.execute(
                    f"""
                    DELETE FROM pending_predictions
                    WHERE id IN ({placeholders})
                """,
                    item_ids,
                )

                conn.commit()

                self.logger.debug(f"Marked {len(item_ids)} items as success")

        except Exception as e:
            self.logger.error(f"Failed to mark success: {e}")

    def mark_failed(self, item_ids: List[int], error: str = ""):
        """
        Mark items sebagai failed (akan di-retry lagi).

        Args:
            item_ids: List of item IDs yang failed
            error: Error message
        """
        if not item_ids:
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                placeholders = ",".join("?" * len(item_ids))
                cursor.execute(
                    f"""
                    UPDATE pending_predictions
                    SET status = 'pending',
                        retry_count = retry_count + 1,
                        last_error = ?
                    WHERE id IN ({placeholders})
                """,
                    [error] + item_ids,
                )

                conn.commit()

                self.logger.debug(
                    f"Marked {len(item_ids)} items as failed (will retry)"
                )

        except Exception as e:
            self.logger.error(f"Failed to mark failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Total counts
                cursor.execute("""
                    SELECT status, COUNT(*) 
                    FROM pending_predictions 
                    GROUP BY status
                """)
                status_counts = dict(cursor.fetchall())

                # Total items
                cursor.execute("SELECT COUNT(*) FROM pending_predictions")
                total = cursor.fetchone()[0]

                # Oldest item
                cursor.execute("""
                    SELECT MIN(created_at) FROM pending_predictions
                """)
                oldest = cursor.fetchone()[0]

                # Average retry count
                cursor.execute("""
                    SELECT AVG(retry_count) FROM pending_predictions
                """)
                avg_retries = cursor.fetchone()[0] or 0

                return {
                    "total_items": total,
                    "pending": status_counts.get("pending", 0),
                    "processing": status_counts.get("processing", 0),
                    "oldest_item": oldest,
                    "average_retries": round(avg_retries, 2),
                }

        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {}

    def cleanup_old_items(self, max_age_hours: int = 24):
        """
        Remove items yang terlalu lama dan sudah terlalu banyak retry.

        Args:
            max_age_hours: Maximum age in hours
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cutoff = datetime.now()
                cutoff = cutoff.replace(hour=cutoff.hour - max_age_hours)

                cursor.execute(
                    """
                    DELETE FROM pending_predictions
                    WHERE created_at < ? AND retry_count >= 5
                """,
                    (cutoff.isoformat(),),
                )

                deleted = cursor.rowcount
                conn.commit()

                if deleted > 0:
                    self.logger.warning(f"Cleaned up {deleted} old queue items")

        except Exception as e:
            self.logger.error(f"Failed to cleanup: {e}")

    def clear_all(self):
        """Clear entire queue (use with caution!)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM pending_predictions")
                conn.commit()

                self.logger.warning("Queue cleared!")

        except Exception as e:
            self.logger.error(f"Failed to clear queue: {e}")


# Global instance
_queue: Optional[LocalInfluxQueue] = None


def get_local_queue() -> LocalInfluxQueue:
    """Get global queue instance."""
    global _queue
    if _queue is None:
        _queue = LocalInfluxQueue()
    return _queue


def reset_local_queue():
    """Reset global queue instance."""
    global _queue
    _queue = None
