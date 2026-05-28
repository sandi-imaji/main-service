"""
Batch InfluxDB Writer untuk optimasi performance.
Mengurangi network calls dengan mengumpulkan predictions sebelum write.
"""

import asyncio
import time
from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from app.database.influx import InfluxDBResultsClient, PredictionResult
from app.logger import Logger


@dataclass
class BatchConfig:
  """Configuration untuk batch writer."""

  batch_size: int = 50  # Flush setelah N predictions
  flush_interval: float = 5.0  # Flush setiap N detik
  max_retries: int = 3
  retry_delay: float = 1.0


class InfluxDBBatchWriter:
  """
  Batch writer untuk InfluxDB dengan auto-flush.

  Features:
  - Automatic batching berdasarkan size atau interval
  - Retry mechanism untuk failed writes
  - Thread-safe (asyncio.Lock)
  - Statistics tracking

  Usage:
      writer = InfluxDBBatchWriter()
      await writer.start()

      # Write single prediction
      await writer.write(prediction)

      # Flush manually if needed
      await writer.flush()

      # Stop writer
      await writer.stop()
  """

  def __init__(
      self,
      influx_client: Optional[InfluxDBResultsClient] = None,
      config: Optional[BatchConfig] = None,
  ):
    self.config = config or BatchConfig()
    self.client = influx_client or InfluxDBResultsClient()
    self.logger = Logger("influxdb_batch")

    # Batch queue
    self._batch: List[PredictionResult] = []
    self._lock = asyncio.Lock()

    # Background task
    self._flush_task: Optional[asyncio.Task] = None
    self._running = False

    # Statistics
    self._stats = {
        "total_written": 0,
        "total_failed": 0,
        "batches_flushed": 0,
        "last_flush": None,
    }

  async def start(self):
    """Start background flush task."""
    if self._running:
      return

    self._running = True
    self._flush_task = asyncio.create_task(self._flush_loop())
    self.logger.info(
        f"Batch writer started (size={self.config.batch_size}, "
        f"interval={self.config.flush_interval}s)"
    )

  async def stop(self):
    """Stop writer dan flush remaining batch."""
    if not self._running:
      return

    self._running = False

    # Cancel flush task
    if self._flush_task:
      self._flush_task.cancel()
      try:
        await self._flush_task
      except asyncio.CancelledError:
        pass

    # Flush remaining
    await self.flush()
    self.logger.info("Batch writer stopped")

  async def _flush_loop(self):
    """Background loop untuk periodic flush."""
    while self._running:
      try:
        await asyncio.sleep(self.config.flush_interval)
        await self._check_and_flush()
      except asyncio.CancelledError:
        break
      except Exception as e:
        self.logger.error(f"Error in flush loop: {e}")

  async def write(self, result: PredictionResult) -> bool:
    """
    Write prediction to batch.

    Args:
        result: PredictionResult to write

    Returns:
        bool: True if queued successfully
    """
    async with self._lock:
      self._batch.append(result)

      # Check if should flush
      if len(self._batch) >= self.config.batch_size:
        asyncio.create_task(self._check_and_flush())

    return True

  async def write_many(self, results: List[PredictionResult]) -> bool:
    """
    Write multiple predictions to batch.

    Args:
        results: List of PredictionResult

    Returns:
        bool: True if queued successfully
    """
    async with self._lock:
      self._batch.extend(results)

      # Check if should flush
      if len(self._batch) >= self.config.batch_size:
        asyncio.create_task(self._check_and_flush())

    return True

  async def _check_and_flush(self):
    """Check conditions dan flush jika perlu."""
    async with self._lock:
      if len(self._batch) >= self.config.batch_size:
        await self._do_flush()

  async def flush(self) -> tuple[int, int]:
    """
    Force flush current batch.

    Returns:
        tuple: (success_count, failed_count)
    """
    async with self._lock:
      return await self._do_flush()

  async def _do_flush(self) -> tuple[int, int]:
    """
    Internal flush implementation.

    Returns:
        tuple: (success_count, failed_count)
    """
    if not self._batch:
      return 0, 0

    # Copy batch dan clear
    batch_to_flush = self._batch.copy()
    self._batch = []

    # Write dengan retry
    success_count = 0
    failed_count = 0
    failed_results = []

    for attempt in range(self.config.max_retries):
      try:
        success, failed = self.client.write_predictions_batch(batch_to_flush)
        success_count += success
        failed_count += failed

        if failed == 0:
          break  # All successful

        # Retry failed ones
        if attempt < self.config.max_retries - 1:
          await asyncio.sleep(self.config.retry_delay * (attempt + 1))
          # Filter hanya yang failed untuk retry
          # Note: Saat ini write_predictions_batch tidak return which ones failed
          # Jadi kita retry semua

      except Exception as e:
        self.logger.error(f"Flush attempt {attempt + 1} failed: {e}")
        if attempt < self.config.max_retries - 1:
          await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        else:
          # All retries failed
          failed_count = len(batch_to_flush)

    # Update stats
    self._stats["total_written"] += success_count
    self._stats["total_failed"] += failed_count
    self._stats["batches_flushed"] += 1
    self._stats["last_flush"] = datetime.now().isoformat()

    self.logger.debug(
        f"Flushed batch: {success_count} success, {failed_count} failed"
    )

    return success_count, failed_count

  def get_stats(self) -> dict:
    """Get writer statistics."""
    return {
        **self._stats,
        "current_batch_size": len(self._batch),
        "is_running": self._running,
    }

  def __del__(self):
    """Cleanup on deletion."""
    try:
      if hasattr(self, "_batch") and self._batch:
        self.logger.warning(
            f"Batch writer destroyed with {len(self._batch)} pending items"
        )
    except Exception:
      # Ignore any errors during destruction
      pass


# Global instance (singleton)
_batch_writer: Optional[InfluxDBBatchWriter] = None


async def get_batch_writer() -> InfluxDBBatchWriter:
  """Get or create global batch writer instance."""
  global _batch_writer
  if _batch_writer is None:
    _batch_writer = InfluxDBBatchWriter()
    await _batch_writer.start()
  return _batch_writer


async def stop_batch_writer():
  """Stop global batch writer."""
  global _batch_writer
  if _batch_writer:
    await _batch_writer.stop()
    _batch_writer = None
