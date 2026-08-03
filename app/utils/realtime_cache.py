"""In-process TTL cache for realtime tag values (API process only).

Purpose: when realtime reads are client-driven, many clients (or one client
polling fast) would otherwise turn into one upstream SL API call *each* for the
same tag. This cache collapses them:

  * fan-out     - reads of a tag within its TTL are served from memory, so N
                  callers cost at most 1 upstream call per TTL window.
  * single-flight - when a tag is cold/expired and several callers race, only
                  one actually fetches; the rest await that same fetch.

Scope: lives in the API process only. pm2 worker loops run in separate
processes and do NOT share this cache (they'd need an external store for that).

The upstream fetch and the clock are injected, so the cache's own logic is fully
testable without any network or real time. The module-level `realtime_cache`
singleton wires in the real async fetch (app.pull.async_get_realtime).
"""
import asyncio
import time
from typing import Awaitable, Callable, Dict, Optional, Tuple

from app.config import Config


class RealtimeCache:
  def __init__(
      self,
      ttl: float,
      fetch: Callable[[str], Awaitable[float]],
      clock: Callable[[], float] = time.monotonic,
  ):
    self._ttl = ttl
    self._fetch = fetch
    self._clock = clock
    self._values: Dict[str, Tuple[float, float]] = {}   # tag -> (value, fetched_at)
    self._locks: Dict[str, asyncio.Lock] = {}

  def _fresh(self, tag: str) -> Optional[float]:
    """The cached value if still within TTL, else None."""
    hit = self._values.get(tag)
    if hit is not None and self._clock() - hit[1] < self._ttl:
      return hit[0]
    return None

  async def get(self, tag: str) -> float:
    cached = self._fresh(tag)
    if cached is not None:
      return cached
    # cold/expired: serialise callers of this tag so only one hits upstream.
    lock = self._locks.setdefault(tag, asyncio.Lock())
    async with lock:
      cached = self._fresh(tag)          # a racer may have filled it while we waited
      if cached is not None:
        return cached
      value = await self._fetch(tag)
      self._values[tag] = (value, self._clock())
      return value

  def invalidate(self, tag: Optional[str] = None) -> None:
    """Drop one tag (or all) so the next read refetches."""
    if tag is None:
      self._values.clear()
    else:
      self._values.pop(tag, None)

  def stats(self) -> Dict[str, float]:
    return {"cached_tags": len(self._values), "ttl": self._ttl}


async def _default_fetch(tag: str) -> float:
  # Imported lazily: app.pull pulls in the DB/helpers stack, which must not be a
  # hard import dependency of this small utility (avoids import cycles at boot).
  from app.pull import async_get_realtime
  from app.logger import LOGGER_GLOBAL
  return await async_get_realtime(tag, logger=LOGGER_GLOBAL)


# Shared by every realtime read served from the API process (REST endpoints and,
# later, the forecast WS `get_actual`). Keyed by tag, so all task types benefit.
realtime_cache = RealtimeCache(ttl=Config.realtime_ttl, fetch=_default_fetch)
