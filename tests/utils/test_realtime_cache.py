"""app.utils.realtime_cache.RealtimeCache.

The cache's collaborators are injected (fetch + clock), so its logic is verified
deterministically with a counting fetcher and a fake clock — no network, no real
sleeping. One reachability-gated test exercises the real default fetch end to end.
"""
import asyncio

import pytest

from app.utils.realtime_cache import RealtimeCache, realtime_cache


class CountingFetch:
  """Awaitable fetcher that records how many upstream calls happened."""
  def __init__(self, value=1.0):
    self.calls = 0
    self.value = value

  async def __call__(self, tag: str) -> float:
    self.calls += 1
    return self.value


class FakeClock:
  def __init__(self):
    self.t = 0.0

  def __call__(self) -> float:
    return self.t

  def advance(self, seconds: float):
    self.t += seconds


class TestFanOut:
  async def test_serves_within_ttl_from_memory(self):
    fetch, clock = CountingFetch(value=42.0), FakeClock()
    cache = RealtimeCache(ttl=5.0, fetch=fetch, clock=clock)

    assert await cache.get("TAG") == 42.0
    clock.advance(4.0)                    # still inside TTL
    assert await cache.get("TAG") == 42.0
    assert fetch.calls == 1               # second read served from cache

  async def test_refetches_after_ttl_expiry(self):
    fetch, clock = CountingFetch(), FakeClock()
    cache = RealtimeCache(ttl=5.0, fetch=fetch, clock=clock)

    await cache.get("TAG")
    clock.advance(5.0)                     # exactly at TTL boundary -> stale
    await cache.get("TAG")
    assert fetch.calls == 2

  async def test_distinct_tags_are_cached_independently(self):
    fetch, clock = CountingFetch(), FakeClock()
    cache = RealtimeCache(ttl=5.0, fetch=fetch, clock=clock)

    await cache.get("A")
    await cache.get("B")
    await cache.get("A")                   # A already cached
    assert fetch.calls == 2                # one per distinct tag


class TestSingleFlight:
  async def test_concurrent_cold_reads_fetch_once(self):
    started = asyncio.Event()
    release = asyncio.Event()

    class SlowFetch:
      def __init__(self):
        self.calls = 0

      async def __call__(self, tag):
        self.calls += 1
        started.set()
        await release.wait()               # hold the fetch open
        return 7.0

    fetch = SlowFetch()
    cache = RealtimeCache(ttl=100.0, fetch=fetch, clock=FakeClock())

    tasks = [asyncio.create_task(cache.get("TAG")) for _ in range(5)]
    await started.wait()                   # first fetch is in-flight
    release.set()
    results = await asyncio.gather(*tasks)

    assert results == [7.0] * 5
    assert fetch.calls == 1                # 4 racers awaited the single fetch


class TestInvalidate:
  async def test_invalidate_one_forces_refetch(self):
    fetch, clock = CountingFetch(), FakeClock()
    cache = RealtimeCache(ttl=100.0, fetch=fetch, clock=clock)

    await cache.get("TAG")
    cache.invalidate("TAG")
    await cache.get("TAG")
    assert fetch.calls == 2

  async def test_invalidate_all_clears(self):
    fetch, clock = CountingFetch(), FakeClock()
    cache = RealtimeCache(ttl=100.0, fetch=fetch, clock=clock)
    await cache.get("A")
    await cache.get("B")
    cache.invalidate()
    assert cache.stats()["cached_tags"] == 0


# --- real end-to-end (live SL API) -----------------------------------------

@pytest.mark.integration
class TestRealFetch:
  async def test_real_tag_is_fetched_then_cached(self):
    from app.pull import query_tagnames
    from app.logger import Logger
    try:
      tags = query_tagnames("CRAH-2DH2.1-*SUPPLY", logger=Logger("cache-it"))
    except Exception:
      pytest.skip("SL API unreachable")
    if not tags:
      pytest.skip("no tags for probe query")

    tag = tags[0]["tagname"]
    realtime_cache.invalidate(tag)
    v1 = await realtime_cache.get(tag)
    v2 = await realtime_cache.get(tag)     # served from cache within TTL
    assert isinstance(v1, float)
    assert v1 == v2
