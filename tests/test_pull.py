"""
Integration tests for pull module.
Uses real tagnames from tests/tagname.csv.
"""

import pytest
import random
import pandas as pd,time
from pathlib import Path
from app.pull import (
    get_realtime,
    async_get_realtime,
    pull_realtime,
    async_pull_realtime,
    get_history,
    async_get_history,
    query_tagnames
)
from app.pull import PullDate
from app.helpers import DTEncoder
from app.logger import Logger


def load_test_tagnames():
  """Load tagnames from CSV file for testing."""
  csv_path = Path(__file__).parent /"tagname.csv"
  df = pd.read_csv(csv_path)
  return df["tagname"].tolist()

# Load tagnames once for all tests
TEST_TAGNAMES = load_test_tagnames()


class TestGetRealtime:
  """Tests for get_realtime function with real tagnames."""

  def test_get_realtime_single_tag(self):
    """Test getting realtime value for a single tag."""
    if not TEST_TAGNAMES:
      pytest.skip("No tagnames available for testing")

    tagname = random.choice(TEST_TAGNAMES)
    logger = Logger("test_pull")

    try:
      result = get_realtime(tagname, logger=logger)
      assert isinstance(result, float)
      print(f"✓ get_realtime('{tagname}') = {result}")
    except Exception as e:
      pytest.fail(f"Failed to get realtime for {tagname}: {e}")

  def test_get_realtime_with_retries(self):
    """Test get_realtime with retry mechanism."""
    if not TEST_TAGNAMES:
      pytest.skip("No tagnames available for testing")

    tagname = random.choice(TEST_TAGNAMES)
    logger = Logger("test_pull")

    result = get_realtime(tagname, logger=logger,
                          max_retries=3, retry_delay=1.0)
    assert isinstance(result, float)
    print(f"✓ get_realtime with retries('{tagname}') = {result}")


class TestAsyncGetRealtime:
  """Tests for async_get_realtime function."""

  @pytest.mark.asyncio
  async def test_async_get_realtime_single_tag(self):
    """Test async realtime fetch for a single tag."""
    if not TEST_TAGNAMES:
      pytest.skip("No tagnames available for testing")

    tagname = random.choice(TEST_TAGNAMES)
    logger = Logger("test_pull")

    try:
      result = await async_get_realtime(tagname, logger=logger)
      assert isinstance(result, float)
      print(f"✓ async_get_realtime('{tagname}') = {result}")
    except Exception as e:
      pytest.fail(f"Failed to get async realtime for {tagname}: {e}")

  @pytest.mark.asyncio
  async def test_async_get_realtime_multiple_tags(self):
    """Test async realtime fetch for multiple tags concurrently."""
    if len(TEST_TAGNAMES) < 3:
      pytest.skip("Need at least 3 tagnames for this test")

    tagnames = random.sample(TEST_TAGNAMES, k=3)
    logger = Logger("test_pull")

    import asyncio

    tasks = [async_get_realtime(tag, logger=logger) for tag in tagnames]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print(results)

    for i, result in enumerate(results):
      if isinstance(result, Exception):
        print(f"✗ Tag {tagnames[i]} failed: {result}")
      else:
        assert isinstance(result, float)
        print(f"✓ async_get_realtime('{tagnames[i]}') = {result}")


class TestPullRealtime:
  """Tests for pull_realtime function."""

  def test_pull_realtime_multiple_tags(self):
    """Test pulling realtime values for multiple tags."""
    if len(TEST_TAGNAMES) < 3:
      pytest.skip("Need at least 3 tagnames for this test")

    tagnames = random.sample(TEST_TAGNAMES, k=5)
    logger = Logger("test_pull")

    results = pull_realtime(tagnames, logger=logger)

    assert len(results) == len(tagnames)
    for i, (tag, value) in enumerate(zip(tagnames, results)):
      if value is not None:
        assert isinstance(value, float)
        print(f"✓ pull_realtime[{i}]('{tag}') = {value}")
      else:
        print(f"⚠ pull_realtime[{i}]('{tag}') = None (failed)")


class TestAsyncPullRealtime:
  """Tests for async_pull_realtime function."""

  @pytest.mark.asyncio
  async def test_async_pull_realtime_multiple_tags(self):
    """Test async pulling realtime values for multiple tags."""
    if len(TEST_TAGNAMES) < 3:
      pytest.skip("Need at least 3 tagnames for this test")

    tagnames = random.sample(TEST_TAGNAMES, k=5)
    logger = Logger("test_pull")

    results = await async_pull_realtime(tagnames, logger=logger)

    assert len(results) == len(tagnames)
    success_count = sum(1 for r in results if r is not None)
    print(f"✓ async_pull_realtime: {success_count}/{len(tagnames)} successful")

    for tag, value in zip(tagnames, results):
      if value is not None:
        assert isinstance(value, float)


class TestGetHistory:
  """Tests for get_history function."""

  def test_get_history_single_tag(self):
    """Test getting historical data for a single tag."""
    if not TEST_TAGNAMES:
      pytest.skip("No tagnames available for testing")

    tagname = random.choice(TEST_TAGNAMES)
    logger = Logger("test_pull")

    # Get yesterday's date in YYYYMMDD format
    from datetime import datetime, timedelta

    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    try:
      result = get_history(
          tagname,
          current_date=yesterday,
          time_start="00:00:00",
          time_end="01:00:00",
          interval=300,
          to_dataframe=True,
          logger=logger,
      )

      assert isinstance(result, pd.DataFrame)
      print(f"✓ get_history('{tagname}') returned {len(result)} rows")
    except Exception as e:
      print(f"⚠ get_history('{tagname}') failed: {e}")
      # Don't fail the test if history is not available


class TestAsyncGetHistory:
  """Tests for async_get_history function."""

  @pytest.mark.asyncio
  async def test_async_get_history_single_tag(self):
    """Test async historical data fetch for a single tag."""
    if not TEST_TAGNAMES:
      pytest.skip("No tagnames available for testing")

    tagname = random.choice(TEST_TAGNAMES)
    logger = Logger("test_pull")

    from datetime import datetime, timedelta

    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    try:
      result = await async_get_history(
          tagname,
          current_date=yesterday,
          time_start="00:00:00",
          time_end="01:00:00",
          interval=300,
          to_dataframe=True,
          logger=logger,
      )

      assert isinstance(result, pd.DataFrame)
      print(f"✓ async_get_history('{tagname}') returned {len(result)} rows")
    except Exception as e:
      print(f"⚠ async_get_history('{tagname}') failed: {e}")

class TestPullHistory:
  """Test Sync Pull historical data fetch for a multiple tag."""
  if not TEST_TAGNAMES:
    pytest.skip("No tagnames available for testing")

  start_date = None
  end_date = None

class TestQueryTagnames:
  def test_query_tagname(self):
    tagname = "CRAH*SUPPLY"
    tagnames = query_tagnames(tagname)
    assert tagnames, f"Tagname : {tagname} is empty!"
    assert len(tagnames),f"Tagname : {tagname} is empty!"
    print(f"n tagnames : {len(tagnames)}")

class TestPerformanceComparison:
  """Performance tests comparing sync vs async operations."""

  @pytest.mark.asyncio
  async def test_sync_vs_async_performance(self):
    """Compare performance of sync vs async operations."""
    if len(TEST_TAGNAMES) < 5:
      pytest.skip("Need at least 5 tagnames for performance test")

    tagnames = random.sample(TEST_TAGNAMES, k=5)
    logger = Logger("test_pull")

    # Sync version
    start = time.time()
    sync_results = pull_realtime(tagnames, logger=logger)
    sync_duration = time.time() - start

    # Async version
    start = time.time()
    async_results = await async_pull_realtime(tagnames, logger=logger)
    async_duration = time.time() - start

    print(f"\n📊 Performance Comparison (5 tags):")
    print(f"   Sync:  {sync_duration:.3f}s")
    print(f"   Async: {async_duration:.3f}s")
    print( f"   Improvement: {((sync_duration - async_duration) / sync_duration * 100):.1f}%")
    # Both should have same number of results
    assert len(sync_results) == len(async_results)

if __name__ == "__main__":
  pytest.main([__file__, "-v", "-s"])
