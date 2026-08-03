"""app.pull — REAL integration tests against the live SL API.

No network mocking: these actually call the SL tag-search / get-history endpoints
using the credentials in Config. A real tagname is discovered at runtime via
`query_tagnames`, then fed to the other calls (so nothing is hardcoded to a tag
that might disappear).

Requires a reachable SL API. If it is unreachable the whole module is SKIPPED
(not failed), so the rest of the suite stays green offline. Run just these with:
    pytest -m integration
Values are live, so assertions check shape/type — never exact numbers.

The pure helpers (PullDate, _retry_budget) need no network and live here too.
"""
import asyncio
import datetime

import pandas as pd
import pytest

from app import pull
from app.helpers import DTEncoder
from app.logger import Logger

pytestmark = pytest.mark.integration

SEARCH_QUERY = "CRAH-2DH2.1-*SUPPLY"       # from the original scratch script
logger = Logger("test-pull")


def _api_reachable() -> bool:
  try:
    return bool(pull.query_tagnames(SEARCH_QUERY, logger=logger))
  except Exception:
    return False


if not _api_reachable():
  pytest.skip("SL API unreachable / no tags for probe query", allow_module_level=True)


@pytest.fixture(scope="module")
def tags():
  """Real tags discovered from the live search API: [{tagname,row_id,currvalue}]."""
  found = pull.query_tagnames(SEARCH_QUERY, logger=logger)
  assert found, "probe query returned no tags"
  return found


@pytest.fixture(scope="module")
def one_tag(tags):
  return tags[0]


# --- pure helpers (no network) ----------------------------------------------

class TestRetryBudget:
  def test_below_one_is_forced_to_one(self):
    assert pull._retry_budget(0) == 1
    assert pull._retry_budget(-3) == 1

  def test_valid_is_unchanged(self):
    assert pull._retry_budget(3) == 3


class TestPullDate:
  def test_is_today(self):
    assert pull.PullDate(current_date=DTEncoder.now_str(), time_start="00:00:00").is_today() is True
    assert pull.PullDate(current_date="19000101", time_start="00:00:00").is_today() is False

  def test_norm_time_pads(self):
    assert pull.PullDate.norm_time("9:5", "00:00:00") == "09:05:00"
    assert pull.PullDate.norm_time(None, "23:59:00") == "23:59:00"

  def test_from_str_with_times_is_normalised(self):
    pd_ = pull.PullDate.from_str("20200101", time_start="9:5", time_end="17:0")
    assert (pd_.time_start, pd_.time_end) == ("09:05:00", "17:00:00")

  def test_time_accessors_return_time_objects(self):
    pd_ = pull.PullDate(current_date="20200101", time_start="09:05:00", time_end="17:30:00")
    assert pd_.time_start_t() == datetime.time(9, 5, 0)
    assert pd_.time_end_t() == datetime.time(17, 30, 0)


# --- query_tagnames (the requested focus) — REAL ----------------------------

class TestQueryTagnames:
  def test_returns_real_tags(self, tags):
    assert len(tags) > 0
    for t in tags:
      assert set(t) >= {"tagname", "row_id", "currvalue"}
      assert isinstance(t["tagname"], str) and t["tagname"]
      assert isinstance(t["currvalue"], float)

  def test_unknown_query_returns_empty(self):
    assert pull.query_tagnames("__no_such_tag_zzz_12345__", logger=logger) == []


# --- get_realtime — REAL ----------------------------------------------------

class TestGetRealtime:
  def test_returns_float_for_real_tag(self, one_tag):
    value = pull.get_realtime(one_tag["tagname"], logger=logger)
    assert isinstance(value, float)


# --- get_history — REAL -----------------------------------------------------

class TestGetHistory:
  def test_today_returns_dataframe_shape(self, one_tag):
    df = pull.get_history(
        one_tag["tagname"], DTEncoder.now_str(),
        to_dataframe=True, row_id=one_tag["row_id"], logger=logger,
    )
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["dt", one_tag["tagname"]]
    if not df.empty:
      assert pd.api.types.is_datetime64_any_dtype(df["dt"])


# --- pull_realtime — REAL ---------------------------------------------------

class TestPullRealtime:
  def test_multiple_tags(self, tags):
    names = [t["tagname"] for t in tags[:2]]
    values = pull.pull_realtime(names, logger=logger)
    assert len(values) == len(names)
    for v in values:
      assert v is None or isinstance(v, float)


# --- pull_history_nday — REAL -----------------------------------------------

class TestPullHistoryNday:
  def test_recent_days_returns_frame(self, one_tag):
    # pass tags mapping so the ~7s get_row_id lookup is skipped
    df = pull.pull_history_nday(
        1, [one_tag["tagname"]], logger=logger,
        tags={one_tag["tagname"]: one_tag["row_id"]},
    )
    assert isinstance(df, pd.DataFrame)
    assert "dt" in df.columns and one_tag["tagname"] in df.columns


# --- async variants — REAL --------------------------------------------------

class TestAsyncGetRealtime:
  async def test_returns_float(self, one_tag):
    value = await pull.async_get_realtime(one_tag["tagname"], logger=logger)
    assert isinstance(value, float)


class TestAsyncGetHistory:
  async def test_today_returns_dataframe(self, one_tag):
    # async path resolves row_id from the tagname itself (real get_row_id call)
    df = await pull.async_get_history(
        one_tag["tagname"], DTEncoder.now_str(), to_dataframe=True, logger=logger,
    )
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["dt", one_tag["tagname"]]


class TestAsyncPullRealtime:
  async def test_returns_tag_value_map(self, one_tag):
    out = await pull.async_pull_realtime([one_tag["tagname"]])
    assert one_tag["tagname"] in out
    assert isinstance(out[one_tag["tagname"]], list)
    assert isinstance(out[one_tag["tagname"]][0], float)
