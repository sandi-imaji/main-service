"""Forecast anchor freshness (app.services.timeseries).

A time-series model forecasts `fh` steps from the END of its training window
(`meta.current_dt`), so that span is fixed at fit time and drains into the past
as the clock advances. The refresh used to wait for the WHOLE horizon to elapse,
which left the displayed forecast half-history on average (measured: an anchor
20h stale, 30/30 points already past).

These lock the replacement rule — refresh once `Config.forecast_refresh_ratio`
of the horizon is consumed — plus the two traps around it: `is_expired` and
`finetune` must gate on the SAME ratio (otherwise finetune declines and the
caller re-fires every tick), and a zero horizon must not spin the refit loop.
"""
import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.config import Config
from app.helpers import DTEncoder
from app.services import timeseries as ts


def _dataset(minutes_stale: float, interval: int = 5, fh: int = 30, aware: bool = True):
  anchor = DTEncoder.now() - datetime.timedelta(minutes=minutes_stale)
  current_dt = anchor.isoformat() if aware else anchor.replace(tzinfo=None).isoformat()
  return SimpleNamespace(interval=interval, target=fh,
                         meta=SimpleNamespace(current_dt=current_dt))


@pytest.fixture
def logger():
  return Mock()


class TestElapsedHorizonRatio:
  def test_fresh_anchor_is_zero(self):
    assert ts.elapsed_horizon_ratio(_dataset(0)) == pytest.approx(0, abs=0.01)

  def test_half_horizon(self):
    # horizon = 5min * 30 = 150min → 75min elapsed is half
    assert ts.elapsed_horizon_ratio(_dataset(75)) == pytest.approx(0.5, abs=0.01)

  def test_beyond_horizon_exceeds_one(self):
    assert ts.elapsed_horizon_ratio(_dataset(300)) > 1.0

  def test_zero_horizon_is_zero_not_infinite(self):
    assert ts.elapsed_horizon_ratio(_dataset(500, interval=0)) == 0.0


class TestIsExpired:
  def test_fresh_anchor_not_expired(self, logger):
    assert ts.is_expired(_dataset(0), logger) is False

  def test_just_below_threshold_not_expired(self, logger, monkeypatch):
    monkeypatch.setattr(Config, "forecast_refresh_ratio", 0.5)
    assert ts.is_expired(_dataset(74), logger) is False

  def test_at_threshold_is_expired(self, logger, monkeypatch):
    monkeypatch.setattr(Config, "forecast_refresh_ratio", 0.5)
    assert ts.is_expired(_dataset(75), logger) is True

  def test_refreshes_before_whole_horizon_elapses(self, logger, monkeypatch):
    """The regression this fixes: at 60% of the horizon the old rule said
    'not expired' and let the anchor keep rotting."""
    monkeypatch.setattr(Config, "forecast_refresh_ratio", 0.5)
    ds = _dataset(90)                       # 60% of a 150min horizon
    assert ts.is_expired(ds, logger) is True

  def test_ratio_one_keeps_legacy_whole_horizon_behaviour(self, logger, monkeypatch):
    monkeypatch.setattr(Config, "forecast_refresh_ratio", 1.0)
    assert ts.is_expired(_dataset(120), logger) is False   # 80% → not yet
    assert ts.is_expired(_dataset(150), logger) is True    # full horizon

  def test_zero_horizon_does_not_spin_refits(self, logger):
    # interval=0 would make every check "expired" and refit forever.
    assert ts.is_expired(_dataset(500, interval=0), logger) is False
    logger.warning.assert_called()

  def test_naive_current_dt_does_not_raise(self, logger):
    # stored timestamps sometimes lack an offset; comparing against the aware
    # DTEncoder.now() would raise TypeError without normalisation.
    assert ts.is_expired(_dataset(200, aware=False), logger) is True


class TestGateLockstep:
  """is_expired and finetune's guard must agree, or finetune no-ops while the
  caller keeps seeing 'expired' → a finetune call every single tick."""

  @pytest.mark.parametrize("stale", [0, 40, 74, 75, 100, 200, 400])
  def test_expired_implies_finetune_would_run(self, logger, stale):
    ds = _dataset(stale)
    expired = ts.is_expired(ds, logger)
    # finetune's guard is: ratio < Config.forecast_refresh_ratio → return early
    finetune_runs = ts.elapsed_horizon_ratio(ds) >= Config.forecast_refresh_ratio
    assert expired == finetune_runs
