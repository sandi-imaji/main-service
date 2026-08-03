"""Umur forecast: seberapa banyak horizon yang sudah dimakan jam dinding.

Modul paling dasar di paket ini — tidak bergantung pada sibling mana pun, jadi
forecast/actuals/retrain semuanya boleh mengimpornya tanpa risiko impor
melingkar.
"""
import datetime

from app.config import Config
from app.database.DB import Dataset
from app.helpers import DTEncoder


def _current_dt(dataset: Dataset) -> datetime.datetime:
  """The dataset's current timestamp: from meta if set, else the last row of
  the stored dataframe. `meta` is a MetaDataset object (attribute access).

  Always returns an aware datetime: stored timestamps occasionally lack an
  offset, and every comparison here is against the aware `DTEncoder.now()`, so
  a naive value would make datetime arithmetic raise TypeError.
  """
  current_dt = dataset.meta.current_dt
  if not current_dt:
    dt = dataset.open_dataframe().iloc[-1]["dt"].to_pydatetime()
  else:
    dt = datetime.datetime.fromisoformat(str(current_dt))
  if dt.tzinfo is None:
    dt = dt.replace(tzinfo=DTEncoder.UTC)
  return dt


def _horizon_minutes(dataset: Dataset) -> int:
  """Total span the forecast covers: interval * forecast horizon (fh)."""
  return int(dataset.interval) * int(dataset.target)


def elapsed_horizon_ratio(dataset: Dataset) -> float:
  """How much of the forecast horizon the wall clock has already consumed.

  0.0 right after a refit (the whole horizon is still in the future), 1.0 once
  the final forecast timestamp is reached, >1.0 when the anchor is stale beyond
  the horizon. Returns 0.0 when the horizon is not computable.
  """
  horizon = _horizon_minutes(dataset)
  if horizon <= 0:
    return 0.0
  elapsed = (DTEncoder.now() - _current_dt(dataset)).total_seconds() / 60.0
  return elapsed / horizon


def is_expired(dataset: Dataset, logger) -> bool:
  """True when the forecast should be refreshed (refit onto newer data).

  The model forecasts `fh` steps from the END of its training window
  (`meta.current_dt`), so the span it covers is fixed at fit time and drains
  into the past as the clock advances. Waiting for the *whole* horizon to
  elapse means that, on average, half of the displayed forecast is already
  history — and the live overlay has that many fewer future points to fill. So
  it is treated as expired once `Config.forecast_refresh_ratio` of the horizon
  has been consumed.

  `finetune` gates on the same ratio: if the two disagreed, this could report
  "expired" while finetune declined to run, re-firing every tick.
  """
  horizon = _horizon_minutes(dataset)
  if horizon <= 0:
    # interval/fh 0 would make every check "expired" and spin the refit loop.
    logger.warning(f"Invalid horizon (interval={dataset.interval}, fh={dataset.target}) — refresh skipped")
    return False
  ratio = elapsed_horizon_ratio(dataset)
  logger.info(f"Now Timestamp : {DTEncoder.now()}")
  logger.info(f"Anchor (current_dt) : {_current_dt(dataset)} | horizon terpakai: {ratio:.0%} "
              f"(refresh di {Config.forecast_refresh_ratio:.0%})")
  return ratio >= Config.forecast_refresh_ratio
