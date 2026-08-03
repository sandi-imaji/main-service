"""Dataset.is_time_to_finetune() — the supervised auto-finetune gate.

current_dt is persisted as an ISO *string*; the gate used to do `str + timedelta`
and raise TypeError every worker tick (skipping inference too). These lock the
parse-before-arithmetic behaviour.
"""
import datetime

from app.database.DB import Dataset
from app.database.schemas import MetaDataset, PreprocessingSchema
from app.helpers import DTEncoder


def _ds(current_dt, interval_finetune=5):
  return Dataset(
      meta=MetaDataset(current_dt=current_dt),
      preprocessing=PreprocessingSchema(interval_finetune=interval_finetune),
  )


class TestIsTimeToFinetune:
  def test_string_current_dt_does_not_raise(self):
    # freshly set -> not due for `interval_finetune` days (and must NOT raise)
    assert _ds(DTEncoder.now().isoformat()).is_time_to_finetune() is False

  def test_string_current_dt_overdue(self):
    old = (DTEncoder.now() - datetime.timedelta(days=10)).isoformat()
    assert _ds(old, interval_finetune=5).is_time_to_finetune() is True

  def test_space_separated_iso_is_parsed(self):
    # some writers store "YYYY-MM-DD HH:MM:SS+00:00" (space, not 'T')
    old = str(DTEncoder.now() - datetime.timedelta(days=10))
    assert _ds(old, interval_finetune=5).is_time_to_finetune() is True

  def test_none_current_dt_not_due(self):
    assert _ds(None).is_time_to_finetune() is False

  def test_exactly_due_today(self):
    due = (DTEncoder.now() - datetime.timedelta(days=5)).isoformat()
    assert _ds(due, interval_finetune=5).is_time_to_finetune() is True
