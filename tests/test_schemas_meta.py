"""app.database.schemas — default timestamps on metadata.

Small, but it guards a class of bug that is expensive to track down: `created_at`
is the only record of when a dataset/model was born, and the dashboard
(`/utils/stats`) uses it to build the recent-activity list.
"""
import time

from app.database.schemas import MetaDataset, MetaModelML
from app.helpers import DTEncoder


class TestCreatedAtDefault:
  """The old version wrote `default_factory=DTEncoder.now().isoformat`.

  Note the parentheses: `DTEncoder.now()` is evaluated ONCE at import time, and
  what gets stored as the factory is that object's `.isoformat` method. Every
  instance therefore received an identical timestamp — the moment the process
  started, not the moment of creation. What masked it was pure luck: `pull.py`
  and `timeseries/retrain.py` set `created_at` explicitly.
  """

  def test_meta_dataset_uses_the_creation_time(self):
    a = MetaDataset()
    time.sleep(0.01)
    b = MetaDataset()
    assert a.created_at != b.created_at

  def test_meta_model_uses_the_creation_time(self):
    a = MetaModelML()
    time.sleep(0.01)
    b = MetaModelML()
    assert a.created_at != b.created_at

  def test_it_is_now_not_import_time(self):
    from app.services.model import _as_datetime
    before = DTEncoder.now()
    meta = MetaDataset()
    after = DTEncoder.now()
    assert before <= _as_datetime(meta.created_at) <= after

  def test_it_is_an_iso_string_that_parses_back(self):
    """`_build_stats` parses this value back into a datetime."""
    from app.services.model import _as_datetime
    meta = MetaDataset()
    assert isinstance(meta.created_at, str)
    assert _as_datetime(meta.created_at) is not None
