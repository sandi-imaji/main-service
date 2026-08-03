"""app.core.contracts.clean_metrics — tidying pycaret's metric table.

Every core stores its metrics through `TrainedModel.from_saved`, and the result
lands in a JSON column that FastAPI later serialises. A single NaN in there is
enough to make an endpoint answer HTTP 500, so the safety net lives here.
"""

import pytest
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from app.core.contracts import TrainedModel, clean_metrics


def _serialisable(payload) -> str:
  """Serialise exactly the way the FastAPI response path does (allow_nan=False)."""
  return JSONResponse(content=jsonable_encoder(payload)).body.decode()


class TestNonFiniteValues:
  def test_nan_becomes_none(self):
    assert clean_metrics({"R2": float("nan")}) == {"R2": None}

  @pytest.mark.parametrize("value", [float("inf"), float("-inf")])
  def test_inf_becomes_none(self, value):
    assert clean_metrics({"MAPE": value}) == {"MAPE": None}

  def test_the_key_survives_rather_than_being_dropped(self):
    """An undefined metric should still be visible as "not available" rather
    than vanishing without trace — the user deserves to know it was not
    computed."""
    assert "R2" in clean_metrics({"R2": float("nan"), "MAE": 0.5})

  def test_ordinary_numbers_are_untouched(self):
    metrics = {"MAE": 0.5, "R2": 0.99, "TT (Sec)": 0.21}
    assert clean_metrics(metrics) == metrics

  def test_non_numeric_values_pass_through(self):
    assert clean_metrics({"Note": "text"}) == {"Note": "text"}

  def test_empty_input_is_safe(self):
    assert clean_metrics({}) == {} and clean_metrics(None) == {}


class TestBookkeepingColumns:
  def test_cutoff_is_dropped(self):
    """`cutoff` is pycaret time-series' fold split point, not a metric. On the
    "Mean" row its value is always NaN."""
    assert "cutoff" not in clean_metrics({"cutoff": float("nan"), "MASE": 0.9})

  def test_model_column_is_dropped(self):
    """The "Model" column holds the algorithm's long name, not a number."""
    assert "Model" not in clean_metrics({"Model": "Linear Regression", "R2": 0.9})

  def test_other_metrics_stay_intact(self):
    result = clean_metrics({"cutoff": float("nan"), "MASE": 0.9, "MAPE": 0.1})
    assert result == {"MASE": 0.9, "MAPE": 0.1}


class TestNumpyTypes:
  def test_numpy_scalar_becomes_native_python(self):
    np = pytest.importorskip("numpy")
    result = clean_metrics({"Silhouette": np.float64(0.43)})
    assert isinstance(result["Silhouette"], float)

  def test_numpy_nan_is_caught_too(self):
    np = pytest.importorskip("numpy")
    assert clean_metrics({"R2": np.float64("nan")}) == {"R2": None}


class TestFastAPISerialisable:
  def test_raw_nan_really_does_break_the_response(self):
    """Confirms the hazard is real rather than over-caution."""
    with pytest.raises(ValueError):
      _serialisable({"cutoff": float("nan")})

  def test_passes_once_cleaned(self):
    body = _serialisable(clean_metrics({"cutoff": float("nan"), "MASE": 0.97}))
    assert body == '{"MASE":0.97}'


class TestWiredIntoFromSaved:
  def test_from_saved_cleans_its_metrics(self, tmp_path):
    """The net is in the shared funnel, so all four cores are covered without
    anyone having to remember to call it."""
    (tmp_path / "naive.pkl").write_bytes(b"x" * 10)
    tm = TrainedModel.from_saved("naive", tmp_path / "naive",
                                 {"cutoff": float("nan"), "MASE": 0.97})
    assert tm.evaluation == {"MASE": 0.97}
    assert tm.size == 10
    _serialisable(tm.evaluation)          # must not raise
