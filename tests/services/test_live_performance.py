"""app.services.live_performance — real error from predictions vs actuals.

Answers what `ModelML.evaluation` cannot: is the model STILL accurate. InfluxDB
is faked here; what is tested is the arithmetic and the filtering decisions.
"""
from types import SimpleNamespace

import pytest

from app.database.schemas import TaskType
from app.exceptions import ValidationException
from app.services import live_performance as lp


def _dataset(task_type=TaskType.TimeSeries, models=()):
  return SimpleNamespace(name="TimeSeries-test", task_type=task_type, models=list(models))


def _model(algorithm, mape):
  return SimpleNamespace(algorithm=algorithm, evaluation={"MAPE": mape})


def _record(actual, results):
  rec = {"timestamp": "2026-08-01T00:00:00+00:00", "results": results}
  if actual is not None:
    rec["actual"] = actual
  return rec


def _patch_influx(monkeypatch, records):
  monkeypatch.setattr(lp, "get_influx_storage",
                      lambda: SimpleNamespace(query_inference=lambda **k: records))


class TestPairing:
  def test_records_without_an_actual_are_skipped(self):
    """A prediction for a time that has not arrived — not a failure, so it must
    neither be counted nor treated as error."""
    pairs = lp._pairs([_record(None, {"naive": 10.0}), _record(9.0, {"naive": 10.0})])
    assert pairs == {"naive": [(10.0, 9.0)]}

  def test_multiple_models_are_kept_apart(self):
    pairs = lp._pairs([_record(9.0, {"naive": 10.0, "ets": 8.0})])
    assert set(pairs) == {"naive", "ets"}

  def test_non_finite_prediction_is_dropped(self):
    assert lp._pairs([_record(9.0, {"naive": float("nan")})]) == {}

  def test_non_finite_actual_is_dropped(self):
    assert lp._pairs([_record(float("inf"), {"naive": 10.0})]) == {}

  def test_missing_results_does_not_blow_up(self):
    assert lp._pairs([{"timestamp": "x", "actual": 9.0}]) == {}


class TestArithmetic:
  def test_perfect_predictions_give_zero(self):
    result = lp._metrics([(10.0, 10.0), (20.0, 20.0)])
    assert result["MAE"] == 0 and result["RMSE"] == 0 and result["MAPE"] == 0

  def test_mae_and_rmse(self):
    # errors: +2 and -4  → MAE 3, RMSE sqrt((4+16)/2) = sqrt(10)
    result = lp._metrics([(12.0, 10.0), (16.0, 20.0)])
    assert result["MAE"] == pytest.approx(3.0)
    assert result["RMSE"] == pytest.approx(10 ** 0.5)

  def test_mape_is_a_ratio_not_a_percentage(self):
    """Consistent with pycaret's training metrics, which are ratios too — a unit
    mismatch would make the drift comparison wrong by a factor of 100."""
    assert lp._metrics([(11.0, 10.0)])["MAPE"] == pytest.approx(0.1)

  def test_bias_separates_one_sided_error_from_random_error(self):
    """Both share a MAE; only the bias reveals that the first is correctable."""
    one_sided = lp._metrics([(15.0, 10.0), (25.0, 20.0)])
    random_err = lp._metrics([(15.0, 10.0), (15.0, 20.0)])
    assert one_sided["Bias"] == pytest.approx(5.0)
    assert random_err["Bias"] == pytest.approx(0.0)
    assert one_sided["MAE"] == random_err["MAE"]

  def test_zero_actuals_are_excluded_from_mape(self):
    """Dividing by zero explodes the MAPE and drowns the whole summary."""
    result = lp._metrics([(1.0, 0.0), (11.0, 10.0)])
    assert result["n"] == 2 and result["n_mape"] == 1
    assert result["MAPE"] == pytest.approx(0.1)

  def test_all_actuals_zero_gives_no_mape_rather_than_zero(self):
    result = lp._metrics([(1.0, 0.0)])
    assert result["MAPE"] is None and result["MAE"] == 1.0


class TestDrift:
  def test_multiplied_error_is_flagged_degraded(self):
    assert lp._drift(0.20, 0.05)["status"] == "degraded"

  def test_comparable_error_counts_as_normal(self):
    d = lp._drift(0.055, 0.05)
    assert d["status"] == "ok" and d["ratio"] == pytest.approx(1.1)

  def test_exactly_at_the_threshold_is_already_degraded(self):
    assert lp._drift(0.10, 0.05)["ratio"] == lp.DRIFT_WARN_RATIO
    assert lp._drift(0.10, 0.05)["status"] == "degraded"

  @pytest.mark.parametrize("live,trained", [(None, 0.05), (0.1, None), (0.1, 0)])
  def test_without_a_baseline_the_status_is_unknown(self, live, trained):
    """More honest than guessing "ok" when there is nothing to compare against."""
    assert lp._drift(live, trained)["status"] == "unknown"


class TestRunningMetrics:
  def test_non_timeseries_tasks_are_rejected(self, monkeypatch):
    """Regression never learns the true value of the points it predicted."""
    _patch_influx(monkeypatch, [])
    with pytest.raises(ValidationException):
      lp.running_metrics(_dataset(TaskType.Regression))

  def test_combines_running_error_with_training_error(self, monkeypatch):
    _patch_influx(monkeypatch, [_record(10.0, {"naive": 11.0}),
                                _record(20.0, {"naive": 22.0})])
    result = lp.running_metrics(_dataset(models=[_model("naive", 0.05)]))
    naive = result["models"]["naive"]
    assert naive["MAPE"] == pytest.approx(0.1)
    assert naive["trained_MAPE"] == 0.05
    assert naive["drift"]["status"] == "degraded"     # 0.1 / 0.05 = 2.0

  def test_reports_data_coverage(self, monkeypatch):
    """Low coverage means the numbers rest on few points — the reader deserves
    to know that before concluding the model is broken."""
    _patch_influx(monkeypatch, [_record(10.0, {"naive": 11.0}),
                                _record(None, {"naive": 12.0}),
                                _record(None, {"naive": 13.0})])
    result = lp.running_metrics(_dataset(models=[_model("naive", 0.05)]))
    assert result["total_records"] == 3
    assert result["records_with_actual"] == 1
    assert result["matched_points"] == 1

  def test_no_actuals_at_all_does_not_raise(self, monkeypatch):
    _patch_influx(monkeypatch, [_record(None, {"naive": 11.0})])
    result = lp.running_metrics(_dataset(models=[_model("naive", 0.05)]))
    assert result["models"] == {} and result["matched_points"] == 0

  def test_model_without_training_metric_is_still_reported(self, monkeypatch):
    """Models from compare_models sometimes carry no MAPE; their running error
    is still useful even without a baseline to compare it to."""
    _patch_influx(monkeypatch, [_record(10.0, {"naive": 11.0})])
    result = lp.running_metrics(_dataset(models=[]))
    assert result["models"]["naive"]["MAPE"] == pytest.approx(0.1)
    assert result["models"]["naive"]["drift"]["status"] == "unknown"

  def test_result_is_json_serialisable(self, monkeypatch):
    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import JSONResponse
    _patch_influx(monkeypatch, [_record(0.0, {"naive": 1.0})])   # MAPE becomes None
    result = lp.running_metrics(_dataset(models=[_model("naive", 0.05)]))
    JSONResponse(content=jsonable_encoder(result))               # must not raise
