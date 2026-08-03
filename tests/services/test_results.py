"""app.services.results — output DTOs built from pure core results."""
import datetime
from unittest.mock import Mock

from app.core.contracts import AnomalyResult, ForecastResult, PredictResult
from app.services.results import (
    AnomalyResultSchema,
    SupervisedResultSchema,
    TimeSeriesResultSchema,
)


class TestSupervisedResult:
  def test_from_result(self):
    core = PredictResult(features={"x1": 1.0}, predictions={"lr": 2.0})
    s = SupervisedResultSchema.from_result(core, "ds-1", actual=3.0)
    assert s.is_valid is True
    assert s.predictions == {"lr": 2.0}
    assert s.actual == 3.0
    assert s.dataset_name == "ds-1"

  def test_failed_flattens_features(self):
    s = SupervisedResultSchema.failed({"x1": [5.0]}, ["lr", "knn"], "ds-1")
    assert s.is_valid is False
    assert s.features == {"x1": 5.0}
    assert s.predictions == {"lr": 0.0, "knn": 0.0}

  def test_write_to_influx_passes_fields(self):
    s = SupervisedResultSchema.from_result(
        PredictResult(features={"x1": 1.0}, predictions={"lr": 2.0}), "ds-1", actual=3.0)
    influx = Mock()
    s.write_to_influx(influx, Mock())
    _, kwargs = influx.write_inference.call_args
    assert kwargs["dataset_name"] == "ds-1"
    assert kwargs["results"] == {"lr": 2.0}
    assert kwargs["actual"] == 3.0


class TestAnomalyResult:
  def test_from_result_and_invalid(self):
    core = AnomalyResult(features={"x1": 1.0}, is_anomaly=True, anomaly_score=0.9)
    s = AnomalyResultSchema.from_result(core, "ds-1")
    assert s.is_valid is True and s.is_anomaly is True

    bad = AnomalyResultSchema.invalid("ds-1")
    assert bad.is_valid is False and bad.is_anomaly is False

  def test_write_to_influx_shape(self):
    s = AnomalyResultSchema.from_result(
        AnomalyResult(features={"x1": 1.0}, is_anomaly=True, anomaly_score=0.9), "ds-1")
    influx = Mock()
    s.write_to_influx(influx, Mock())
    _, kwargs = influx.write_inference.call_args
    assert kwargs["results"] == {"is_anomaly": True, "anomaly_score": 0.9}
    assert kwargs["task_type"] == "Anomaly"


class TestTimeSeriesResult:
  def _timestamps(self):
    base = datetime.datetime(2026, 7, 1, tzinfo=datetime.timezone.utc)
    return [base + datetime.timedelta(minutes=5 * i) for i in range(3)]

  def test_from_forecast(self):
    ts = self._timestamps()
    core = ForecastResult(forecast={"ets": [1, 2, 3]})
    s = TimeSeriesResultSchema.from_forecast(core, ts, "ds-1")
    assert s.is_valid is True
    assert s.forecast == {"ets": [1, 2, 3]}
    assert len(s.timestamps) == 3

  def test_invalid_has_empty_forecast(self):
    s = TimeSeriesResultSchema.invalid("ds-1")
    assert s.is_valid is False and s.forecast == {} and s.timestamps == []

  def test_write_to_influx_writes_each_step(self):
    ts = self._timestamps()
    s = TimeSeriesResultSchema.from_forecast(
        ForecastResult(forecast={"ets": [1, 2, 3]}), ts, "ds-1")
    influx = Mock()
    s.write_to_influx(influx, Mock())
    assert influx.write_inference.call_count == 3
