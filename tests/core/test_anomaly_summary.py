"""app.core.anomaly.Anomaly._summarise — the detection summary.

Tested apart from PyCaret with hand-built labelled frames, so the edge cases (no
anomalies, empty data) can be checked without training anything. Its contract
against real PyCaret is locked down in tests/integration.
"""
import pandas as pd
import pytest

from app.core.anomaly import Anomaly


def _labelled(anomaly, score):
  return pd.DataFrame({"x1": range(len(anomaly)), "Anomaly": anomaly,
                       "Anomaly_Score": score})


class TestCountAndRate:
  def test_counts_what_was_flagged(self):
    result = Anomaly._summarise(_labelled([0, 0, 1, 1], [0.1, 0.2, 0.9, 0.8]), 0.5)
    assert result["AnomalyCount"] == 2
    assert result["TotalRows"] == 4
    assert result["AnomalyRate"] == 0.5

  def test_the_requested_fraction_is_stored_too(self):
    """Without it the resulting rate cannot be compared against what was asked —
    and that comparison is exactly what tells you whether the model obeyed."""
    assert Anomaly._summarise(_labelled([0, 1], [0.1, 0.9]), 0.05)["FractionRequested"] == 0.05


class TestScoreSpread:
  def test_mean_and_maximum(self):
    result = Anomaly._summarise(_labelled([0, 0, 1], [0.0, 0.5, 1.0]), 0.33)
    assert result["ScoreMean"] == pytest.approx(0.5)
    assert result["ScoreMax"] == 1.0

  def test_p95_sits_between_the_mean_and_the_maximum(self):
    scores = [i / 100 for i in range(101)]
    result = Anomaly._summarise(_labelled([0] * 100 + [1], scores), 0.01)
    assert result["ScoreMean"] < result["ScoreP95"] <= result["ScoreMax"]


class TestThreshold:
  def test_threshold_is_the_lowest_flagged_score(self):
    result = Anomaly._summarise(_labelled([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9]), 0.5)
    assert result["Threshold"] == 0.7

  def test_with_no_anomalies_the_threshold_is_none_not_zero(self):
    """0.0 would read as a very low threshold — when the truth is that there is
    no threshold at all."""
    result = Anomaly._summarise(_labelled([0, 0], [0.1, 0.2]), 0.05)
    assert result["Threshold"] is None
    assert result["AnomalyCount"] == 0 and result["AnomalyRate"] == 0.0


class TestEdgeCases:
  def test_empty_frame_does_not_divide_by_zero(self):
    result = Anomaly._summarise(_labelled([], []), 0.05)
    assert result["TotalRows"] == 0
    assert result["AnomalyRate"] is None and result["ScoreMean"] is None

  def test_everything_flagged(self):
    result = Anomaly._summarise(_labelled([1, 1], [0.8, 0.9]), 1.0)
    assert result["AnomalyRate"] == 1.0 and result["Threshold"] == 0.8

  def test_boolean_labels_are_accepted_too(self):
    """PyCaret returns 0/1; another frame might use real booleans."""
    result = Anomaly._summarise(_labelled([False, True], [0.1, 0.9]), 0.5)
    assert result["AnomalyCount"] == 1

  def test_every_value_is_json_serialisable(self):
    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import JSONResponse
    result = Anomaly._summarise(_labelled([0, 0], [0.1, 0.2]), 0.05)
    JSONResponse(content=jsonable_encoder(result))      # None is fine, NaN is not
