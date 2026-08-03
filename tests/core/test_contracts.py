"""Contracts are plain, frozen value objects — no Dataset/DB knowledge."""
import dataclasses

import pandas as pd
import pytest

from app.core.contracts import (
    AnomalyResult,
    ClusterRequest,
    ClusterResult,
    ForecastResult,
    PredictResult,
    SupervisedTrainRequest,
    TrainedModel,
)


class TestTrainedModel:
  def test_from_saved_reads_pkl_size(self, tmp_path):
    # pycaret saves `<path>.pkl`; from_saved must size that file, not `path`.
    path = tmp_path / "lr"
    (tmp_path / "lr.pkl").write_bytes(b"1234567890")

    tm = TrainedModel.from_saved("lr", path, {"MAE": 0.1})

    assert tm.algorithm == "lr"
    assert tm.path == str(path)
    assert tm.evaluation == {"MAE": 0.1}
    assert tm.size == 10

  def test_missing_pkl_raises(self, tmp_path):
    with pytest.raises(OSError):
      TrainedModel.from_saved("lr", tmp_path / "nope", {})


class TestFrozen:
  def test_train_request_is_frozen(self):
    req = SupervisedTrainRequest(
        df=pd.DataFrame({"x": [1]}), preprocessing={}, out_dir="/tmp",
        task="Regression", target="y")
    with pytest.raises(dataclasses.FrozenInstanceError):
      req.target = "z"

  def test_cluster_request_carries_combined_frame(self):
    df = pd.DataFrame({"x1": [1, 2], "x2": [3, 4]})
    req = ClusterRequest(df=df, algorithms=["kmeans"], n_clusters=2,
                         preprocessing={}, task="Clustering")
    assert list(req.algorithms) == ["kmeans"]
    assert req.n_clusters == 2
    assert req.df.shape == (2, 2)


class TestResults:
  def test_predict_result_shape(self):
    r = PredictResult(features={"x1": 1.0}, predictions={"lr": 2.0})
    assert r.predictions["lr"] == 2.0

  def test_anomaly_result_shape(self):
    r = AnomalyResult(features={"x1": 1.0}, is_anomaly=True, anomaly_score=0.9)
    assert r.is_anomaly is True and r.anomaly_score == 0.9

  def test_forecast_and_cluster_results(self):
    assert ForecastResult(forecast={"ets": [1, 2]}).forecast["ets"] == [1, 2]
    cr = ClusterResult(clusters={"kmeans": 0}, assignments={"kmeans": [0, 1, 0]})
    assert cr.clusters["kmeans"] == 0
    assert cr.assignments["kmeans"] == [0, 1, 0]
