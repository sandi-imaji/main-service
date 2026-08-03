"""The app.core ⇄ PyCaret contract, tested against REAL PyCaret.

Every other test fakes PyCaret (`monkeypatch.setattr("app.core.*.mod", ...)`)
because real training takes seconds, not milliseconds. The consequence: those
tests only prove our code is consistent with *our assumptions* about PyCaret. If
PyCaret changes an output column name, an exception type, or where it saves
files, the whole suite stays green while production breaks.

This file closes that gap. Each test locks down one concrete assumption that
`app/core/*.py` actually relies on, and its comment names the calling line so it
is obvious what broke when the test goes red.

Deliberately skipped during day-to-day development and run before a release or
after bumping the PyCaret version:

    python -m pytest -m "not slow"          # daily, fast
    python -m pytest tests/integration -v   # pre-release (~25 seconds)

Verified against PyCaret 3.3.2 / scikit-learn 1.4.2.
"""
import os
import warnings

import numpy as np
import pandas as pd
import pytest

from app.config import Config, Verbose
from app.core.anomaly import Anomaly
from app.core.contracts import (
    AnomalyTrainRequest,
    ClusterAssignRequest,
    ClusteringTrainRequest,
    ForecastRequest,
    PredictRequest,
    SupervisedTrainRequest,
    TimeSeriesTrainRequest,
)
from app.core.supervised import Supervised
from app.core.time_series import TimeSeries
from app.core.unsupervised import Unsupervised

pytestmark = [pytest.mark.slow, pytest.mark.integration]

SEED = 0


@pytest.fixture(scope="session", autouse=True)
def quiet_pycaret():
  """PyCaret is chatty; its noise drowns out the actual failures."""
  with pytest.MonkeyPatch.context() as mp:
    mp.setattr(Config, "verbose", Verbose.SILENT)
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      yield


@pytest.fixture(scope="session")
def logger():
  from unittest.mock import Mock
  return Mock()


# --------------------------------------------------------------------------- #
# Data. Deliberately small with an obvious pattern: what is under test is the
# interface contract, not model quality. A clear pattern means a failure reads as
# "the contract changed", not "the model happened to be poor".
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def df_regression():
  rng = np.random.default_rng(SEED)
  n = 60
  df = pd.DataFrame({
      "dt": pd.date_range("2026-01-01", periods=n, freq="5min").astype(str),
      "x1": rng.normal(10, 2, n),
      "x2": rng.normal(5, 1, n),
  })
  df["y"] = 3 * df["x1"] + 2 * df["x2"] + rng.normal(0, 0.1, n)  # y = 3·x1 + 2·x2
  return df


@pytest.fixture(scope="session")
def df_cluster():
  """Two well-separated blobs — the grouping must not be ambiguous."""
  rng = np.random.default_rng(SEED)
  X = np.vstack([rng.normal(0, 0.5, (30, 2)), rng.normal(8, 0.5, (30, 2))])
  return pd.DataFrame({
      "dt": pd.date_range("2026-01-01", periods=60, freq="5min").astype(str),
      "x1": X[:, 0], "x2": X[:, 1],
  })


@pytest.fixture(scope="session")
def df_series():
  rng = np.random.default_rng(SEED)
  n = 60
  return pd.DataFrame({
      "dt": pd.date_range("2026-01-01", periods=n, freq="h").astype(str),
      "y": np.sin(np.arange(n) / 3.0) * 5 + 20 + rng.normal(0, 0.2, n),
  })


# --------------------------------------------------------------------------- #
# Trained artifacts. Once per session — retraining in every test would make this
# file too slow for anyone to actually run.
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def model_regression(df_regression, logger, tmp_path_factory):
  out = tmp_path_factory.mktemp("reg") / "top_model"
  req = SupervisedTrainRequest(df=df_regression, preprocessing={}, out_dir=out,
                               task="Regression", n_top=1, target="y")
  return Supervised.train_one(req, "lr", logger)


@pytest.fixture(scope="session")
def model_cluster(df_cluster, logger, tmp_path_factory):
  out = tmp_path_factory.mktemp("clu") / "top_model"
  req = ClusteringTrainRequest(df=df_cluster, preprocessing={}, out_dir=out,
                               task="Clustering", n_top=1, n_clusters=2)
  result = Unsupervised.compare_models(req, logger)
  reference = pd.read_csv(out.parent / "results" / "clusters.csv")
  return result, reference


@pytest.fixture(scope="session")
def model_anomaly(df_series, logger, tmp_path_factory):
  rng = np.random.default_rng(SEED)
  n = 80
  df = pd.DataFrame({
      "dt": pd.date_range("2026-01-01", periods=n, freq="5min").astype(str),
      "x1": rng.normal(10, 1, n), "x2": rng.normal(5, 1, n),
  })
  root = tmp_path_factory.mktemp("ano")
  req = AnomalyTrainRequest(df=df, preprocessing={}, out_dir=root / "top_model",
                            task="Anomaly", n_top=1, fraction=0.05, algorithm="iforest")
  return Anomaly.train_one(req, logger), root


@pytest.fixture(scope="session")
def model_series(df_series, logger, tmp_path_factory):
  out = tmp_path_factory.mktemp("ts") / "top_model"
  req = TimeSeriesTrainRequest(df=df_series, preprocessing={}, out_dir=out,
                               task="TimeSeries", n_top=1, fh=3)
  return TimeSeries.train_one(req, "naive", logger)


# --------------------------------------------------------------------------- #


class TestSaveConvention:
  """`TrainedModel.from_saved` sizes `f"{path}.pkl"` (contracts.py:70).

  Every other layer follows the same convention — `check_integrity_model` checks
  `Config.dir / f"{m.path}.pkl"`. If PyCaret stopped appending the `.pkl` suffix,
  models would appear missing system-wide.
  """

  def test_save_model_writes_a_pkl_file(self, model_regression):
    assert os.path.exists(f"{model_regression.path}.pkl")

  def test_artifact_size_is_readable(self, model_regression):
    assert model_regression.size > 0

  def test_path_is_stored_without_the_suffix(self, model_regression):
    """The path is stored bare; `.pkl` is appended by callers, not stored."""
    assert not model_regression.path.endswith(".pkl")


class TestSupervised:
  def test_train_one_uses_the_mean_row_from_pull(self, model_regression):
    """`mod.pull().loc["Mean"]` (supervised.py:153). If PyCaret renamed that
    summary row, training would raise KeyError."""
    assert {"MAE", "RMSE", "R2", "MAPE"} <= set(model_regression.evaluation)

  def test_prediction_column_is_named_prediction_label(self, model_regression, logger):
    """`res.to_dict(orient="list")["prediction_label"][0]` (supervised.py:97)."""
    req = PredictRequest(features={"x1": [10.0], "x2": [5.0]},
                         models=[("lr", model_regression.path)], task="Regression")
    result = Supervised.predict(req, logger)
    assert "lr" in result.predictions

  def test_the_model_actually_learns(self, model_regression, logger):
    """Sanity guard: if the preprocessing pipeline silently mangles features,
    the contract still holds but the predictions are garbage."""
    req = PredictRequest(features={"x1": [10.0], "x2": [5.0]},
                         models=[("lr", model_regression.path)], task="Regression")
    result = Supervised.predict(req, logger)
    assert result.predictions["lr"] == pytest.approx(40.0, abs=1.0)  # 3·10 + 2·5

  def test_input_features_are_returned_verbatim(self, model_regression, logger):
    req = PredictRequest(features={"x1": [10.0], "x2": [5.0]},
                         models=[("lr", model_regression.path)], task="Regression")
    assert Supervised.predict(req, logger).features == {"x1": 10.0, "x2": 5.0}

  def test_setup_accepts_ignore_features_dt(self, model_regression):
    """The core always passes `ignore_features=["dt"]` (supervised.py:110).
    If that argument were rejected, EVERY supervised training would fail."""
    assert model_regression.algorithm == "lr"

  # Trains EVERY regression algorithm: ~10s on a dev machine, and the global 30s
  # limit in pytest.ini is too tight for a slower CI box.
  @pytest.mark.timeout(300)
  def test_compare_models_pull_is_indexed_by_algorithm_name(self, df_regression, logger,
                                                        tmp_path_factory):
    """`metrics = mod.pull().to_dict("index")`, then `keys[i]` is used as the
    ALGORITHM NAME and the filename both (supervised.py:128-138). If that index
    became numeric, model names in the database would turn numeric too."""
    out = tmp_path_factory.mktemp("cmp") / "top_model"
    req = SupervisedTrainRequest(df=df_regression, preprocessing={}, out_dir=out,
                                 task="Regression", n_top=2, target="y")
    result = Supervised.compare_models(req, logger)

    assert len(result) == 2
    for m in result:
      assert isinstance(m.algorithm, str) and not m.algorithm.isdigit()
      assert os.path.exists(f"{m.path}.pkl")
      # `evaluation.pop("Model", None)` (supervised.py:136) drops that column;
      # what remains must be metrics, not text.
      assert "Model" not in m.evaluation
      assert "R2" in m.evaluation


class TestClustering:
  def test_assign_model_yields_a_cluster_column(self, model_cluster):
    """`clusters_df[algo] = mod.assign_model(model)["Cluster"]` (unsupervised.py:200)."""
    _, reference = model_cluster
    result, _ = model_cluster
    assert result[0].algorithm in reference.columns

  def test_labels_use_the_cluster_n_format(self, model_cluster):
    """Penamaan cluster oleh user memetakan dari labels mentah ini; formatnya
    ikut muncul di `naming_clusters.json` dan di tabel unduhan."""
    result, reference = model_cluster
    labels = set(reference[result[0].algorithm])
    assert labels == {"Cluster 0", "Cluster 1"}

  def test_ranking_metrics_are_present(self, model_cluster):
    """`_score()` ranks by these three metrics (unsupervised.py:186-188). If they
    disappeared, every algorithm would fall back to the default score and the
    ordering would be arbitrary."""
    result, _ = model_cluster
    assert {"Silhouette", "Calinski-Harabasz", "Davies-Bouldin"} <= set(result[0].evaluation)

  def test_two_separated_blobs_are_recognised(self, model_cluster):
    result, reference = model_cluster
    assert reference[result[0].algorithm].nunique() == 2

  def test_predict_model_yields_a_cluster_column(self, model_cluster, logger):
    """`result["Cluster"].iloc[0]` (unsupervised.py:98)."""
    result, reference = model_cluster
    algo = result[0].algorithm
    req = ClusterAssignRequest(features={"x1": [0.1], "x2": [0.1]},
                               models=[(algo, result[0].path)], task="Clustering",
                               reference=reference)
    assert Unsupervised.assign(req, logger).clusters[algo].startswith("Cluster")

  def test_labels_are_stable_across_calls(self, model_cluster, logger):
    """The whole reason `assign()` exists: cluster numbering must hold over time.

    Its predecessor (`predict()`, transductive) refit on every tick, so the
    grouping came out identical (ARI 1.000) but 0% of labels matched — realtime
    chart colours became incomparable and cluster names stuck to the wrong group.
    The cache is cleared each iteration deliberately, so what is proven is the
    stability of the ARTIFACT on disk, not a coincidence of the same in-memory
    object being reused.
    """
    result, reference = model_cluster
    algo = result[0].algorithm
    req = ClusterAssignRequest(features={"x1": [0.1], "x2": [0.1]},
                               models=[(algo, result[0].path)], task="Clustering",
                               reference=reference)
    labels = []
    for _ in range(3):
      Unsupervised.clear_cache()
      labels.append(Unsupervised.assign(req, logger).clusters[algo])
    assert len(set(labels)) == 1, f"labels bergeser antar pemanggilan: {labels}"

  def test_points_in_different_blobs_get_different_labels(self, model_cluster, logger):
    result, reference = model_cluster
    algo, path = result[0].algorithm, result[0].path
    def _label(x):
      req = ClusterAssignRequest(features={"x1": [x], "x2": [x]}, models=[(algo, path)],
                                 task="Clustering", reference=reference)
      return Unsupervised.assign(req, logger).clusters[algo]
    assert _label(0.1) != _label(8.0)

  def test_distance_to_centroid_is_computed(self, model_cluster, logger):
    """`_cluster_stats` uses `pipeline[:-1].transform` (unsupervised.py:64) —
    assuming the saved artifact is an sklearn Pipeline with the model in last
    position. If PyCaret changed that shape, the distance ratio would silently
    disappear."""
    result, reference = model_cluster
    algo = result[0].algorithm
    req = ClusterAssignRequest(features={"x1": [0.1], "x2": [0.1]},
                               models=[(algo, result[0].path)], task="Clustering",
                               reference=reference)
    d = Unsupervised.assign(req, logger).distances[algo]
    assert d["distance"] >= 0 and d["radius"] > 0
    assert d["ratio"] == pytest.approx(d["distance"] / d["radius"], abs=0.01)

  def test_spectral_raises_type_error_and_nothing_else(self, df_cluster, tmp_path):
    """`assign()` catches a SPECIFIC TypeError for models without `predict`
    (unsupervised.py:99). If PyCaret swapped it for a different exception, the
    nearest-centroid fallback would never run and realtime clustering with `sc`
    would die outright — and `sc` is listed in algorithm_list.json.
    """
    from pycaret import clustering as mod
    mod.setup(df_cluster, verbose=False, ignore_features=["dt"])
    model = mod.create_model("sc", num_clusters=2, verbose=False)
    path = tmp_path / "sc"
    mod.save_model(model, str(path))

    with pytest.raises(TypeError):
      mod.predict_model(mod.load_model(str(path)),
                        data=pd.DataFrame({"x1": [0.0], "x2": [0.0]}))

  def test_nearest_centroid_fallback_is_used_for_spectral(self, df_cluster, logger,
                                                             tmp_path):
    """The other side of the test above: once the TypeError appears, `assign()`
    must still return a label — not propagate the error to the WebSocket."""
    from pycaret import clustering as mod
    mod.setup(df_cluster, verbose=False, ignore_features=["dt"])
    model = mod.create_model("sc", num_clusters=2, verbose=False)
    path = tmp_path / "sc"
    mod.save_model(model, str(path))

    reference = df_cluster.copy()
    reference["sc"] = mod.assign_model(model)["Cluster"]

    Unsupervised.clear_cache()
    req = ClusterAssignRequest(features={"x1": [0.1], "x2": [0.1]},
                               models=[("sc", str(path))], task="Clustering",
                               reference=reference)
    assert Unsupervised.assign(req, logger).clusters["sc"].startswith("Cluster")


class TestAnomaly:
  def test_labelled_csv_has_the_anomaly_columns(self, model_anomaly):
    """`mod.assign_model(model).to_csv(...)` (anomaly.py:69)."""
    _, root = model_anomaly
    columns = pd.read_csv(root / "anomaly.csv").columns
    assert {"Anomaly", "Anomaly_Score"} <= set(columns)

  def test_model_is_saved_at_the_storage_root_not_top_model(self, model_anomaly):
    """Anomaly keeps ONE model at `<storage>/anomaly`, not per-algorithm under
    `top_model/` (anomaly.py:71). The streamer looks exactly there — a mismatched
    path produces "Model not found | model_path : anomaly.pkl"."""
    result, root = model_anomaly
    assert result.path == str(root / "anomaly")
    assert os.path.exists(root / "anomaly.pkl")

  def test_predict_yields_anomaly_and_score(self, model_anomaly, logger):
    """`predict_model(...)[["Anomaly", "Anomaly_Score"]]` (anomaly.py:44)."""
    result, _ = model_anomaly
    req = PredictRequest(features={"x1": [10.0], "x2": [5.0]},
                         models=[("iforest", result.path)], task="Anomaly")
    r = Anomaly.predict(req, logger)
    assert isinstance(r.is_anomaly, bool)
    assert isinstance(r.anomaly_score, float)

  def test_extreme_point_is_flagged_as_anomaly(self, model_anomaly, logger):
    """Sanity: if this fails while the others pass, the contract is intact but
    the model is detecting nothing."""
    result, _ = model_anomaly
    def _anomali(values):
      req = PredictRequest(features={"x1": [values], "x2": [values]},
                           models=[("iforest", result.path)], task="Anomaly")
      return Anomaly.predict(req, logger).is_anomaly
    assert _anomali(500.0) is True

  def test_detection_summary_is_stored(self, model_anomaly):
    """Anomaly has no comparison metric, but that does not mean no numbers:
    `_summarise` condenses the `assign_model()` output into something judgeable.
    Its evaluation used to be an empty `{}`, leaving no way to compare algorithms
    or assess the chosen `fraction`."""
    result, _ = model_anomaly
    assert {"AnomalyCount", "AnomalyRate", "FractionRequested",
            "ScoreMean", "ScoreP95", "Threshold"} <= set(result.evaluation)

  def test_rate_follows_the_requested_fraction(self, model_anomaly):
    """The point of it: comparing the two shows whether the model actually
    honoured what was asked of it."""
    result, _ = model_anomaly
    assert result.evaluation["AnomalyRate"] == pytest.approx(
        result.evaluation["FractionRequested"], abs=0.02)

  def test_metrics_are_serialisable(self, model_anomaly):
    """These land in a JSON column and get sent by FastAPI — a NaN means a 500."""
    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import JSONResponse
    result, _ = model_anomaly
    JSONResponse(content=jsonable_encoder(result.evaluation))    # must not raise


class TestTimeSeries:
  def test_training_uses_the_mean_row_from_pull(self, model_series):
    """`mod.pull().loc["Mean"]` (time_series.py:110)."""
    assert {"MASE", "MAE", "RMSE", "MAPE"} <= set(model_series.evaluation)

  def test_forecast_yields_a_y_pred_column(self, model_series, logger):
    """`mod.predict_model(model, req.fh)["y_pred"]` (time_series.py:54). PyCaret
    time-series uses a DIFFERENT output column name than supervised
    (`prediction_label`) — that difference is what is locked down here."""
    result = TimeSeries.forecast(
        ForecastRequest(models=[("naive", model_series.path)], fh=3, task="TimeSeries"),
        logger)
    assert list(result.forecast) == ["naive"]

  def test_forecast_length_matches_the_horizon(self, model_series, logger):
    for fh in (1, 5):
      result = TimeSeries.forecast(
          ForecastRequest(models=[("naive", model_series.path)], fh=fh, task="TimeSeries"),
          logger)
      assert len(result.forecast["naive"]) == fh

  def test_forecast_values_are_plausible(self, model_series, df_series, logger):
    """The series oscillates around 20; a forecast outside the data range means
    the preprocessing pipeline or `finalize_model` broke something."""
    result = TimeSeries.forecast(
        ForecastRequest(models=[("naive", model_series.path)], fh=3, task="TimeSeries"),
        logger)
    values = result.forecast["naive"]
    assert all(df_series["y"].min() - 5 <= v <= df_series["y"].max() + 5 for v in values)
