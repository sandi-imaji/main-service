"""Pencilan, kesepakatan antar algoritma, dan pola waktu (cluster_report).

Semuanya generik: tidak ada metrik di sini yang bergantung pada nama atau arti
fitur. Kolom sengaja dinamai netral.
"""
import numpy as np
import pandas as pd
import pytest

from app.services import cluster_report


def _dt(n, start="2026-07-01 00:00:00", freq="5min"):
  return pd.Series(pd.date_range(start, periods=n, freq=freq, tz="UTC"))


class TestOutliers:
  def test_finds_the_point_far_from_its_own_centroid(self):
    """Titik berlabel cluster 'normal' tapi duduk jauh dari pusatnya — tidak
    akan pernah muncul lewat label, hanya lewat jarak."""
    df = pd.DataFrame({"f1": [0.0, 0.1, -0.1, 0.0, 40.0], "f2": [0.0, 0.1, 0.0, -0.1, 0.0]})
    labels = pd.Series(["A"] * 5)
    out = cluster_report.outliers(df, labels, ["f1", "f2"], _dt(5), top_k=1)
    assert len(out) == 1
    assert out[0]["dt"].startswith("2026-07-01T00:20")      # baris ke-5

  def test_ratio_compares_against_the_cluster_spread(self):
    df = pd.DataFrame({"f1": [0.0, 0.1, -0.1, 0.0, 40.0], "f2": [0.0] * 5})
    out = cluster_report.outliers(df, pd.Series(["A"] * 5), ["f1", "f2"], _dt(5), top_k=1)
    assert out[0]["ratio"] > 1                               # jauh di atas radius biasa

  def test_reports_which_features_drive_it(self):
    df = pd.DataFrame({"f1": [0, 0, 0, 0, 40.0], "f2": [1.0, 1.1, 0.9, 1.0, 1.0]})
    out = cluster_report.outliers(df, pd.Series(["A"] * 5), ["f1", "f2"], _dt(5), top_k=1)
    assert out[0]["drivers"][0]["feature"] == "f1"
    assert out[0]["drivers"][0]["value"] == 40.0

  def test_sorted_by_distance_and_capped(self):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"f1": rng.normal(0, 1, 50), "f2": rng.normal(0, 1, 50)})
    out = cluster_report.outliers(df, pd.Series(["A"] * 50), ["f1", "f2"], _dt(50), top_k=5)
    assert len(out) == 5
    assert [o["distance"] for o in out] == sorted([o["distance"] for o in out], reverse=True)

  def test_looks_within_each_cluster_not_globally(self):
    """Cluster jauh yang rapat TIDAK boleh dianggap pencilan hanya karena
    posisinya jauh dari pusat data keseluruhan."""
    df = pd.DataFrame({"f1": [0, 0.1, -0.1, 100, 100.1, 99.9], "f2": [0.0] * 6})
    labels = pd.Series(["A", "A", "A", "B", "B", "B"])
    out = cluster_report.outliers(df, labels, ["f1", "f2"], _dt(6), top_k=6)
    assert max(o["distance"] for o in out) < 2               # dua-duanya rapat

  def test_no_usable_features_returns_empty(self):
    df = pd.DataFrame({"f1": [1.0, 1.0]})
    assert cluster_report.outliers(df, pd.Series(["A", "A"]), ["f1"], _dt(2)) == []


class TestAgreement:
  def test_identical_partitions_score_one_despite_different_numbering(self):
    a = pd.Series(["Cluster 0", "Cluster 0", "Cluster 1"])
    b = pd.Series(["Cluster 9", "Cluster 9", "Cluster 3"])   # penomoran beda, isi sama
    assert cluster_report.agreement(a, b)["ari"] == pytest.approx(1.0)
    assert cluster_report.agreement(a, b)["verdict"] == "strong"

  def test_degenerate_split_scores_near_zero(self):
    """Kasus nyata: satu algoritma memecah wajar, satunya menaruh 99% di satu
    cluster — ARI-nya mendekati nol dan harus ditandai 'weak'."""
    a = pd.Series(["A"] * 50 + ["B"] * 50)
    b = pd.Series(["X"] * 99 + ["Y"])
    out = cluster_report.agreement(a, b)
    assert out["ari"] < 0.1 and out["verdict"] == "weak"

  def test_empty_input_returns_none(self):
    assert cluster_report.agreement(pd.Series(dtype=str), pd.Series(dtype=str)) is None


class TestHourlyDistribution:
  def test_shares_sum_to_one_per_hour(self):
    dt = _dt(24, freq="1h")
    labels = pd.Series(["A" if i % 2 else "B" for i in range(24)])
    for row in cluster_report.hourly_distribution(dt, labels):
      assert sum(row["shares"].values()) == pytest.approx(1.0)

  def test_exposes_a_daily_pattern(self):
    """Cluster B hanya muncul pukul 03:00 — polanya harus terbaca."""
    dt = _dt(48, freq="1h")
    labels = pd.Series(["B" if d.hour == 3 else "A" for d in dt])
    rows = {r["hour"]: r["shares"] for r in cluster_report.hourly_distribution(dt, labels)}
    assert rows[3].get("B") == pytest.approx(1.0)
    assert rows[10].get("B", 0) == 0

  def test_empty_input_is_empty(self):
    assert cluster_report.hourly_distribution(pd.Series(dtype="datetime64[ns]"),
                                              pd.Series(dtype=str)) == []


class TestTransitions:
  def test_counts_moves_not_dwell_time(self):
    labels = pd.Series(["A", "A", "A", "B", "B", "A"])
    out = {(t["from"], t["to"]): t["count"] for t in cluster_report.transitions(labels)}
    assert out == {("A", "B"): 1, ("B", "A"): 1}

  def test_sorted_by_frequency(self):
    labels = pd.Series(["A", "B", "A", "B", "A", "C"])
    out = cluster_report.transitions(labels)
    assert out[0]["count"] >= out[-1]["count"]

  def test_never_moving_yields_nothing(self):
    assert cluster_report.transitions(pd.Series(["A"] * 10)) == []


class TestDrift:
  def test_splits_by_time_not_row_count(self):
    """Potongan dibagi menurut waktu; lubang data tidak boleh membuat satu
    potongan mewakili durasi yang jauh lebih panjang."""
    dt = pd.Series(pd.to_datetime(
        ["2026-07-01 00:00", "2026-07-01 01:00",           # padat di awal
         "2026-07-01 02:00", "2026-07-01 23:00"], utc=True))
    labels = pd.Series(["A", "A", "A", "B"])
    out = cluster_report.drift(dt, labels)
    assert len(out) == 2
    assert out[0]["total"] == 3 and out[1]["total"] == 1

  def test_reports_a_composition_shift(self):
    dt = _dt(100, freq="1h")
    labels = pd.Series(["A"] * 50 + ["B"] * 50)
    out = cluster_report.drift(dt, labels)
    assert out[0]["shares"].get("A") == pytest.approx(1.0)
    assert out[1]["shares"].get("B") == pytest.approx(1.0)

  def test_single_timestamp_returns_nothing(self):
    dt = pd.Series(pd.to_datetime(["2026-07-01 00:00"] * 3, utc=True))
    assert cluster_report.drift(dt, pd.Series(["A"] * 3)) == []


class TestReportIntegration:
  def _frame(self):
    n = 60
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "dt": pd.date_range("2026-07-01", periods=n, freq="1h", tz="UTC"),
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(0, 1, n),
        "algo1": ["A" if i % 3 else "B" for i in range(n)],
        "algo2": ["X" if i % 3 else "Y" for i in range(n)],
    })

  def test_every_section_is_present(self):
    out = cluster_report.build_report(self._frame(), ["algo1", "algo2"], ["f1", "f2"])
    for key in ("composition", "episodes", "geometry", "outliers",
                "hourly", "transitions", "drift", "agreement"):
      assert key in out, key

  def test_agreement_is_keyed_by_algorithm_pair(self):
    out = cluster_report.build_report(self._frame(), ["algo1", "algo2"], ["f1", "f2"])
    assert list(out["agreement"]) == ["algo1|algo2"]
    assert out["agreement"]["algo1|algo2"]["ari"] == pytest.approx(1.0)

  def test_single_algorithm_has_no_agreement_entry(self):
    out = cluster_report.build_report(self._frame(), ["algo1"], ["f1", "f2"])
    assert out["agreement"] == {}

  def test_empty_window_keeps_the_same_shape(self):
    out = cluster_report.build_report(self._frame().iloc[0:0], ["algo1"], ["f1"])
    for key in ("outliers", "hourly", "transitions", "drift"):
      assert out[key]["algo1"] == []
