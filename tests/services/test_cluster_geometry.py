"""app.services.cluster_report.cluster_geometry — bentuk cluster di ruang fitur.

Nama kolom sengaja netral (f1, f2, …): tidak satu pun metrik di sini boleh
bergantung pada arti fitur. Yang dijaga terutama tiga hal yang tidak terlihat
di scatter plot maupun tabel centroid — fitur mana yang benar-benar membedakan,
apakah sebuah "cluster" sebenarnya kumpulan pencilan, dan apakah dua cluster
sebenarnya saling tumpang tindih.
"""
import numpy as np
import pandas as pd
import pytest

from app.services import cluster_report


def _two_groups(spread_b=0.1, gap=10.0, n=50):
  """Dua kelompok yang terpisah pada f1; f2 hanya derau; f3 konstan."""
  rng = np.random.default_rng(42)
  a = pd.DataFrame({"f1": rng.normal(0, 0.1, n), "f2": rng.normal(0, 1, n), "f3": 7.0})
  b = pd.DataFrame({"f1": rng.normal(gap, spread_b, n), "f2": rng.normal(0, 1, n), "f3": 7.0})
  df = pd.concat([a, b], ignore_index=True)
  labels = pd.Series(["A"] * n + ["B"] * n)
  return df, labels, ["f1", "f2", "f3"]


class TestZScores:
  def test_constant_columns_are_dropped(self):
    df, _, feats = _two_groups()
    assert list(cluster_report.z_scores(df, feats).columns) == ["f1", "f2"]

  def test_result_is_standardised(self):
    df, _, feats = _two_groups()
    z = cluster_report.z_scores(df, feats)
    assert z["f1"].mean() == pytest.approx(0, abs=1e-9)
    assert z["f1"].std() == pytest.approx(1, abs=1e-9)

  def test_all_constant_gives_empty_frame_not_nan(self):
    df = pd.DataFrame({"f1": [1, 1, 1]})
    assert cluster_report.z_scores(df, ["f1"]).shape[1] == 0


class TestDistinctiveFeatures:
  def test_ranks_the_feature_that_actually_separates(self):
    """f1 memisahkan kedua kelompok, f2 hanya derau — urutannya harus f1 dulu."""
    df, labels, feats = _two_groups()
    out = cluster_report.cluster_geometry(df, labels, feats)["distinctive"]
    assert out["A"][0]["feature"] == "f1"
    assert out["B"][0]["feature"] == "f1"

  def test_direction_is_reported(self):
    df, labels, feats = _two_groups()
    out = cluster_report.cluster_geometry(df, labels, feats)["distinctive"]
    assert out["A"][0]["direction"] == "below"      # A di bawah rata-rata global
    assert out["B"][0]["direction"] == "above"

  def test_carries_original_units_not_just_z(self):
    """Angka ber-z tidak bisa dibaca operator; nilai aslinya harus ikut."""
    df, labels, feats = _two_groups(gap=10.0)
    top = cluster_report.cluster_geometry(df, labels, feats)["distinctive"]["B"][0]
    assert top["mean"] == pytest.approx(10.0, abs=0.5)
    assert top["p10"] <= top["mean"] <= top["p90"]

  def test_constant_feature_never_appears(self):
    df, labels, feats = _two_groups()
    out = cluster_report.cluster_geometry(df, labels, feats)["distinctive"]
    assert all(f["feature"] != "f3" for rows in out.values() for f in rows)

  def test_top_k_limits_the_list(self):
    df, labels, feats = _two_groups()
    out = cluster_report.cluster_geometry(df, labels, feats, top_k=1)["distinctive"]
    assert len(out["A"]) == 1


class TestCompactness:
  def test_tight_cluster_has_small_radius(self):
    df, labels, feats = _two_groups(spread_b=0.1)
    comp = cluster_report.cluster_geometry(df, labels, feats)["compactness"]
    assert comp["B"]["mean_radius"] < 1.0

  def test_scattered_group_is_exposed_by_a_large_radius(self):
    """Kasus nyata: sebuah 'cluster' beranggota 8 titik dengan radius 18,
    sementara cluster lain 1,6 — itu pencilan, bukan kelompok.

    Polanya harus ditiru persis: segelintir titik yang saling berjauhan di
    tengah data yang rapat. Kalau kelompok penyebarnya besar, sebarannya sendiri
    ikut menggelembungkan standar deviasi global dan kontrasnya tersamar.
    """
    rng = np.random.default_rng(0)
    main = pd.DataFrame({"f1": rng.normal(0, 1, 200), "f2": rng.normal(0, 1, 200)})
    strays = pd.DataFrame({"f1": [50.0, -60.0, 80.0], "f2": [-70.0, 90.0, 40.0]})
    df = pd.concat([main, strays], ignore_index=True)
    labels = pd.Series(["A"] * 200 + ["B"] * 3)

    comp = cluster_report.cluster_geometry(df, labels, ["f1", "f2"])["compactness"]
    assert comp["B"]["mean_radius"] > comp["A"]["mean_radius"] * 3

  def test_p95_is_at_least_the_mean(self):
    df, labels, feats = _two_groups()
    for stats in cluster_report.cluster_geometry(df, labels, feats)["compactness"].values():
      assert stats["p95_radius"] >= stats["mean_radius"]


class TestSeparation:
  def test_distant_clusters_are_not_flagged_as_overlapping(self):
    df, labels, feats = _two_groups(gap=10.0, spread_b=0.1)
    pair = cluster_report.cluster_geometry(df, labels, feats)["separation"][0]
    assert pair["overlapping"] is False
    assert pair["separation"] > 1

  def test_clusters_sitting_on_top_of_each_other_are_flagged(self):
    """Dua cluster yang jarak centroid-nya lebih kecil dari sebarannya sendiri
    hanya terpisah di atas kertas."""
    df, labels, feats = _two_groups(gap=0.2, spread_b=1.0)
    pair = cluster_report.cluster_geometry(df, labels, feats)["separation"][0]
    assert pair["overlapping"] is True
    assert pair["separation"] < 1

  def test_lists_every_pair_once(self):
    n = 20
    df = pd.DataFrame({"f1": list(range(3 * n)), "f2": [1.0] * (3 * n)})
    labels = pd.Series(["A"] * n + ["B"] * n + ["C"] * n)
    sep = cluster_report.cluster_geometry(df, labels, ["f1", "f2"])["separation"]
    assert len(sep) == 3                                    # AB, AC, BC
    assert all(p["between"][0] < p["between"][1] for p in sep)

  def test_sorted_by_distance_so_the_closest_pair_is_first(self):
    df, labels, feats = _two_groups()
    sep = cluster_report.cluster_geometry(df, labels, feats)["separation"]
    assert sep == sorted(sep, key=lambda s: s["distance"])


class TestDegenerateInput:
  def test_single_cluster_has_no_pairs(self):
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    out = cluster_report.cluster_geometry(df, pd.Series(["A"] * 3), ["f1"])
    assert out["separation"] == []
    assert "A" in out["compactness"]

  def test_only_constant_features_returns_empty_not_nan(self):
    df = pd.DataFrame({"f1": [5.0, 5.0]})
    out = cluster_report.cluster_geometry(df, pd.Series(["A", "B"]), ["f1"])
    assert out == {"distinctive": {}, "compactness": {}, "separation": []}

  def test_missing_values_do_not_poison_the_distances(self):
    df = pd.DataFrame({"f1": [0.0, 1.0, np.nan, 10.0], "f2": [1.0, 2.0, 3.0, 4.0]})
    out = cluster_report.cluster_geometry(df, pd.Series(["A", "A", "A", "B"]), ["f1", "f2"])
    assert not np.isnan(out["compactness"]["A"]["mean_radius"])


class TestReportIntegration:
  def test_geometry_is_part_of_the_report(self):
    df, labels, feats = _two_groups()
    df["dt"] = pd.date_range("2026-07-01", periods=len(df), freq="5min", tz="UTC")
    df["algo"] = labels
    out = cluster_report.build_report(df, ["algo"], feats)
    assert set(out["geometry"]["algo"]) == {"distinctive", "compactness", "separation"}
    assert out["geometry"]["algo"]["distinctive"]["A"][0]["feature"] == "f1"
