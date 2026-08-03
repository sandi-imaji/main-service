"""app.services.cluster_report — ringkasan clustering.

Semua metrik di sini harus bisa dihitung TANPA tahu arti fitur, jadi test-nya
memakai nama kolom netral (f1, f2). Yang dijaga: episode adalah kompresi dari
deret label (bukan sekadar hitungan titik), kolom konstan dikenali sebelum
sempat merusak perhitungan, dan bentuk kosong tidak menjatuhkan endpoint.
"""
import pandas as pd
import pytest

from app.services import cluster_report


def _frame(labels, start="2026-07-01 00:00:00", freq="5min", features=None):
  n = len(labels)
  data = {"dt": pd.date_range(start, periods=n, freq=freq, tz="UTC"), "algo": labels}
  data.update(features or {"f1": list(range(n)), "f2": [1.0] * n})
  return pd.DataFrame(data)


class TestComposition:
  def test_counts_and_shares(self):
    out = cluster_report.composition(pd.Series(["A", "A", "A", "B"]))
    by_label = {c["label"]: c for c in out["clusters"]}
    assert out["total"] == 4 and out["effective_clusters"] == 2
    assert by_label["A"]["count"] == 3
    assert by_label["A"]["share"] == pytest.approx(0.75)

  def test_balance_is_one_when_even(self):
    assert cluster_report.composition(pd.Series(["A", "A", "B", "B"]))["balance"] == pytest.approx(1.0)

  def test_balance_drops_when_one_cluster_swallows_everything(self):
    """Kasus nyata: spectral clustering menaruh 99,8% titik di satu cluster."""
    labels = pd.Series(["A"] * 999 + ["B"])
    assert cluster_report.composition(labels)["balance"] < 0.05

  def test_micro_cluster_is_flagged(self):
    labels = pd.Series(["A"] * 995 + ["B"] * 5)      # B = 0,5%
    by_label = {c["label"]: c for c in cluster_report.composition(labels)["clusters"]}
    assert by_label["B"]["is_micro"] is True
    assert by_label["A"]["is_micro"] is False

  def test_human_name_is_attached_when_available(self):
    out = cluster_report.composition(pd.Series(["Cluster 0"]), naming={"Cluster 0": "dingin"})
    assert out["clusters"][0]["name"] == "dingin"

  def test_empty_input_does_not_raise(self):
    out = cluster_report.composition(pd.Series(dtype=str))
    assert out["total"] == 0 and out["clusters"] == []


class TestEpisodes:
  def test_consecutive_rows_collapse_into_one_episode(self):
    df = _frame(["A", "A", "A", "B", "B"])
    eps = cluster_report.episodes(df["dt"], df["algo"], 5.0)
    assert [(e["label"], e["points"]) for e in eps] == [("A", 3), ("B", 2)]

  def test_same_label_returning_later_is_a_separate_episode(self):
    """Inti panel ini: 4 titik A yang terpisah ≠ satu kejadian panjang."""
    df = _frame(["A", "B", "A", "B", "A"])
    eps = cluster_report.episodes(df["dt"], df["algo"], 5.0)
    assert len([e for e in eps if e["label"] == "A"]) == 3

  def test_duration_counts_the_last_interval(self):
    """Episode 1 baris berdurasi satu interval, bukan 0 menit."""
    df = _frame(["A"])
    assert cluster_report.episodes(df["dt"], df["algo"], 5.0)[0]["duration_minutes"] == 5.0

  def test_duration_spans_the_whole_run(self):
    df = _frame(["A", "A", "A"])                     # 3 baris × 5 menit
    assert cluster_report.episodes(df["dt"], df["algo"], 5.0)[0]["duration_minutes"] == 15.0

  def test_unsorted_input_is_ordered_first(self):
    df = _frame(["A", "A", "B"]).iloc[::-1]          # dibalik
    eps = cluster_report.episodes(df["dt"], df["algo"], 5.0)
    assert [e["label"] for e in eps] == ["A", "B"]

  def test_empty_input_returns_no_episodes(self):
    assert cluster_report.episodes(pd.Series(dtype="datetime64[ns]"),
                                   pd.Series(dtype=str), 5.0) == []


class TestEpisodeStats:
  def test_aggregates_per_cluster(self):
    df = _frame(["A", "A", "B", "A"])
    stats = cluster_report.episode_stats(cluster_report.episodes(df["dt"], df["algo"], 5.0))
    assert stats["A"]["episodes"] == 2
    assert stats["A"]["total_minutes"] == 15.0        # (2×5) + (1×5)
    assert stats["A"]["max_minutes"] == 10.0
    assert stats["B"]["episodes"] == 1


class TestIntervalAndConstantFeatures:
  def test_interval_is_inferred_from_the_data(self):
    df = _frame(["A"] * 4, freq="15min")
    assert cluster_report.infer_interval_minutes(df["dt"]) == 15.0

  def test_single_row_falls_back_instead_of_raising(self):
    df = _frame(["A"])
    assert cluster_report.infer_interval_minutes(df["dt"]) == cluster_report.DEFAULT_INTERVAL_MINUTES

  def test_constant_columns_are_detected(self):
    """19 dari 30 fitur pada dataset nyata bernilai tetap; kalau ikut
    dinormalisasi hasilnya pembagian dengan nol."""
    df = _frame(["A", "B"], features={"varies": [1, 2], "flat": [7, 7]})
    assert cluster_report.constant_features(df, ["varies", "flat"]) == ["flat"]


class TestBuildReport:
  def test_full_shape(self):
    df = _frame(["A", "A", "B", "B", "A"])
    out = cluster_report.build_report(df, ["algo"], ["f1", "f2"])

    assert out["total"] == 5
    assert out["interval_minutes"] == 5.0
    assert out["range"]["start"].startswith("2026-07-01T00:00")
    assert out["features"] == {"total": 2, "effective": 1, "constant": ["f2"]}
    assert out["composition"]["algo"]["effective_clusters"] == 2
    assert len(out["episodes"]["algo"]) == 3
    assert out["episode_stats"]["algo"]["A"]["episodes"] == 2

  def test_empty_window_returns_empty_report_not_an_error(self):
    """Rentang tanggal yang tidak memuat data adalah hal biasa — UI cukup
    menampilkan kosong, bukan 500."""
    out = cluster_report.build_report(_frame([]).iloc[0:0], ["algo"], ["f1"])
    assert out["total"] == 0
    assert out["episodes"]["algo"] == []
    assert out["composition"]["algo"]["clusters"] == []

  def test_unknown_algorithm_column_is_ignored(self):
    df = _frame(["A", "B"])
    out = cluster_report.build_report(df, ["algo", "tidak_ada"], ["f1"])
    assert set(out["composition"]) == {"algo"}
