"""app.services.dataset_table — isi dataset untuk tabel dan unduhan.

Yang dijaga terutama satu janji: apa yang TAMPIL di tabel sama dengan apa yang
TERUNDUH. Untuk clustering itu berarti `results/clusters.csv` dengan nama
cluster sudah dipetakan, bukan `data.csv` mentah.
"""
import io
import json

import pandas as pd
import pytest

from app.config import Config
from app.database.schemas import TaskType
from app.exceptions import NotFoundException, ValidationException
from app.services import dataset_table


class _Dataset:
  def __init__(self, name, task_type):
    self.name = name
    self.task_type = task_type


def _siapkan(tmp_path, monkeypatch, task_type=TaskType.Regression,
            n=120, clusters=False, naming=None):
  monkeypatch.setattr(Config, "dir", tmp_path)
  nama = f"{task_type.name}-uji"
  results = tmp_path / "storages" / nama / "results"
  results.mkdir(parents=True)

  data = pd.DataFrame({
      "dt": pd.date_range("2026-07-01", periods=n, freq="5min", tz="UTC").astype(str),
      "x1": range(n), "x2": [i * 0.5 for i in range(n)],
  })
  data.to_csv(tmp_path / "storages" / nama / "data.csv", index=False)

  if clusters:
    kolom = data.copy()
    kolom["kmeans"] = ["Cluster 0" if i % 2 else "Cluster 1" for i in range(n)]
    kolom.to_csv(results / "clusters.csv", index=False)
  if naming is not None:
    (results / "naming_clusters.json").write_text(json.dumps(naming))

  return _Dataset(nama, task_type)


class TestPemilihanSumber:
  def test_non_clustering_memakai_data_csv(self, tmp_path, monkeypatch):
    ds = _siapkan(tmp_path, monkeypatch, TaskType.Regression)
    _, sumber = dataset_table.resolve_source(ds)
    assert sumber == "data.csv"

  def test_clustering_memakai_hasil_cluster(self, tmp_path, monkeypatch):
    ds = _siapkan(tmp_path, monkeypatch, TaskType.Clustering, clusters=True)
    df, sumber = dataset_table.resolve_source(ds)
    assert sumber == "results/clusters.csv"
    assert "kmeans" in df.columns                    # kolom label ikut tampil

  def test_clustering_belum_dilatih_jatuh_ke_data_csv(self, tmp_path, monkeypatch):
    """Tabel harus tetap bisa dibuka sebelum training, bukan halaman kosong."""
    ds = _siapkan(tmp_path, monkeypatch, TaskType.Clustering, clusters=False)
    df, sumber = dataset_table.resolve_source(ds)
    assert sumber == "data.csv" and "kmeans" not in df.columns

  def test_data_csv_hilang_memberi_404_yang_jelas(self, tmp_path, monkeypatch):
    ds = _siapkan(tmp_path, monkeypatch, TaskType.Regression)
    (tmp_path / "storages" / ds.name / "data.csv").unlink()
    with pytest.raises(NotFoundException):
      dataset_table.resolve_source(ds)


class TestPenamaanCluster:
  def test_nama_pemberian_user_dipakai(self, tmp_path, monkeypatch):
    ds = _siapkan(tmp_path, monkeypatch, TaskType.Clustering, clusters=True,
                  naming={"kmeans": {"Cluster 0": "dingin", "Cluster 1": "panas"}})
    df, _ = dataset_table.resolve_source(ds)
    assert set(df["kmeans"]) == {"dingin", "panas"}

  def test_label_yang_belum_dinamai_tidak_hilang(self, tmp_path, monkeypatch):
    """Regression guard: versi lama memakai `.map`, yang mengubah label di luar
    pemetaan menjadi NaN — penamaan separuh jalan menghapus data."""
    ds = _siapkan(tmp_path, monkeypatch, TaskType.Clustering, clusters=True,
                  naming={"kmeans": {"Cluster 0": "dingin"}})
    df, _ = dataset_table.resolve_source(ds)
    assert set(df["kmeans"]) == {"dingin", "Cluster 1"}
    assert df["kmeans"].isna().sum() == 0

  def test_tanpa_penamaan_label_mentah_dipertahankan(self, tmp_path, monkeypatch):
    ds = _siapkan(tmp_path, monkeypatch, TaskType.Clustering, clusters=True)
    df, _ = dataset_table.resolve_source(ds)
    assert set(df["kmeans"]) == {"Cluster 0", "Cluster 1"}


class TestPaginasi:
  def test_bentuk_respons(self, tmp_path, monkeypatch):
    ds = _siapkan(tmp_path, monkeypatch, n=120)
    out = dataset_table.page(ds, page=1, page_size=50)
    assert out["total"] == 120 and out["pages"] == 3
    assert len(out["rows"]) == 50
    assert out["columns"] == ["dt", "x1", "x2"]      # urutan kolom terjaga

  def test_halaman_terakhir_boleh_kurang_dari_satu_halaman(self, tmp_path, monkeypatch):
    ds = _siapkan(tmp_path, monkeypatch, n=120)
    assert len(dataset_table.page(ds, page=3, page_size=50)["rows"]) == 20

  def test_halaman_di_luar_batas_dijepit(self, tmp_path, monkeypatch):
    """Meminta halaman 99 sebaiknya membalas halaman terakhir, bukan kosong."""
    ds = _siapkan(tmp_path, monkeypatch, n=120)
    out = dataset_table.page(ds, page=99, page_size=50)
    assert out["page"] == 3 and out["rows"]

  def test_halaman_berbeda_berisi_baris_berbeda(self, tmp_path, monkeypatch):
    ds = _siapkan(tmp_path, monkeypatch, n=120)
    p1 = dataset_table.page(ds, page=1, page_size=10)["rows"]
    p2 = dataset_table.page(ds, page=2, page_size=10)["rows"]
    assert p1[0]["x1"] == 0 and p2[0]["x1"] == 10

  def test_page_size_di_luar_batas_ditolak(self, tmp_path, monkeypatch):
    ds = _siapkan(tmp_path, monkeypatch)
    with pytest.raises(ValidationException):
      dataset_table.page(ds, page_size=dataset_table.MAX_PAGE_SIZE + 1)

  def test_dataset_kosong_tidak_membagi_dengan_nol(self, tmp_path, monkeypatch):
    ds = _siapkan(tmp_path, monkeypatch, n=0)
    out = dataset_table.page(ds)
    assert out["total"] == 0 and out["pages"] == 1 and out["rows"] == []


class TestPengurutan:
  def test_urut_menurun(self, tmp_path, monkeypatch):
    ds = _siapkan(tmp_path, monkeypatch, n=30)
    out = dataset_table.page(ds, page=1, page_size=5, sort="x1", order="desc")
    assert [r["x1"] for r in out["rows"]] == [29, 28, 27, 26, 25]

  def test_pengurutan_lintas_halaman_bukan_hanya_halaman_ini(self, tmp_path, monkeypatch):
    """Justru inti pengurutan di server: baris terbesar harus muncul di halaman
    pertama, bukan hanya diurutkan di antara 5 baris yang kebetulan tampil."""
    ds = _siapkan(tmp_path, monkeypatch, n=100)
    out = dataset_table.page(ds, page=1, page_size=5, sort="x1", order="desc")
    assert out["rows"][0]["x1"] == 99

  def test_kolom_tak_dikenal_ditolak(self, tmp_path, monkeypatch):
    ds = _siapkan(tmp_path, monkeypatch)
    with pytest.raises(ValidationException):
      dataset_table.page(ds, sort="tidak_ada")


class TestUnduhan:
  def test_csv_memuat_seluruh_baris(self, tmp_path, monkeypatch):
    ds = _siapkan(tmp_path, monkeypatch, n=120)
    buffer, nama = dataset_table.to_csv(ds)
    df = pd.read_csv(io.BytesIO(buffer.getvalue()))
    assert len(df) == 120                            # utuh, bukan satu halaman
    assert nama.endswith("-data.csv")

  def test_csv_clustering_memuat_nama_cluster(self, tmp_path, monkeypatch):
    """Janji utamanya: berkas unduhan sama dengan yang tampil."""
    ds = _siapkan(tmp_path, monkeypatch, TaskType.Clustering, clusters=True,
                  naming={"kmeans": {"Cluster 0": "dingin", "Cluster 1": "panas"}})
    buffer, nama = dataset_table.to_csv(ds)
    df = pd.read_csv(io.BytesIO(buffer.getvalue()))
    assert set(df["kmeans"]) == {"dingin", "panas"}
    assert nama.endswith("-clusters.csv")

  def test_xlsx_bisa_dibuka_kembali(self, tmp_path, monkeypatch):
    ds = _siapkan(tmp_path, monkeypatch, n=40)
    buffer, nama = dataset_table.to_excel(ds)
    df = pd.read_excel(io.BytesIO(buffer.getvalue()))
    assert len(df) == 40 and nama.endswith("-data.xlsx")

  def test_xlsx_menulis_dt_sebagai_tanggal(self, tmp_path, monkeypatch):
    """openpyxl menolak datetime bertimezone; offsetnya harus dibuang lebih dulu
    supaya ekspor tidak gagal DAN selnya tetap bertipe tanggal."""
    ds = _siapkan(tmp_path, monkeypatch, n=10)
    buffer, _ = dataset_table.to_excel(ds)
    df = pd.read_excel(io.BytesIO(buffer.getvalue()))
    assert pd.api.types.is_datetime64_any_dtype(df["dt"])

  def test_xlsx_clustering_tetap_jalan_dengan_kolom_label(self, tmp_path, monkeypatch):
    ds = _siapkan(tmp_path, monkeypatch, TaskType.Clustering, clusters=True,
                  naming={"kmeans": {"Cluster 0": "dingin"}})
    buffer, _ = dataset_table.to_excel(ds)
    df = pd.read_excel(io.BytesIO(buffer.getvalue()))
    assert "kmeans" in df.columns and "dingin" in set(df["kmeans"])


class TestJsonAman:
  def test_nilai_kosong_menjadi_null_bukan_nan(self, tmp_path, monkeypatch):
    """JSON tidak punya NaN — mengirimnya apa adanya membuat client gagal parse."""
    monkeypatch.setattr(Config, "dir", tmp_path)
    nama = "Regression-nan"
    (tmp_path / "storages" / nama).mkdir(parents=True)
    pd.DataFrame({"dt": ["2026-07-01T00:00:00Z"], "x1": [None]}).to_csv(
        tmp_path / "storages" / nama / "data.csv", index=False)

    out = dataset_table.page(_Dataset(nama, TaskType.Regression))
    assert out["rows"][0]["x1"] is None
    json.dumps(out)                                  # harus bisa diserialisasi
