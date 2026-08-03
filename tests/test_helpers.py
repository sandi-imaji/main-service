"""app.helpers — utilitas waktu, log, dan dataframe.

Isinya dipakai hampir di seluruh service, jadi kesalahan di sini menyebar jauh.
Yang tidak diuji di sini: `get_row_id`, karena memanggil API SmartLink (dijaga
`tests/test_pull.py` sebagai test integrasi), dan `parse_log_line` yang
round-trip-nya sudah dikunci di
`tests/utils/test_log_pipeline.py`.
"""
import datetime

import numpy as np
import pandas as pd
import pytest

from app.helpers import (
  DTEncoder,
  init_storages,
  miss_val_handling_df,
  parse_log_line,
  pca,
  read_logs,
)


class TestDTEncoder:
  def test_now(self):
    dt_now  = DTEncoder.now()
    now = datetime.datetime.now()
    assert dt_now.date() == now.date()
    assert dt_now.hour == now.hour
    assert dt_now.minute == now.minute
    assert dt_now.tzinfo == datetime.UTC

  def test_to_pull_str(self):
    timestamp = "10/5/2026"
    date_object = datetime.datetime.strptime(timestamp, "%d/%m/%Y")
    pull_str = DTEncoder.dt_to_str(date_object)
    assert pull_str == "20260510"
    dt = DTEncoder.str_to_dt(pull_str)
    print(dt)


class TestKonvensiWaktu:
  """`now()` memakai konvensi UTC+7 yang dilabeli UTC.

  Nilainya jam Jakarta, tapi `tzinfo`-nya UTC — jadi timestamp-nya "berbohong"
  soal zonanya sendiri. Seluruh sistem memakai konvensi yang sama secara
  konsisten (pull, Influx, penjadwalan), jadi dikunci di sini supaya perubahan
  di satu tempat tidak diam-diam menggeser yang lain.
  """

  def test_now_tujuh_jam_di_depan_utc(self):
    selisih = DTEncoder.now() - DTEncoder.now_utc()
    assert abs(selisih.total_seconds() - 7 * 3600) < 2

  def test_now_utc_benar_benar_utc(self):
    beda = DTEncoder.now_utc() - datetime.datetime.now(datetime.timezone.utc)
    assert abs(beda.total_seconds()) < 2

  def test_to_utc_membalik_konvensi(self):
    dt = DTEncoder.now()
    assert abs((dt - DTEncoder.to_utc(dt)).total_seconds() - 7 * 3600) < 1

  def test_to_utc_dan_from_utc_saling_membatalkan(self):
    dt = datetime.datetime(2026, 7, 20, 10, 0, tzinfo=datetime.timezone.utc)
    assert DTEncoder.from_utc(DTEncoder.to_utc(dt)) == dt

  def test_from_utc_menerima_string_iso(self):
    hasil = DTEncoder.from_utc("2026-07-20T03:00:00+00:00")
    assert hasil.hour == 10                      # +7 jam


class TestKonversiTanggal:
  @pytest.mark.parametrize("teks,harapan", [
    ("20260510", (2026, 5, 10)),
    ("2026-05-10", (2026, 5, 10)),
    ("10/05/2026", (2026, 5, 10)),
    ("10-05-2026", (2026, 5, 10)),
  ])
  def test_str_to_dt_menerima_beberapa_format(self, teks, harapan):
    dt = DTEncoder.str_to_dt(teks)
    assert (dt.year, dt.month, dt.day) == harapan
    assert dt.tzinfo == datetime.timezone.utc

  def test_str_to_dt_menolak_format_asing(self):
    with pytest.raises(ValueError, match="tidak dikenali"):
      DTEncoder.str_to_dt("Mei 10 2026")

  def test_str_to_dt_menolak_bukan_string(self):
    with pytest.raises(TypeError):
      DTEncoder.str_to_dt(20260510)

  def test_dt_to_str_menolak_bukan_datetime(self):
    with pytest.raises(TypeError):
      DTEncoder.dt_to_str("20260510")

  def test_bolak_balik_dt_dan_string(self):
    dt = datetime.datetime(2026, 12, 31, tzinfo=datetime.timezone.utc)
    assert DTEncoder.str_to_dt(DTEncoder.dt_to_str(dt)).date() == dt.date()

  def test_now_str_memakai_format_pull(self):
    teks = DTEncoder.now_str()
    assert len(teks) == 8 and teks.isdigit()


class TestUnixTimestamp:
  def test_dalam_milidetik_bukan_detik(self):
    dt = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    assert DTEncoder.dt_to_unixTS(dt) == int(dt.timestamp() * 1000)

  def test_datetime_naif_dianggap_utc(self):
    naif = datetime.datetime(2026, 1, 1)
    aware = naif.replace(tzinfo=datetime.timezone.utc)
    assert DTEncoder.dt_to_unixTS(naif) == DTEncoder.dt_to_unixTS(aware)

  def test_dt_to_unixTS_menolak_bukan_datetime(self):
    with pytest.raises(TypeError):
      DTEncoder.dt_to_unixTS(1767225600000)

  @pytest.mark.parametrize("nilai", [1767225600000, 1767225600000.0, "1767225600000"])
  def test_unixTS_to_dt_menerima_int_float_string(self, nilai):
    assert DTEncoder.unixTS_to_dt(nilai).year == 2026

  def test_unixTS_to_dt_menolak_yang_tak_bisa_dikonversi(self):
    with pytest.raises(TypeError):
      DTEncoder.unixTS_to_dt("bukan angka")

  def test_unixTS_to_dt_ikut_konvensi_plus_tujuh(self):
    """Sama seperti `now()`: hasilnya digeser +7 jam meski berlabel UTC —
    inilah yang membuat kolom `dt` hasil pull sejalan dengan `now()`."""
    dt = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    hasil = DTEncoder.unixTS_to_dt(DTEncoder.dt_to_unixTS(dt))
    assert (hasil - dt).total_seconds() == 7 * 3600


class TestJendelaWaktu:
  def test_maju_dan_mundur_saling_membatalkan(self):
    dt = datetime.datetime(2026, 7, 20, 12, 0, tzinfo=datetime.timezone.utc)
    maju = DTEncoder.get_end_datetime(dt, interval_minutes=5, n=12)
    assert DTEncoder.get_start_datetime(maju, interval_minutes=5, n=12) == dt

  def test_end_datetime_maju_sebanyak_n_interval(self):
    dt = datetime.datetime(2026, 7, 20, 0, 0, tzinfo=datetime.timezone.utc)
    assert DTEncoder.get_end_datetime(dt, 5, 12).hour == 1      # 12 × 5 menit

  def test_generate_dt_menghasilkan_n_langkah_ke_depan(self):
    dt = datetime.datetime(2026, 7, 20, 0, 0, tzinfo=datetime.timezone.utc)
    hasil = DTEncoder.generate_dt(n=3, interval_minutes=10, last_dt=dt)
    assert [d.minute for d in hasil] == [10, 20, 30]            # tidak memuat dt itu sendiri

  def test_generate_dt_bisa_mengembalikan_string_iso(self):
    dt = datetime.datetime(2026, 7, 20, tzinfo=datetime.timezone.utc)
    hasil = DTEncoder.generate_dt(n=2, interval_minutes=5, last_dt=dt, to_str=True)
    assert all(isinstance(x, str) and "T" in x for x in hasil)

  @pytest.mark.parametrize("n", [0, -1])
  def test_generate_dt_menolak_n_tidak_positif(self, n):
    with pytest.raises(ValueError):
      DTEncoder.generate_dt(n=n, interval_minutes=5)

  def test_compare_memakai_toleransi(self):
    a = datetime.datetime(2026, 7, 20, 0, 0, 0, tzinfo=datetime.timezone.utc)
    assert DTEncoder.compare(a, a + datetime.timedelta(seconds=1)) is True
    assert DTEncoder.compare(a, a + datetime.timedelta(seconds=5)) is False

  def test_compare_toleransi_bisa_disetel(self):
    a = datetime.datetime(2026, 7, 20, tzinfo=datetime.timezone.utc)
    assert DTEncoder.compare(a, a + datetime.timedelta(seconds=5), tolerance=10) is True


class TestTimeAgo:
  """Dipakai panel aktivitas dashboard, jadi batas satuannya dikunci."""

  SEKARANG = datetime.datetime(2026, 7, 20, 12, 0, tzinfo=datetime.timezone.utc)

  @pytest.mark.parametrize("lalu,harapan", [
    (30, "30 seconds ago"),
    (60, "1 minutes ago"),
    (3599, "59 minutes ago"),
    (3600, "1 hours ago"),
    (86399, "23 hours ago"),
    (86400, "1 days ago"),
    (86400 * 3, "3 days ago"),
  ])
  def test_batas_setiap_satuan(self, lalu, harapan):
    dt = self.SEKARANG - datetime.timedelta(seconds=lalu)
    assert DTEncoder.time_ago(dt, self.SEKARANG) == harapan

  def test_datetime_naif_tidak_melempar(self):
    """Nilai dari kolom JSON kadang tanpa offset; membandingkannya dengan
    `now()` yang aware akan melempar TypeError tanpa normalisasi."""
    naif = datetime.datetime(2026, 7, 20, 11, 0)
    assert DTEncoder.time_ago(naif, self.SEKARANG).endswith("ago")


class TestMissValHandling:
  def _df(self, nilai):
    return pd.DataFrame({
        "dt": pd.date_range("2026-07-01", periods=len(nilai), freq="5min", tz="UTC"),
        "x": nilai,
    })

  def test_neighbor_value_mengisi_dari_tetangga(self, mock_logger):
    hasil = miss_val_handling_df(self._df([1.0, None, 3.0]), mock_logger, "NEIGHBOR_VALUE")
    assert hasil["x"].tolist() == [1.0, 1.0, 3.0]              # maju dari nilai sebelumnya

  def test_neighbor_value_mengisi_ke_belakang_bila_baris_pertama_kosong(self, mock_logger):
    hasil = miss_val_handling_df(self._df([None, 2.0, 3.0]), mock_logger, "NEIGHBOR_VALUE")
    assert hasil["x"].tolist() == [2.0, 2.0, 3.0]

  @pytest.mark.parametrize("mode,harapan", [("MEAN", 2.0), ("MEDIAN", 2.0),
                                            ("MIN", 1.0), ("MAX", 3.0)])
  def test_mode_statistik(self, mode, harapan, mock_logger):
    hasil = miss_val_handling_df(self._df([1.0, None, 3.0]), mock_logger, mode)
    assert hasil["x"].iloc[1] == harapan

  def test_mode_none_membiarkan_data_apa_adanya(self, mock_logger):
    df = self._df([1.0, None, 3.0])
    hasil = miss_val_handling_df(df, mock_logger, "NONE")
    assert hasil["x"].isna().sum() == 1

  def test_tanpa_nilai_kosong_mengembalikan_frame_utuh(self, mock_logger):
    hasil = miss_val_handling_df(self._df([1.0, 2.0, 3.0]), mock_logger, "MEAN")
    assert hasil is not None and len(hasil) == 3

  @pytest.mark.xfail(strict=True, reason=(
      "Bug tercatat: `return df` hanya dieksekusi bila SELURUH NaN berhasil "
      "terisi. Kolom yang seluruhnya kosong dengan mode MEAN menghasilkan "
      "fillna(NaN), sehingga fungsi mengembalikan None dan pemanggil "
      "(`pull_history`) kena AttributeError pada `df.shape`."))
  def test_kolom_seluruhnya_kosong_tidak_boleh_mengembalikan_none(self, mock_logger):
    hasil = miss_val_handling_df(self._df([None, None, None]), mock_logger, "MEAN")
    assert hasil is not None


class TestPca:
  def _df(self):
    return pd.DataFrame({
        "dt": pd.date_range("2026-07-01", periods=6, freq="5min", tz="UTC"),
        "x1": [1.0, 2, 3, 4, 5, 6], "x2": [2.0, 1, 4, 3, 6, 5],
    })

  def test_pca_mengabaikan_kolom_dt(self):
    obj, hasil = pca(self._df(), n_components=2)
    assert hasil.shape == (6, 2)                               # bukan 3 kolom
    assert obj.n_features_in_ == 2

  def test_pca_only_obj_mengembalikan_model_saja(self):
    obj = pca(self._df(), n_components=2, only_obj=True)
    assert hasattr(obj, "transform") and not isinstance(obj, tuple)

  def test_pca_menerima_array_numpy(self):
    _, hasil = pca(np.array([[1.0, 2], [3, 4], [5, 6]]), n_components=1)
    assert hasil.shape == (3, 1)


class TestInitStorages:
  def test_membuat_folder_artefak(self, tmp_path, monkeypatch):
    from app.config import Config
    monkeypatch.setattr(Config, "dir", tmp_path)
    init_storages("Regression-baru")
    dasar = tmp_path / "storages" / "Regression-baru"
    assert (dasar / "top_model").is_dir() and (dasar / "results").is_dir()

  def test_tidak_lagi_membuat_folder_logs(self, tmp_path, monkeypatch):
    """Log pindah ke `Config.log_dir`, supaya `rm -rf storages/<ds>` tidak ikut
    menghapus jejak kejadiannya."""
    from app.config import Config
    monkeypatch.setattr(Config, "dir", tmp_path)
    init_storages("Regression-baru")
    assert not (tmp_path / "storages" / "Regression-baru" / "logs").exists()

  def test_boleh_dipanggil_dua_kali(self, tmp_path, monkeypatch):
    from app.config import Config
    monkeypatch.setattr(Config, "dir", tmp_path)
    init_storages("Regression-baru")
    init_storages("Regression-baru")                           # tidak melempar


class TestReadLogs:
  def test_dataset_tanpa_berkas_log_mengembalikan_daftar_kosong(self, tmp_path, monkeypatch):
    from app.config import Config
    from app.logger import LogManager
    monkeypatch.setattr(Config, "log_dir", tmp_path)
    LogManager.reset()
    assert read_logs("Regression-belum-ada") == []

  def test_membaca_dari_lokasi_yang_ditentukan_logmanager(self, tmp_path, monkeypatch):
    from app.config import Config, Verbose
    from app.logger import LogManager
    monkeypatch.setattr(Config, "log_dir", tmp_path)
    monkeypatch.setattr(Config, "verbose", Verbose.NORMAL)
    LogManager.reset()
    LogManager.get("Regression-a1b2").info("halo dari test")
    try:
      assert [l["message"] for l in read_logs("Regression-a1b2")] == ["halo dari test"]
    finally:
      LogManager.reset()


class TestParseLogLine:
  """Round-trip formatnya dikunci di test_log_pipeline; di sini kasus tepinya."""

  def test_baris_kosong_mengembalikan_dict_kosong(self):
    assert parse_log_line("   ") == {}

  def test_baris_tak_berformat_masuk_jalur_cadangan(self):
    hasil = parse_log_line("sekadar teks biasa")
    assert hasil["message"] == "sekadar teks biasa"
    assert hasil["level"] == "INFO" and hasil["line"] is None

  def test_level_warn_diseragamkan_menjadi_warning(self):
    baris = "[2026-07-01 10:00:00] [WARN] [ds] app.pull:pulling:1 - hati-hati"
    assert parse_log_line(baris)["level"] == "WARNING"

  def test_timestamp_aneh_dibiarkan_apa_adanya(self):
    baris = "[kemarin] [INFO] [ds] app.pull:pulling:1 - pesan"
    assert parse_log_line(baris)["timestamp"] == "kemarin"
