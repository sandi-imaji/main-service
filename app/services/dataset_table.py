"""Tabel isi dataset: paginasi untuk ditampilkan, dan berkas untuk diunduh.

Sumber datanya bergantung task type — dan itu keputusan yang sengaja dipusatkan
di sini supaya apa yang TAMPIL di tabel selalu sama dengan apa yang TERUNDUH:

    Clustering  → `results/clusters.csv` (fitur + kolom label per algoritma),
                  dengan nama cluster pemberian user sudah dipetakan
    Lainnya     → `data.csv` hasil pulling

Kolom `dt` sengaja dibiarkan sebagai string apa adanya dari CSV. Mem-parse-nya
jadi datetime ber-timezone akan menggagalkan penulisan Excel — openpyxl menolak
datetime bertimezone — dan urutan ISO 8601 tetap benar meski dibandingkan
sebagai teks.
"""
from __future__ import annotations

import io
import json
import math
from typing import Optional, Tuple

import pandas as pd

from app.config import Config
from app.exceptions import NotFoundException, ValidationException
from app.utils.security import safe_path_join

MAX_PAGE_SIZE = 500
DEFAULT_PAGE_SIZE = 50


def resolve_source(dataset) -> Tuple[pd.DataFrame, str]:
  """Frame yang mewakili dataset ini, beserta nama sumbernya.

  Clustering memakai hasil cluster kalau sudah ada; kalau belum dilatih, jatuh
  ke `data.csv` supaya tabel tetap bisa dibuka — bukan halaman kosong.
  """
  storage = safe_path_join(Config.dir, "storages", dataset.name)

  if dataset.task_type.is_clustering():
    clusters = storage / "results" / "clusters.csv"
    if clusters.exists():
      # modul penamaan sengaja yang diimpor, bukan `services.clustering` —
      # yang terakhir menarik PyCaret (~2 detik) padahal tidak dibutuhkan
      from app.services import cluster_naming
      df = pd.read_csv(clusters)
      naming = cluster_naming.get_naming_clusters(dataset.name)
      if naming:
        df = cluster_naming.rename_clusters(df, naming)
      return df, "results/clusters.csv"

  data = storage / "data.csv"
  if not data.exists():
    raise NotFoundException("Dataset CSV", dataset.name)
  return pd.read_csv(data), "data.csv"


def _records(df: pd.DataFrame) -> list:
  """Baris sebagai list dict yang aman untuk JSON.

  Lewat `to_json` supaya NaN menjadi `null` — JSON tidak punya NaN, dan
  mengirimnya apa adanya menghasilkan payload yang gagal di-parse client.
  """
  return json.loads(df.to_json(orient="records", date_format="iso"))


def page(dataset, page: int = 1, page_size: int = DEFAULT_PAGE_SIZE,
         sort: Optional[str] = None, order: str = "asc") -> dict:
  """Satu halaman isi dataset.

  Membaca CSV utuh lalu memotong: pada 45.000 baris pembacaannya ~13 ms,
  sementara jumlah total baris ikut didapat gratis — tidak perlu trik offset
  yang menghitung baris dua kali.
  """
  df, source = resolve_source(dataset)

  if page_size < 1 or page_size > MAX_PAGE_SIZE:
    raise ValidationException(f"page_size must be between 1 and {MAX_PAGE_SIZE}", field="page_size")
  if sort:
    if sort not in df.columns:
      raise ValidationException(f"Column '{sort}' does not exist in {source}", field="sort")
    df = df.sort_values(sort, ascending=(order or "asc").lower() != "desc",
                        kind="stable")

  total = len(df)
  pages = max(1, math.ceil(total / page_size))
  page = max(1, min(page, pages))                 # jepit, jangan balas halaman kosong
  start = (page - 1) * page_size

  return {
      "source": source,
      "columns": list(df.columns),               # urutan kolom tidak dijamin oleh objek JSON
      "total": total,
      "page": page,
      "page_size": page_size,
      "pages": pages,
      "rows": _records(df.iloc[start:start + page_size]),
  }


def to_csv(dataset) -> Tuple[io.BytesIO, str]:
  """Seluruh isi tabel sebagai CSV. Untuk clustering, nama cluster sudah ikut
  terpetakan — jadi berkas unduhan sama dengan yang tampil di layar."""
  df, source = resolve_source(dataset)
  buffer = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
  return buffer, _filename(dataset, source, "csv")


def to_excel(dataset) -> Tuple[io.BytesIO, str]:
  """Seluruh isi tabel sebagai XLSX.

  Ini satu-satunya jalur yang mahal (±1 detik per 45.000 baris), jadi berkasnya
  dibangkitkan saat diminta — tidak disimpan di disk.
  """
  df, source = resolve_source(dataset)
  df = _excel_safe(df)

  buffer = io.BytesIO()
  with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="Data")
  buffer.seek(0)
  return buffer, _filename(dataset, source, "xlsx")


def _excel_safe(df: pd.DataFrame) -> pd.DataFrame:
  """Jadikan `dt` sel tanggal sungguhan di Excel.

  openpyxl menolak datetime yang membawa timezone, jadi offsetnya dibuang
  setelah konversi. Kalau kolomnya tidak bisa di-parse, dibiarkan sebagai teks —
  lebih baik teks daripada ekspor yang gagal.
  """
  if "dt" not in df.columns:
    return df
  try:
    parsed = pd.to_datetime(df["dt"], errors="raise", utc=True)
  except (ValueError, TypeError):
    return df
  out = df.copy()
  out["dt"] = parsed.dt.tz_localize(None)
  return out


def _filename(dataset, source: str, ext: str) -> str:
  jenis = "clusters" if source.endswith("clusters.csv") else "data"
  return f"{dataset.name}-{jenis}.{ext}"
