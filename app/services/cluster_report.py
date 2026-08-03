"""Ringkasan hasil clustering — perhitungan murni di atas DataFrame.

Tidak tahu apa-apa soal Dataset, storage, atau HTTP: masuk DataFrame + nama
kolom, keluar dict. Semua metrik di sini dihitung TANPA mengetahui arti fitur,
jadi berlaku untuk dataset apa pun.

Kenapa `episodes` dan bukan deret per-titik: label cluster pada data deret waktu
hampir selalu berulang beruntun. 2.665 baris pada dataset uji menyusut jadi 47
episode — bentuk yang sama persis informasinya, tapi cukup untuk menggambar
strip timeline langsung tanpa mengirim satu pun titik mentah.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Cluster di bawah ambang ini lebih tepat dibaca sebagai pencilan daripada
# kelompok — ditandai supaya UI tidak menampilkannya setara cluster besar.
MICRO_SHARE = 0.01

# Dipakai kalau interval tidak bisa diturunkan dari data (mis. hanya 1 baris).
DEFAULT_INTERVAL_MINUTES = 1.0


def infer_interval_minutes(dt: pd.Series) -> float:
  """Jarak antar baris, dari median selisih — tahan terhadap lubang data."""
  if len(dt) < 2:
    return DEFAULT_INTERVAL_MINUTES
  delta = pd.to_datetime(dt).sort_values().diff().median()
  if pd.isna(delta) or delta.total_seconds() <= 0:
    return DEFAULT_INTERVAL_MINUTES
  return delta.total_seconds() / 60.0


def constant_features(df: pd.DataFrame, features: List[str]) -> List[str]:
  """Kolom yang nilainya tidak pernah berubah.

  Bukan sekadar info: fitur seperti ini tidak membawa apa pun, dan kalau ikut
  dinormalisasi akan membuat pembagian dengan nol.
  """
  return [c for c in features if c in df.columns and df[c].nunique(dropna=True) <= 1]


def z_scores(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
  """Fitur dinormalisasi, kolom konstan dibuang.

  Wajib sebelum menghitung jarak apa pun: fitur di sini bisa bersatuan °C,
  persen, atau liter/menit sekaligus, jadi tanpa penyeragaman skala yang
  rentangnya paling lebar akan mendominasi jarak. Kolom konstan dibuang karena
  standar deviasinya nol — kalau ikut, seluruh jarak jadi NaN.
  """
  usable = [f for f in features
            if f in df.columns and df[f].nunique(dropna=True) > 1]
  if not usable:
    return pd.DataFrame(index=df.index)
  sub = df[usable].astype(float)
  # NaN sisa diisi 0 SETELAH normalisasi = "sama dengan rata-rata", jadi nilai
  # yang hilang tidak menarik jarak ke arah mana pun.
  return ((sub - sub.mean()) / sub.std()).fillna(0.0)


def cluster_geometry(df: pd.DataFrame, labels: pd.Series, features: List[str],
                     top_k: int = 5, z: Optional[pd.DataFrame] = None) -> dict:
  """Bentuk tiap cluster di ruang fitur: apa yang membedakannya, seberapa
  rapat, dan seberapa terpisah satu sama lain.

  `z` boleh dioper kalau sudah dihitung di tempat lain — normalisasinya sama
  untuk semua algoritma, jadi tidak perlu diulang per algoritma.
  """
  z = z_scores(df, features) if z is None else z
  empty = {"distinctive": {}, "compactness": {}, "separation": []}
  if z.empty or z.shape[1] == 0 or labels.empty:
    return empty

  labels = labels.astype(str)
  centroids = z.groupby(labels).mean()
  raw = df[z.columns]

  # --- Fitur pembeda: seberapa jauh centroid cluster dari rata-rata global,
  #     dalam satuan standar deviasi. Nilainya sudah dinormalisasi, jadi bisa
  #     dibandingkan antar fitur yang satuannya berbeda-beda.
  distinctive: Dict[str, List[dict]] = {}
  for label in centroids.index:
    ranked = centroids.loc[label].abs().sort_values(ascending=False).head(top_k)
    rows = raw[labels == label]
    distinctive[label] = [{
        "feature": feature,
        "z": round(float(centroids.loc[label, feature]), 3),
        "direction": "above" if centroids.loc[label, feature] > 0 else "below",
        "mean": round(float(rows[feature].mean()), 4),
        "global_mean": round(float(raw[feature].mean()), 4),
        "p10": round(float(rows[feature].quantile(0.10)), 4),
        "p90": round(float(rows[feature].quantile(0.90)), 4),
    } for feature in ranked.index]

  # --- Kekompakan: sebaran anggota terhadap centroid-nya sendiri. Radius besar
  #     berarti "cluster" itu sebenarnya kumpulan titik yang berjauhan.
  compactness: Dict[str, dict] = {}
  radii: Dict[str, float] = {}
  for label in centroids.index:
    d = np.linalg.norm(z[labels == label].to_numpy() - centroids.loc[label].to_numpy(), axis=1)
    radii[label] = float(d.mean()) if len(d) else 0.0
    compactness[label] = {
        "mean_radius": round(radii[label], 3),
        "p95_radius": round(float(np.percentile(d, 95)), 3) if len(d) else 0.0,
    }

  # --- Keterpisahan antar pasangan cluster. `separation` tidak bersatuan:
  #     jarak antar centroid dibagi rata-rata radius keduanya. Di bawah 1 berarti
  #     kedua cluster saling tumpang tindih — pemisahannya lebih nama daripada
  #     kenyataan.
  separation: List[dict] = []
  order = list(centroids.index)
  for i, a in enumerate(order):
    for b in order[i + 1:]:
      distance = float(np.linalg.norm(centroids.loc[a].to_numpy() - centroids.loc[b].to_numpy()))
      spread = (radii[a] + radii[b]) / 2 or np.nan
      ratio = distance / spread if spread and not np.isnan(spread) else None
      separation.append({
          "between": [a, b],
          "distance": round(distance, 3),
          "separation": round(float(ratio), 3) if ratio is not None else None,
          "overlapping": bool(ratio is not None and ratio < 1),
      })

  return {"distinctive": distinctive, "compactness": compactness,
          "separation": sorted(separation, key=lambda s: s["distance"])}


def outliers(df: pd.DataFrame, labels: pd.Series, features: List[str], dt: pd.Series,
             top_k: int = 10, z: Optional[pd.DataFrame] = None) -> List[dict]:
  """Titik yang paling jauh dari centroid cluster-nya sendiri.

  Label cluster tidak pernah memunculkan ini: sebuah baris bisa dilabeli cluster
  "normal" tapi duduk jauh sekali dari pusatnya. Rasio terhadap radius rata-rata
  cluster membuatnya bisa dibandingkan antar cluster yang sebarannya berbeda.
  """
  z = z_scores(df, features) if z is None else z
  if z.empty or z.shape[1] == 0 or labels.empty:
    return []

  labels = labels.astype(str)
  centroids = z.groupby(labels).mean()
  dt = pd.to_datetime(dt)

  # Jarak dihitung vektor per cluster, lalu HANYA top_k yang dirinci. Merinci
  # tiap baris (2.665 kali `nlargest`) membuat laporan ini 27× lebih lambat
  # padahal 2.655 hasilnya langsung dibuang.
  distance = pd.Series(0.0, index=z.index)
  radius: Dict[str, float] = {}
  offsets: Dict[str, pd.DataFrame] = {}
  for label in centroids.index:
    member_z = z[labels == label]
    offset = member_z - centroids.loc[label]
    offsets[label] = offset
    d = pd.Series(np.linalg.norm(offset.to_numpy(), axis=1), index=member_z.index)
    distance.loc[member_z.index] = d
    radius[label] = float(d.mean()) if len(d) else 0.0

  rows: List[dict] = []
  for idx in distance.nlargest(top_k).index:
    label = labels.loc[idx]
    # fitur mana yang paling menyumbang jaraknya — supaya "aneh"-nya bisa dibaca
    drivers = offsets[label].loc[idx].abs().nlargest(3)
    rows.append({
        "dt": dt.loc[idx].isoformat(),
        "label": label,
        "distance": round(float(distance.loc[idx]), 3),
        "cluster_radius": round(radius[label], 3),
        "ratio": round(float(distance.loc[idx] / radius[label]), 2) if radius[label] else None,
        "drivers": [{"feature": f, "value": round(float(df.loc[idx, f]), 4)}
                    for f in drivers.index],
    })
  return rows


def agreement(labels_a: pd.Series, labels_b: pd.Series) -> Optional[dict]:
  """Seberapa sepakat dua algoritma memotong data yang sama.

  ARI kebal terhadap penomoran: dua algoritma yang mengelompokkan identik tapi
  memberi nomor berbeda tetap bernilai 1. Mendekati 0 berarti keduanya bercerita
  hal yang berbeda — dan user berhak tahu itu sebelum menyimpulkan dari salah
  satunya.
  """
  try:
    from sklearn.metrics import adjusted_rand_score
  except ImportError:                       # sklearn tidak wajib untuk modul ini
    return None
  if labels_a.empty or labels_b.empty:
    return None
  score = float(adjusted_rand_score(labels_a.astype(str), labels_b.astype(str)))
  return {"ari": round(score, 4), "verdict": _agreement_verdict(score)}


def _agreement_verdict(score: float) -> str:
  if score >= 0.75: return "strong"
  if score >= 0.4: return "partial"
  return "weak"


def hourly_distribution(dt: pd.Series, labels: pd.Series) -> List[dict]:
  """Proporsi tiap cluster per jam (0–23).

  Untuk data deret waktu, "kapan" biasanya pertanyaan pertama. Pola harian
  muncul di sini tanpa perlu menggambar satu titik pun.
  """
  if labels.empty:
    return []
  frame = pd.DataFrame({"hour": pd.to_datetime(dt).dt.hour, "label": labels.astype(str)})
  share = pd.crosstab(frame["hour"], frame["label"], normalize="index")
  counts = frame.groupby("hour").size()
  return [{
      "hour": int(hour),
      "total": int(counts[hour]),
      "shares": {str(label): round(float(share.loc[hour, label]), 4) for label in share.columns},
  } for hour in share.index]


def transitions(labels: pd.Series) -> List[dict]:
  """Perpindahan antar cluster berturut-turut — seberapa sering dan ke mana.

  Hanya perpindahan yang dicatat (bukan bertahan di cluster yang sama), karena
  yang menarik justru alur: dari mana ke mana sistem berpindah.
  """
  if len(labels) < 2:
    return []
  labels = labels.astype(str).reset_index(drop=True)
  moved = labels != labels.shift()
  pairs = pd.DataFrame({"from": labels.shift(), "to": labels})[moved].dropna()
  if pairs.empty:
    return []
  counts = pairs.groupby(["from", "to"]).size().reset_index(name="count")
  return [{"from": r["from"], "to": r["to"], "count": int(r["count"])}
          for _, r in counts.sort_values("count", ascending=False).iterrows()]


def drift(dt: pd.Series, labels: pd.Series, parts: int = 2) -> List[dict]:
  """Komposisi cluster per potongan periode — apakah perilakunya bergeser.

  Rentang dibagi rata menurut WAKTU, bukan jumlah baris, supaya lubang data
  tidak membuat potongan mewakili durasi yang berbeda-beda.
  """
  if labels.empty or parts < 2:
    return []
  dt = pd.to_datetime(dt)
  start, end = dt.min(), dt.max()
  if start == end:
    return []

  edges = [start + (end - start) * i / parts for i in range(parts + 1)]
  out: List[dict] = []
  for i in range(parts):
    lo, hi = edges[i], edges[i + 1]
    mask = (dt >= lo) & (dt <= hi) if i == parts - 1 else (dt >= lo) & (dt < hi)
    chunk = labels[mask.to_numpy()].astype(str)
    if chunk.empty:
      continue
    share = chunk.value_counts(normalize=True)
    out.append({
        "start": lo.isoformat(), "end": hi.isoformat(), "total": int(len(chunk)),
        "shares": {str(k): round(float(v), 4) for k, v in share.items()},
    })
  return out


def composition(labels: pd.Series, naming: Optional[Dict[str, str]] = None) -> dict:
  """Sebaran cluster: jumlah, proporsi, dan seberapa seimbang."""
  labels = labels.dropna()
  total = len(labels)
  if total == 0:
    return {"total": 0, "effective_clusters": 0, "balance": 0.0, "clusters": []}

  counts = labels.value_counts()
  shares = counts / total

  # Entropi ternormalisasi: 1 = semua cluster sama besar, mendekati 0 = satu
  # cluster menelan hampir semuanya (tanda pemisahan yang gagal).
  balance = 0.0
  if len(counts) > 1:
    balance = float(-(shares * np.log(shares)).sum() / np.log(len(counts)))

  clusters = [{
      "label": str(label),
      "name": (naming or {}).get(str(label)),
      "count": int(counts[label]),
      "share": round(float(shares[label]), 6),
      "is_micro": bool(shares[label] < MICRO_SHARE),
  } for label in counts.index]

  return {
      "total": total,
      "effective_clusters": int(len(counts)),
      "balance": round(balance, 4),
      "clusters": clusters,
  }


def episodes(dt: pd.Series, labels: pd.Series, interval_minutes: float,
             naming: Optional[Dict[str, str]] = None) -> List[dict]:
  """Rentetan baris berurutan dengan cluster sama, jadi satu kejadian.

  "8 titik" dan "4 kejadian × 10 menit" adalah dua pesan yang sangat berbeda:
  yang pertama terbaca sebagai pencilan acak, yang kedua sebagai pola.
  """
  if len(labels) == 0:
    return []

  frame = pd.DataFrame({"dt": pd.to_datetime(dt), "label": labels.astype(str)})
  frame = frame.sort_values("dt").reset_index(drop=True)

  group = (frame["label"] != frame["label"].shift()).cumsum()
  out: List[dict] = []
  for _, chunk in frame.groupby(group, sort=True):
    start, end = chunk["dt"].iloc[0], chunk["dt"].iloc[-1]
    label = chunk["label"].iloc[0]
    # +interval: baris terakhir mewakili satu interval penuh, bukan satu titik
    # tanpa durasi — tanpa ini episode 1 baris tampak berdurasi 0 menit.
    duration = (end - start).total_seconds() / 60.0 + interval_minutes
    out.append({
        "label": label,
        "name": (naming or {}).get(label),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "duration_minutes": round(duration, 2),
        "points": int(len(chunk)),
    })
  return out


def episode_stats(eps: List[dict]) -> Dict[str, dict]:
  """Ringkasan episode per cluster: berapa kali, berapa lama, kapan terakhir."""
  stats: Dict[str, dict] = {}
  for label in {e["label"] for e in eps}:
    durations = [e["duration_minutes"] for e in eps if e["label"] == label]
    last = max(e["end"] for e in eps if e["label"] == label)
    stats[label] = {
        "episodes": len(durations),
        "median_minutes": round(float(np.median(durations)), 2),
        "max_minutes": round(float(max(durations)), 2),
        "total_minutes": round(float(sum(durations)), 2),
        "last_seen": last,
    }
  return stats


def build_report(df: pd.DataFrame, algorithms: List[str], features: List[str],
                 naming: Optional[Dict[str, Dict[str, str]]] = None,
                 dt_column: str = "dt") -> dict:
  """Laporan lengkap untuk satu rentang data yang sudah difilter."""
  naming = naming or {}
  algorithms = [a for a in algorithms if a in df.columns]

  if df.empty:
    return {
        "total": 0, "interval_minutes": None, "range": None,
        "features": {"total": len(features), "effective": 0, "constant": []},
        "composition": {a: composition(pd.Series(dtype=str)) for a in algorithms},
        "episodes": {a: [] for a in algorithms},
        "episode_stats": {a: {} for a in algorithms},
        "geometry": {a: {"distinctive": {}, "compactness": {}, "separation": []}
                     for a in algorithms},
        "outliers": {a: [] for a in algorithms},
        "hourly": {a: [] for a in algorithms},
        "transitions": {a: [] for a in algorithms},
        "drift": {a: [] for a in algorithms},
        "agreement": {},
    }

  df = df.sort_values(dt_column)
  dt = pd.to_datetime(df[dt_column])
  interval = infer_interval_minutes(dt)
  const = constant_features(df, features)

  # dihitung sekali per algoritma, dipakai dua kali (daftar + ringkasannya)
  eps = {a: episodes(dt, df[a], interval, naming.get(a)) for a in algorithms}
  # normalisasi fitur sama untuk semua algoritma → hitung sekali, pakai berulang
  z = z_scores(df, features)

  return {
      "total": int(len(df)),
      "interval_minutes": round(interval, 4),
      "range": {"start": dt.iloc[0].isoformat(), "end": dt.iloc[-1].isoformat()},
      "features": {
          "total": len(features),
          "effective": len(features) - len(const),
          "constant": const,
      },
      "composition": {a: composition(df[a], naming.get(a)) for a in algorithms},
      "episodes": eps,
      "episode_stats": {a: episode_stats(e) for a, e in eps.items()},
      "geometry": {a: cluster_geometry(df, df[a], features, z=z) for a in algorithms},
      "outliers": {a: outliers(df, df[a], features, dt, z=z) for a in algorithms},
      "hourly": {a: hourly_distribution(dt, df[a]) for a in algorithms},
      "transitions": {a: transitions(df[a]) for a in algorithms},
      "drift": {a: drift(dt, df[a]) for a in algorithms},
      # kesepakatan hanya bermakna kalau ada lebih dari satu algoritma
      "agreement": {f"{a}|{b}": agreement(df[a], df[b])
                    for i, a in enumerate(algorithms) for b in algorithms[i + 1:]},
  }
