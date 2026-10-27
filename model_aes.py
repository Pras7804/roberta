"""Fungsi murni AES Bi-Encoder; mengikuti notebook pengguna TANPA regresi linear.

Input encoder: Jawaban_Siswa_clean dan Kunci_Guru_clean secara TERPISAH.
Pertanyaan hanya informasi bagi guru, bukan bagian dari input model.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
import numpy as np

MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 512
KOLOM_WAJIB = ("Kode_Soal", "Pertanyaan", "Jawaban_Siswa", "Kunci_Guru", "Bobot")
RENAMING = {
    "Kode Soal": "Kode_Soal", "Kunci Guru": "Kunci_Guru",
    "Jawaban Siswa": "Jawaban_Siswa", "Skor Guru": "Skor",
}


def muat_kamus(path: str | Path) -> dict[str, str]:
    """Pakai kamus yang DIEKSPOR dari notebook, jangan unduh versi lain."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Kamus dari Colab tidak ditemukan: {p}")
    kamus = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(kamus, dict) or not kamus:
        raise ValueError("kamus_colab.json harus berupa objek pasangan kata, tidak kosong.")
    return {str(k): str(v) for k, v in kamus.items()}


def pra_pemrosesan(text, kamus: dict[str, str]) -> str:
    """Sama dengan pra_pemrosesan di notebook Bi-Encoder tanpa regresi."""
    if text is None or (isinstance(text, (float, np.floating)) and np.isnan(text)):
        return ""
    text = str(text).lower()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(kamus.get(w, w) for w in text.split())


def siap_dataset(df, butuh_kode: bool = True):
    """Validasi dan bersihkan dataframe input. Tidak mengubah teks jawaban."""
    import pandas as pd

    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    df = df.rename(columns=RENAMING)
    if not butuh_kode and "Kode_Soal" not in df.columns:
        df["Kode_Soal"] = None
    kurang = [kolom for kolom in KOLOM_WAJIB if kolom not in df.columns]
    if kurang:
        raise ValueError("Kolom wajib tidak ada: " + ", ".join(kurang))
    df = df.dropna(how="all").reset_index(drop=True)
    for kol in ("Pertanyaan", "Jawaban_Siswa", "Kunci_Guru"):
        df[kol] = df[kol].fillna("").astype(str)
    df["Bobot"] = pd.to_numeric(
        df["Bobot"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )
    if df["Bobot"].isna().any() or df["Bobot"].le(0).any():
        raise ValueError("Bobot harus angka positif pada semua baris.")
    if df["Kode_Soal"].notna().any():
        df["Kode_Soal"] = pd.to_numeric(df["Kode_Soal"], errors="coerce").astype("Int64")
    if "Skor" in df.columns:
        df["Skor"] = pd.to_numeric(
            df["Skor"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )
    for kol in ("Nama", "Kelas", "Absen"):
        if kol not in df.columns:
            df[kol] = ""
        else:
            df[kol] = df[kol].fillna("")
    return df


def bank_dari_dataset(df) -> dict[int, dict]:
    """Ambil pertanyaan, kunci, bobot yang konsisten untuk tiap Kode_Soal."""
    bank = {}
    data = df.dropna(subset=["Kode_Soal"])
    for kode, group in data.groupby("Kode_Soal", sort=True):
        kode = int(kode)
        for col in ("Pertanyaan", "Kunci_Guru", "Bobot"):
            unik = group[col].dropna().unique()
            if len(unik) != 1:
                raise ValueError(f"Kode soal {kode}: {col} tidak konsisten, periksa Excel.")
        pertama = group.iloc[0]
        bank[kode] = {
            "kode": kode,
            "pertanyaan": str(pertama["Pertanyaan"]),
            "kunci": str(pertama["Kunci_Guru"]),
            "bobot": float(pertama["Bobot"]),
            "jumlah_jawaban": int(len(group)),
        }
    if not bank:
        raise ValueError("Dataset belum berisi Kode_Soal yang valid.")
    return bank


def cosine_ke_nilai(cosine: float, bobot: float) -> dict[str, float]:
    """Notebook asli: np.clip(cos,0,1); cos*bobot; np.round(skor*2)/2."""
    bobot = float(bobot)
    if not np.isfinite(bobot) or bobot <= 0:
        raise ValueError("Bobot harus positif dan finite.")
    cosine = float(np.clip(float(cosine), 0.0, 1.0))
    skor_kontinu = cosine * bobot
    skor_rubrik = float(np.maximum(0.0, np.round(skor_kontinu * 2.0) / 2.0))
    return {
        "cosine": cosine,
        "persentase": cosine * 100.0,
        "skor_kontinu": skor_kontinu,
        "skor_rubrik": skor_rubrik,
    }


def similarity_rows(mat_siswa: np.ndarray, mat_guru: np.ndarray) -> np.ndarray:
    """Menggunakan fungsi sklearn sama seperti notebook pengguna."""
    from sklearn.metrics.pairwise import cosine_similarity

    if mat_siswa.shape != mat_guru.shape or mat_siswa.ndim != 2:
        raise ValueError("Ukuran matriks siswa dan guru harus sama (N, 768).")
    nilai = [
        float(cosine_similarity(s.reshape(1, -1), g.reshape(1, -1))[0, 0])
        for s, g in zip(mat_siswa, mat_guru)
    ]
    return np.clip(np.array(nilai, dtype=float), 0.0, 1.0)
