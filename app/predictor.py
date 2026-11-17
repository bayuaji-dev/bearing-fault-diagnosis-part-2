
"""Memuat pipeline model dan menjalankan prediksi multiclass."""
import json
import time
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.features import FITUR_INPUT, siapkan_fitur

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "artifacts" / "model.joblib"
METADATA_PATH = ROOT_DIR / "artifacts" / "metadata.json"

@lru_cache(maxsize=1)
def muat_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {MODEL_PATH}. "
            "Letakkan model training di artifacts/model.joblib."
        )
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata tidak ditemukan: {METADATA_PATH}")

    # Hanya muat file joblib dari sumber yang dipercaya.
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata

def prediksi_baris(rows):
    mulai = time.perf_counter()
    model, metadata = muat_artifact()

    if isinstance(rows, dict):
        X = siapkan_fitur(rows)
    else:
        X = siapkan_fitur(pd.DataFrame(rows))

    hasil = model.predict(X)
    kelas = [str(k) for k in getattr(model, "classes_", metadata.get("kelas", []))]

    probabilitas = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if len(proba) > 0:
            probabilitas = [
                {kelas[i]: float(nilai) for i, nilai in enumerate(row)}
                for row in proba
            ]

    output = []
    for i, label in enumerate(hasil):
        output.append({
            "prediksi_kondisi": str(label),
            "probabilitas_per_kelas": probabilitas[i] if probabilitas else None
        })

    return {
        "hasil": output,
        "versi_model": metadata.get("versi_model", "tidak diketahui"),
        "latensi_ms": round((time.perf_counter() - mulai) * 1000, 2)
    }

def daftar_fitur():
    _, metadata = muat_artifact()
    return metadata.get("fitur_input", FITUR_INPUT)
