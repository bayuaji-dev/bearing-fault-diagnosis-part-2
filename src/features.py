
"""Validasi fitur untuk aplikasi Bearing Fault Diagnosis.

Tidak ada rekayasa fitur tambahan di sini. Model training sudah menerima
12 fitur ringkasan getaran yang sama, sehingga aplikasi hanya memastikan
nama dan urutan kolom konsisten.
"""
import pandas as pd

FITUR_INPUT = [
    "posisi_sensor", "kecepatan_rpm", "beban_kN", "suhu_housing_C",
    "rms_g", "puncak_g", "kurtosis", "crest_factor", "energi_bpfo",
    "energi_bpfi", "energi_bsf", "frekuensi_dominan_hz"
]

FITUR_KATEGORIK = ["posisi_sensor"]
FITUR_NUMERIK = [c for c in FITUR_INPUT if c not in FITUR_KATEGORIK]

def siapkan_fitur(data):
    """Mengambil fitur yang diperlukan dan mengurutkannya sesuai training."""
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    elif not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    data = data.copy()
    hilang = [kolom for kolom in FITUR_INPUT if kolom not in data.columns]
    if hilang:
        raise ValueError(f"Kolom fitur belum tersedia: {hilang}")

    return data[FITUR_INPUT]
