
# Bearing Fault Diagnosis

Proyek deployment model Machine Learning untuk klasifikasi kondisi bearing berdasarkan fitur getaran.

## Struktur proyek

```text
bearing_fault_diagnosis_project/
├── app/
│   ├── api/
│   │   └── main.py
│   ├── ui/
│   │   └── streamlit_app.py
│   └── predictor.py
├── artifacts/
│   ├── model.joblib
│   └── metadata.json
├── data/
│   └── template_bearing.csv
├── src/
│   └── features.py
├── tests/
├── requirements.txt
└── README.md
```

## 1. Persiapan

Gunakan Python yang kompatibel dengan environment saat model dilatih. Sebaiknya catat versi Python, scikit-learn, pandas, numpy, dan joblib dari Google Colab training.

```bash
python -m venv .venv
```

Aktifkan environment:
- Windows: `.venv\\Scripts\\activate`
- macOS/Linux: `source .venv/bin/activate`

Instal dependensi:

```bash
pip install -r requirements.txt
```

## 2. Letakkan model

File model hasil training harus berada di:

`artifacts/model.joblib`

File yang digunakan pada proyek ini berasal dari `bearing_fault_diagnosis_model.joblib` yang disediakan bersama proyek. File joblib/pickle hanya boleh dimuat jika berasal dari sumber yang dipercaya.

## 3. Jalankan Streamlit

Dari folder utama proyek:

```bash
streamlit run app/ui/streamlit_app.py
```

Streamlit menyediakan:
- Prediksi satu data melalui formulir fitur bearing.
- Prediksi banyak data melalui unggahan CSV.
- Tampilan label prediksi dan probabilitas per kelas (jika model mendukung `predict_proba`).

## 4. Jalankan FastAPI (opsional)

```bash
uvicorn app.api.main:app --reload
```

Dokumentasi API tersedia di `http://127.0.0.1:8000/docs`.

Endpoint:
- `GET /health`
- `POST /predict`
- `POST /predict/batch`

Contoh body `/predict`:

```json
{
  "data": {
    "posisi_sensor": "drive_end",
    "kecepatan_rpm": 1500,
    "beban_kN": 1.0,
    "suhu_housing_C": 40,
    "rms_g": 0.5,
    "puncak_g": 1.0,
    "kurtosis": 3.0,
    "crest_factor": 2.0,
    "energi_bpfo": 0.1,
    "energi_bpfi": 0.1,
    "energi_bsf": 0.1,
    "frekuensi_dominan_hz": 100
  }
}
```

## Catatan kesesuaian model

Model harus berupa pipeline klasifikasi yang dilatih menggunakan fitur berikut dengan nama kolom yang sama:
`posisi_sensor`, `kecepatan_rpm`, `beban_kN`, `suhu_housing_C`, `rms_g`, `puncak_g`, `kurtosis`, `crest_factor`, `energi_bpfo`, `energi_bpfi`, `energi_bsf`, `frekuensi_dominan_hz`.

`id_rekaman` tidak digunakan sebagai fitur prediksi. Aplikasi tidak melakukan rekayasa fitur tambahan karena fitur yang digunakan harus sama dengan pipeline saat training.

Kelas yang ditulis di metadata adalah asumsi label hasil standardisasi: `normal`, `outer_race`, `inner_race`, `bola`. Prediksi aktual mengikuti `model.classes_` dari artefak model.

Jika model gagal dimuat, periksa apakah file berasal dari training pipeline yang benar dan apakah versi library kompatibel. Jangan mengganti model CNC dari repositori referensi dengan model bearing tanpa menyesuaikan struktur input dan output.
