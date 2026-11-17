
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.predictor import prediksi_baris, muat_artifact
from src.features import FITUR_INPUT

st.set_page_config(page_title="Bearing Fault Diagnosis", page_icon="⚙️", layout="wide")
st.title("⚙️ Bearing Fault Diagnosis")
st.caption("Klasifikasi kondisi bearing berdasarkan fitur getaran.")

try:
    model, metadata = muat_artifact()
except Exception as exc:
    st.error(f"Model belum dapat dimuat: {exc}")
    st.info("Pastikan file artifacts/model.joblib tersedia dan berasal dari training yang tepercaya.")
    st.stop()

st.caption(f"Versi model: {metadata.get('versi_model', 'tidak diketahui')}")

tab_satu, tab_batch = st.tabs(["Satu data", "Banyak data (CSV)"])

with tab_satu:
    st.subheader("Prediksi satu kondisi bearing")
    with st.form("form_prediksi"):
        c1, c2 = st.columns(2)
        with c1:
            posisi_sensor = st.selectbox("Posisi sensor", ["drive_end", "fan_end"])
            kecepatan_rpm = st.number_input("Kecepatan (RPM)", value=1500.0, step=10.0)
            beban_kN = st.number_input("Beban (kN)", value=1.0, step=0.1)
            suhu_housing_C = st.number_input("Suhu housing (°C)", value=40.0, step=0.5)
            rms_g = st.number_input("RMS (g)", value=0.5, step=0.01, format="%.4f")
            puncak_g = st.number_input("Puncak (g)", value=1.0, step=0.01, format="%.4f")
        with c2:
            kurtosis = st.number_input("Kurtosis", value=3.0, step=0.1, format="%.4f")
            crest_factor = st.number_input("Crest factor", value=2.0, step=0.1, format="%.4f")
            energi_bpfo = st.number_input("Energi BPFO", value=0.1, step=0.01, format="%.5f")
            energi_bpfi = st.number_input("Energi BPFI", value=0.1, step=0.01, format="%.5f")
            energi_bsf = st.number_input("Energi BSF", value=0.1, step=0.01, format="%.5f")
            frekuensi_dominan_hz = st.number_input("Frekuensi dominan (Hz)", value=100.0, step=1.0)

        submit = st.form_submit_button("Prediksi kondisi")

    if submit:
        baris = {
            "posisi_sensor": posisi_sensor,
            "kecepatan_rpm": kecepatan_rpm,
            "beban_kN": beban_kN,
            "suhu_housing_C": suhu_housing_C,
            "rms_g": rms_g,
            "puncak_g": puncak_g,
            "kurtosis": kurtosis,
            "crest_factor": crest_factor,
            "energi_bpfo": energi_bpfo,
            "energi_bpfi": energi_bpfi,
            "energi_bsf": energi_bsf,
            "frekuensi_dominan_hz": frekuensi_dominan_hz
        }
        try:
            hasil = prediksi_baris(baris)
            satu = hasil["hasil"][0]
            st.success(f"Prediksi kondisi bearing: **{satu['prediksi_kondisi']}**")
            st.metric("Waktu prediksi", f"{hasil['latensi_ms']} ms")
            if satu["probabilitas_per_kelas"]:
                df_proba = pd.DataFrame(
                    [{"Kelas": k, "Probabilitas": v} for k, v in satu["probabilitas_per_kelas"].items()]
                ).sort_values("Probabilitas", ascending=False)
                st.subheader("Probabilitas setiap kelas")
                st.bar_chart(df_proba.set_index("Kelas"))
                st.dataframe(df_proba, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Prediksi gagal: {exc}")

with tab_batch:
    st.subheader("Prediksi dari file CSV")
    st.write("CSV harus memiliki 12 kolom fitur sesuai template. Kolom `id_rekaman` boleh disertakan, tetapi tidak digunakan model.")
    template = pd.DataFrame(columns=FITUR_INPUT)
    st.download_button(
        "Unduh template CSV",
        template.to_csv(index=False).encode("utf-8"),
        file_name="template_bearing.csv",
        mime="text/csv"
    )
    uploaded = st.file_uploader("Unggah CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            if "id_rekaman" in df.columns:
                ids = df["id_rekaman"].astype(str)
            else:
                ids = pd.Series([f"baris_{i+1}" for i in range(len(df))])
            hasil = prediksi_baris(df)
            hasil_df = pd.DataFrame(hasil["hasil"])
            hasil_df.insert(0, "id_rekaman", ids.values)
            st.success(f"Berhasil memproses {len(hasil_df)} baris.")
            st.dataframe(hasil_df, use_container_width=True)
            st.download_button(
                "Unduh hasil prediksi",
                hasil_df.to_csv(index=False).encode("utf-8"),
                file_name="hasil_prediksi_bearing.csv",
                mime="text/csv"
            )
        except Exception as exc:
            st.error(f"File tidak dapat diproses: {exc}")
