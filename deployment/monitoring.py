
import streamlit as st
import pandas as pd
import requests
import json
from pathlib import Path

st.set_page_config(
    page_title="Personality Monitoring",
    page_icon="📊",
    layout="wide"
)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "monitoring.csv"
API_URL = "http://127.0.0.1:5000/predict_batch"
MAX_BATCH_SIZE = 500

FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Education",
    "Introversion Score",
    "Sensing Score",
    "Thinking Score",
    "Judging Score",
    "Interest"
]

st.title("Personality Prediction Monitoring")
st.write(
    "Dashboard untuk mengirim data CSV ke API Flask, "
    "melihat hasil prediksi, dan memantau riwayat prediksi."
)

# ==================================================
# BAGIAN 1: UPLOAD CSV DAN PREDIKSI BATCH
# ==================================================

st.header("Prediksi Kepribadian dari CSV")

uploaded_file = st.file_uploader(
    "Upload file CSV",
    type=["csv"],
    help="CSV harus memiliki delapan kolom fitur yang dibutuhkan model."
)

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)

        # Menyesuaikan nama kolom jika CSV memakai
        # nama fitur tanpa kata 'Score'.
        rename_columns = {
            "Introversion": "Introversion Score",
            "Sensing": "Sensing Score",
            "Thinking": "Thinking Score",
            "Judging": "Judging Score"
        }

        input_df = input_df.rename(columns=rename_columns)

        # Kolom target tidak digunakan sebagai input prediksi.
        if "Personality" in input_df.columns:
            input_df = input_df.drop(columns=["Personality"])

        missing_columns = [
            col for col in FEATURE_COLUMNS
            if col not in input_df.columns
        ]

        if missing_columns:
            st.error(
                "Kolom berikut belum tersedia: "
                + ", ".join(missing_columns)
            )
            st.write("Kolom yang tersedia:", input_df.columns.tolist())

        elif input_df.empty:
            st.warning("File CSV tidak memiliki data.")

        elif input_df[FEATURE_COLUMNS].isnull().any().any():
            st.error(
                "Masih ada nilai kosong pada fitur yang dibutuhkan. "
                "Perbaiki CSV terlebih dahulu."
            )

        else:
            st.write(f"Jumlah data: **{len(input_df):,} baris**")
            st.dataframe(
                input_df[FEATURE_COLUMNS].head(5),
                use_container_width=True,
                hide_index=True
            )

            if st.button(
                "Mulai Prediksi",
                type="primary"
            ):
                all_results = []
                total_rows = len(input_df)
                total_batches = (
                    total_rows + MAX_BATCH_SIZE - 1
                ) // MAX_BATCH_SIZE

                progress = st.progress(0)
                status_text = st.empty()

                try:
                    for batch_number, start in enumerate(
                        range(0, total_rows, MAX_BATCH_SIZE),
                        start=1
                    ):
                        batch_df = input_df.iloc[
                            start:start + MAX_BATCH_SIZE
                        ]

                        # Mengubah NaN menjadi None agar JSON valid.
                        records = json.loads(
                            batch_df.to_json(
                                orient="records"
                            )
                        )

                        response = requests.post(
                            API_URL,
                            json={"data": records},
                            timeout=180
                        )

                        response.raise_for_status()
                        response_data = response.json()

                        results = response_data.get("results")

                        if not isinstance(results, list):
                            raise ValueError(
                                "Format respons API tidak sesuai."
                            )

                        if len(results) != len(batch_df):
                            raise ValueError(
                                "Jumlah hasil prediksi tidak sesuai "
                                "dengan jumlah data yang dikirim."
                            )

                        all_results.extend(results)

                        progress.progress(
                            batch_number / total_batches
                        )
                        status_text.write(
                            f"Memproses batch {batch_number} "
                            f"dari {total_batches}..."
                        )

                    # Gabungkan fitur input dengan hasil API.
                    result_df = input_df[
                        FEATURE_COLUMNS
                    ].reset_index(drop=True)

                    api_results = pd.DataFrame(all_results)

                    if "prediction" not in api_results.columns:
                        raise ValueError(
                            "Respons API tidak memiliki kolom "
                            "'prediction'. Periksa app.py."
                        )

                    result_df["Prediction"] = (
                        api_results["prediction"].values
                    )

                    if "probabilitas" in api_results.columns:
                        result_df["Probabilitas"] = (
                            api_results["probabilitas"].apply(
                                lambda value: (
                                    json.dumps(value, ensure_ascii=False)
                                    if isinstance(value, (dict, list))
                                    else value
                                )
                            ).values
                        )

                    if "status" in api_results.columns:
                        result_df["Status"] = (
                            api_results["status"].values
                        )

                    st.session_state["batch_results"] = result_df

                    progress.progress(1.0)
                    status_text.success(
                        f"Berhasil memprediksi {len(result_df):,} data."
                    )

                except requests.RequestException as e:
                    st.error(
                        "Gagal menghubungi API Flask. "
                        "Pastikan server Flask masih berjalan."
                    )
                    st.caption(str(e))

                except (ValueError, KeyError, TypeError) as e:
                    st.error(f"Proses prediksi gagal: {e}")

    except (pd.errors.ParserError, UnicodeDecodeError) as e:
        st.error(f"File CSV tidak dapat dibaca: {e}")

# Menampilkan hasil prediksi yang tersimpan selama sesi dashboard.
if "batch_results" in st.session_state:
    result_df = st.session_state["batch_results"]

    st.subheader("Hasil Prediksi Batch")
    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="Download Hasil Prediksi CSV",
        data=result_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="hasil_prediksi_batch.csv",
        mime="text/csv"
    )

st.divider()

# ==================================================
# BAGIAN 2: MONITORING RIWAYAT PREDIKSI DARI FLASK
# ==================================================

st.header("Riwayat Monitoring")

if st.button("Muat Ulang Data Monitoring"):
    st.rerun()

if not LOG_PATH.exists():
    st.info(
        "Belum ada file monitoring.csv. "
        "Kirim prediksi melalui API Flask terlebih dahulu."
    )
    st.stop()

try:
    df = pd.read_csv(LOG_PATH)
except (pd.errors.EmptyDataError, pd.errors.ParserError):
    st.info("File monitoring belum memiliki data yang valid.")
    st.stop()

if df.empty:
    st.info("Belum ada riwayat prediksi.")
    st.stop()

required_log_columns = [
    "waktu",
    "prediction",
    "probabilitas",
    "data_input",
    "status"
]

missing_log_columns = [
    col for col in required_log_columns
    if col not in df.columns
]

if missing_log_columns:
    st.error(
        "Kolom monitoring.csv tidak lengkap: "
        + ", ".join(missing_log_columns)
    )
    st.stop()

df["waktu"] = pd.to_datetime(
    df["waktu"],
    errors="coerce"
)

df = df.dropna(subset=["waktu"])
df = df.sort_values("waktu", ascending=False)

if df.empty:
    st.warning("Tidak ada waktu prediksi yang valid.")
    st.stop()

# Filter tanggal
tanggal = df["waktu"].dt.date

rentang = st.date_input(
    "Filter tanggal prediksi",
    value=(tanggal.min(), tanggal.max())
)

if isinstance(rentang, (tuple, list)) and len(rentang) == 2:
    awal, akhir = rentang

    df = df[
        (df["waktu"].dt.date >= awal)
        & (df["waktu"].dt.date <= akhir)
    ]

if df.empty:
    st.warning("Tidak ada data pada rentang tanggal tersebut.")
    st.stop()

# Ringkasan monitoring
total = len(df)
jumlah_tipe = df["prediction"].nunique()
prediksi_terakhir = df.iloc[0]["prediction"]

c1, c2, c3 = st.columns(3)

c1.metric("Total Prediksi", f"{total:,}")
c2.metric("Tipe Kepribadian Berbeda", jumlah_tipe)
c3.metric("Prediksi Terakhir", prediksi_terakhir)

st.divider()

# Grafik distribusi prediksi
st.subheader("Distribusi Tipe Kepribadian")

distribusi = (
    df["prediction"]
    .value_counts()
    .rename_axis("Tipe Kepribadian")
    .reset_index(name="Jumlah")
)

st.bar_chart(
    distribusi,
    x="Tipe Kepribadian",
    y="Jumlah"
)

# Grafik aktivitas per hari
st.subheader("Aktivitas Prediksi per Hari")

aktivitas = (
    df.assign(tanggal=df["waktu"].dt.date)
    .groupby("tanggal")
    .size()
    .rename("Jumlah Prediksi")
)

st.line_chart(aktivitas)

# Tabel riwayat
st.subheader("Riwayat Prediksi")

st.dataframe(
    df[required_log_columns],
    use_container_width=True,
    hide_index=True
)

# Detail probabilitas
st.subheader("Detail Probabilitas")

pilihan = st.selectbox(
    "Pilih hasil prediksi",
    options=df.index,
    format_func=lambda i: (
        f"{df.loc[i, 'waktu']} — "
        f"{df.loc[i, 'prediction']}"
    )
)

try:
    probabilities = json.loads(
        df.loc[pilihan, "probabilitas"]
    )

    if probabilities:
        prob_df = pd.DataFrame({
            "Tipe Kepribadian": list(probabilities.keys()),
            "Probabilitas": list(probabilities.values())
        })

        st.bar_chart(
            prob_df,
            x="Tipe Kepribadian",
            y="Probabilitas"
        )
    else:
        st.info("Probabilitas tidak tersedia.")

except (ValueError, TypeError, KeyError):
    st.info("Data probabilitas tidak dapat dibaca.")

st.caption(
    "Dashboard ini memantau riwayat prediksi, "
    "bukan akurasi model terhadap label aktual."
)
