
import pandas as pd
import requests

df = pd.read_csv("data.csv")
url = "http://127.0.0.1:5000/predict_batch"

ukuran_batch = 500
semua_hasil = []

for mulai in range(0, len(df), ukuran_batch):
    batch = df.iloc[mulai:mulai + ukuran_batch]

    print(
        f"Mengirim data {mulai + 1}"
        f" sampai {mulai + len(batch)}..."
    )

    response = requests.post(
        url,
        json={"data": batch.to_dict(orient="records")},
        timeout=120
    )

    if response.status_code != 200:
        print("Terjadi error:", response.status_code)
        print(response.text)
        break

    hasil = response.json()
    semua_hasil.extend(hasil["results"])

    print("Berhasil:", len(hasil["results"]), "data")

if len(semua_hasil) == len(df):
    hasil_df = pd.DataFrame(semua_hasil)
    hasil_df.to_csv(
        "hasil_prediksi_batch.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSemua data berhasil diprediksi!")
    print("Hasil tersimpan di hasil_prediksi_batch.csv")
    print("Total prediksi:", len(hasil_df))
