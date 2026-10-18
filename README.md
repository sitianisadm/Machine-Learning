# Predict People Personality Types

Project Machine Learning untuk memprediksi tipe kepribadian seseorang menggunakan algoritma **Random Forest Classification** dan **Logistic Regression**. Model terbaik disimpan menggunakan Joblib, di-deploy melalui Flask API, dan dilengkapi dashboard monitoring menggunakan Streamlit.

## Dataset
Dataset yang digunakan adalah **Predict People Personality Types**, yang berisi karakteristik seseorang sebagai fitur untuk memprediksi tipe kepribadian.

Dataset yang digunakan berasal dari Kaggle:

[Predict People Personality Types](https://www.kaggle.com/datasets/stealthtechnologies/predict-people-personality-types)

Fitur yang digunakan meliputi `Age`, `Gender`, `Education`, `Introversion Score`, `Sensing Score`, `Thinking Score`, `Judging Score`, dan `Interest`. Target klasifikasi adalah `Personality`.

## Workflow Machine Learning

1. **Data Loading:** membaca dataset menggunakan Pandas.
2. **Data Cleaning:** memeriksa struktur data, missing value, dan kesesuaian fitur.
3. **Data Splitting:** membagi data menjadi 80% training dan 20% testing dengan stratifikasi.
4. **Preprocessing:** menggunakan One Hot Encoding untuk fitur `Gender` dan `Interest`, serta Pipeline dan ColumnTransformer.
5. **Modeling:** melatih Random Forest dan Logistic Regression.
6. **Prediction:** memprediksi tipe kepribadian menggunakan data testing.
7. **Evaluation:** membandingkan performa kedua model.
8. **Model Persistence:** menyimpan model terpilih menggunakan Joblib.
9. **Deployment:** menyediakan API prediksi menggunakan Flask.
10. **Monitoring:** mencatat aktivitas prediksi ke CSV dan menampilkannya melalui dashboard Streamlit.

## Hasil Evaluasi Model
Evaluasi dilakukan menggunakan data testing dengan metrik Accuracy, Precision, Recall, dan F1-Score.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Random Forest | 0.9000 | 0.9010 | 0.9000 | 0.9000 |
| Logistic Regression | 0.8533 | 0.8552 | 0.8533 | 0.8533 |

Berdasarkan hasil evaluasi, **Random Forest dipilih sebagai model terbaik** dengan accuracy sebesar 90%.

## Model Persistence dan Deployment
Model Random Forest disimpan dalam file `model_personality.pkl` menggunakan Joblib agar dapat digunakan tanpa training ulang.
Flask API menyediakan dua endpoint:

- `POST /predict` — memprediksi satu data.
- `POST /predict_batch` — memprediksi banyak data, maksimal 500 baris per request.

```bash
python .\deployment\app.py
```

API lokal tersedia di `http://127.0.0.1:5000`.

## Pengujian Batch dan Monitoring

Pengujian batch dilakukan menggunakan `deployment/test_batch.py`, dengan hasil prediksi disimpan ke `hasil_prediksi_batch.csv`. Pengujian yang dilakukan berhasil memproses 43.744 data melalui API.

Monitoring menggunakan `deployment/monitoring.py` untuk membaca log prediksi pada `deployment/monitoring.csv`. Log mencakup waktu prediksi, hasil klasifikasi, probabilitas, data input, dan status pemrosesan.

Jalankan dashboard pada terminal terpisah:

```bash
streamlit run .\deployment\monitoring.py
```

File `monitoring.csv` dibuat otomatis ketika API berhasil mencatat prediksi.

### Endpoint prediksi

Endpoint yang digunakan dalam project ini adalah:

```text
POST /predict
```

Alamat lengkap untuk pengujian lokal:

```text
http://127.0.0.1:5000/predict
```

Client mengirimkan data menggunakan metode `POST` dengan format JSON. Nama kolom, tipe data, dan fitur yang dikirim harus sesuai dengan kebutuhan model serta implementasi fungsi `predict()` di `app.py`.

Respons API berisi hasil prediksi sesuai format JSON yang didefinisikan dalam aplikasi Flask.

### Pengujian API

API dapat diuji menggunakan Thunder Client pada Visual Studio Code atau tools API client lainnya.

Tahapan pengujian:

1. Jalankan aplikasi Flask.
2. Buat request baru dengan metode `POST`.
3. Masukkan URL endpoint `/predict`.
4. Pilih Body dengan format JSON.
5. Masukkan data sesuai fitur yang dibutuhkan model.
6. Kirim request dan periksa respons yang diterima.

Pengujian dilakukan untuk memastikan endpoint dapat menerima input, memproses data, dan mengembalikan hasil prediksi dengan benar.

## Monitoring Model dan API

Monitoring merupakan proses memantau kondisi aplikasi dan performa model setelah digunakan. Tujuannya adalah mengetahui apakah API berjalan dengan baik dan apakah model masih memberikan hasil yang layak digunakan.

## Library yang Digunakan

Beberapa library yang digunakan dalam project ini meliputi:

- **Pandas**: membaca dan mengolah dataset.
- **NumPy**: mendukung operasi numerik.
- **Matplotlib** dan **Seaborn**: membuat visualisasi data.
- **Scikit-learn**: preprocessing, training, prediction, dan evaluasi model.
- **Joblib**: menyimpan dan memuat model Machine Learning.
- **Flask**: menyediakan API untuk deployment model.
- **Requests**: dapat digunakan untuk mengirim request HTTP saat pengujian melalui script Python.

Library tambahan seperti `requests` digunakan jika diperlukan untuk pengujian API.

## Tools
- Python
- Pandas dan NumPy
- Scikit-learn
- Joblib
- Flask
- Streamlit
- Jupyter Notebook
- Visual Studio Code
- Git dan GitHub

## Kesimpulan

Project ini menerapkan tahapan Machine Learning mulai dari pengolahan data, pelatihan dan evaluasi model, penyimpanan model, deployment API, hingga monitoring aktivitas prediksi. Random Forest dipilih karena memberikan hasil evaluasi lebih baik dibandingkan Logistic Regression pada data testing yang digunakan.
