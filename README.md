# Predict People Personality Types

Project Machine Learning untuk memprediksi tipe kepribadian seseorang berdasarkan karakteristik yang terdapat pada dataset **Predict People Personality Types**.

Project ini menggunakan algoritma **Random Forest Classification** dan **Logistic Regression** untuk melakukan klasifikasi tipe kepribadian. Proses pengembangan model meliputi data loading, data cleaning, data splitting, preprocessing, modeling, prediction, evaluation, penyimpanan model menggunakan Joblib, serta deployment model melalui API berbasis Flask.

## Tujuan

Tujuan project ini adalah membangun model Machine Learning yang dapat memprediksi tipe kepribadian seseorang berdasarkan data yang tersedia. Model dilatih menggunakan data training dan diuji menggunakan data testing untuk mengetahui performanya dalam melakukan klasifikasi.

Selain membangun model, project ini juga menerapkan penyimpanan model menggunakan Joblib agar model yang telah dilatih dapat digunakan kembali tanpa perlu melakukan training setiap kali aplikasi dijalankan. Model kemudian diintegrasikan ke dalam API Flask agar dapat menerima input dan mengembalikan hasil prediksi.

## Dataset

Dataset yang digunakan adalah **Predict People Personality Types**, yang berisi karakteristik seseorang sebagai fitur untuk memprediksi tipe kepribadian.

Beberapa atribut yang digunakan antara lain:

- **Gender**: atribut kategorikal yang menunjukkan jenis kelamin.
- **Interest**: atribut kategorikal yang menunjukkan minat seseorang.
- **Atribut lainnya**: fitur tambahan yang digunakan dalam proses klasifikasi.
- **Personality Type**: target yang akan diprediksi oleh model.

## Workflow Machine Learning

### Data Loading

Dataset dimuat menggunakan library Pandas untuk membaca dan menampilkan data. Tahap ini dilakukan untuk mengetahui struktur dataset, jumlah data, nama kolom, tipe data, serta informasi awal yang diperlukan sebelum proses pengolahan data.

### Data Cleaning

Tahap Data Cleaning dilakukan untuk memastikan dataset siap digunakan dalam proses Machine Learning. Pemeriksaan meliputi:

- Memeriksa struktur dan tipe data.
- Memeriksa missing value.
- Memeriksa data duplikat dan data yang tidak sesuai.
- Memastikan fitur dan target tersedia sesuai kebutuhan model.

Berdasarkan pemeriksaan pada dataset yang digunakan, tidak ditemukan missing value sehingga tidak diperlukan proses pengisian atau penghapusan data berdasarkan missing value.

### Data Splitting

Dataset dibagi menjadi data fitur (**X**) dan data target (**y**). Selanjutnya, data dibagi menjadi data training dan data testing menggunakan `train_test_split`.

- **X_train dan y_train** digunakan untuk melatih model.
- **X_test dan y_test** digunakan untuk menguji model menggunakan data yang tidak digunakan dalam proses pelatihan.

Pembagian data dilakukan sebelum proses preprocessing yang mempelajari parameter dari data, seperti encoding dan scaling, untuk menghindari data leakage. Preprocessing kemudian dipelajari dari data training dan diterapkan pada data testing melalui pipeline.

### Data Processing

Tahap Data Processing dilakukan untuk mengubah fitur agar dapat digunakan oleh algoritma Machine Learning.

#### One Hot Encoding Gender

Kolom `Gender` diproses menggunakan One Hot Encoding untuk mengubah setiap kategori menjadi kolom numerik dengan nilai 0 dan 1.

Contoh inisialisasi encoder:

```python
from sklearn.preprocessing import OneHotEncoder
```

#### Encoding Interest
Kolom **Interest** juga diproses menggunakan **One Hot Encoding** karena memiliki beberapa kategori.
One Hot Encoding mengubah setiap kategori menjadi kolom baru dengan nilai 0 dan 1 sehingga data dapat diproses oleh model Machine Learning.

#### Preprocessing Pipeline

Pipeline dan `ColumnTransformer` digunakan untuk menggabungkan preprocessing dengan model. Pendekatan ini membantu menjaga agar tahapan preprocessing yang digunakan saat training tetap sama ketika model digunakan untuk prediksi.

### Modeling

Project ini menggunakan dua algoritma klasifikasi.

**Random Forest Classification**

Random Forest merupakan algoritma yang menggabungkan hasil dari sejumlah Decision Tree untuk menghasilkan prediksi. Algoritma ini dapat digunakan untuk mempelajari pola hubungan antara fitur dan target kepribadian.

**Logistic Regression**

Logistic Regression merupakan algoritma Machine Learning yang dapat digunakan untuk klasifikasi. Model mempelajari hubungan antara fitur dan kelas target untuk menentukan kelas prediksi.

Kedua model dilatih menggunakan `X_train` dan `y_train`, kemudian dievaluasi menggunakan data testing. Hasil evaluasi dapat digunakan untuk membandingkan performa kedua algoritma.

### Prediction

Setelah proses training selesai, model digunakan untuk memprediksi tipe kepribadian pada data testing. Hasil prediksi dibandingkan dengan label aktual pada `y_test` untuk mengetahui kemampuan model dalam melakukan klasifikasi.

### Model Evaluation

Evaluasi dilakukan untuk mengetahui performa model menggunakan data testing. Metrik evaluasi yang digunakan meliputi:

- **Accuracy**: persentase prediksi yang benar dari seluruh data pengujian.
- **Precision**: ketepatan model ketika memprediksi suatu kelas.
- **Recall**: kemampuan model menemukan data yang termasuk dalam suatu kelas.
- **F1-Score**: rata-rata harmonis antara precision dan recall.
- **Confusion Matrix**: tabel untuk melihat perbandingan antara kelas aktual dan kelas prediksi.

Hasil evaluasi Random Forest dan Logistic Regression dapat digunakan untuk menentukan model yang akan dipilih untuk deployment. Pemilihan model dilakukan berdasarkan hasil pengujian, bukan hanya berdasarkan jenis algoritmanya.

## Model Persistence Menggunakan Joblib

Joblib digunakan untuk menyimpan model Machine Learning yang telah dilatih ke dalam file agar dapat dimuat kembali tanpa perlu melakukan training ulang setiap kali aplikasi dijalankan.

Dalam project ini, model disimpan dalam format `.pkl`. Pipeline yang disimpan sebaiknya mencakup preprocessing dan estimator sehingga input baru diproses dengan cara yang sama seperti ketika model dilatih.

### Menyimpan model

Contoh penyimpanan model Random Forest setelah proses training selesai:

```python
import joblib

# Menyimpan pipeline Random Forest yang telah dilatih
joblib.dump(rf_model, "model_personality.pkl")
```

Jika model yang dipilih adalah Logistic Regression, objek yang disimpan dapat diganti menjadi `lr_model`.

### Memuat model

Model yang sudah disimpan dapat dimuat kembali menggunakan:

```python
import joblib

# Memuat model yang telah disimpan
model = joblib.load("model_personality.pkl")
```

Setelah dimuat, model dapat digunakan untuk melakukan prediksi tanpa perlu menjalankan training kembali.

**Catatan:** objek yang disimpan harus sudah dilatih. File model juga sebaiknya dimuat menggunakan versi library yang kompatibel dengan versi ketika model disimpan. File Joblib hanya boleh dimuat dari sumber yang tepercaya karena proses pemuatan dapat mengeksekusi kode.

## Deployment Model Menggunakan Flask

Deployment merupakan proses mengintegrasikan model Machine Learning ke dalam aplikasi agar dapat digunakan untuk menerima input dan memberikan hasil prediksi.

Pada project ini, Flask digunakan untuk menyediakan API yang menghubungkan model Machine Learning dengan aplikasi atau client yang mengirimkan data.

### Alur deployment

1. Model yang telah dilatih disimpan menggunakan Joblib.
2. Aplikasi Flask memuat file model yang telah disimpan.
3. Client mengirimkan input dalam format JSON ke endpoint API.
4. Flask memproses input dan meneruskannya ke model.
5. Model menghasilkan prediksi tipe kepribadian.
6. Flask mengembalikan hasil prediksi dalam format JSON.

### Menjalankan aplikasi Flask secara lokal

Pastikan virtual environment sudah aktif dan seluruh dependensi telah terpasang. Kemudian jalankan:

```bash
python app.py
```

Jika aplikasi berhasil dijalankan, Flask akan menampilkan alamat lokal, misalnya:

```text
http://127.0.0.1:5000
```

Alamat tersebut digunakan untuk pengujian API pada komputer yang sama.

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
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Flask
- Jupyter Notebook
- Visual Studio Code
- Git dan GitHub

## Hasil Project

Project ini menghasilkan pipeline Machine Learning untuk melakukan klasifikasi tipe kepribadian menggunakan Random Forest dan Logistic Regression.

Model dievaluasi menggunakan data testing dengan metrik evaluasi klasifikasi. Model yang dipilih dapat disimpan menggunakan Joblib, kemudian dimuat oleh aplikasi Flask untuk menyediakan layanan prediksi melalui API.

Pengujian API dilakukan untuk memastikan proses pengiriman input dan pengembalian hasil prediksi berjalan sesuai implementasi. Monitoring dapat diterapkan untuk memantau ketersediaan API, waktu respons, error, dan performa model apabila tersedia data berlabel terbaru.

## Kesimpulan

Project Predict People Personality Types menerapkan tahapan Machine Learning mulai dari data loading, data cleaning, data splitting, data processing, modeling, prediction, hingga evaluation. Fitur kategorikal seperti Gender dan Interest diproses menggunakan One Hot Encoding agar dapat digunakan dalam proses klasifikasi.

Algoritma Random Forest Classification dan Logistic Regression digunakan untuk membangun serta membandingkan model prediksi. Joblib digunakan untuk menyimpan model yang telah dilatih, sedangkan Flask digunakan untuk mengintegrasikan model ke dalam API agar dapat menerima input dan memberikan hasil
