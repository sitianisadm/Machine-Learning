# Predict People Personality Types
Project Machine Learning untuk melakukan prediksi tipe kepribadian seseorang berdasarkan karakteristik yang terdapat pada dataset **Predict People Personality Types**.
Project ini menggunakan algoritma **Random Forest Classification** dan **Logistic Regression** untuk melakukan klasifikasi tipe kepribadian berdasarkan data yang telah melalui proses cleaning, splitting, dan preprocessing.

## Tujuan
Tujuan dari project ini adalah membangun model Machine Learning yang dapat memprediksi tipe kepribadian seseorang berdasarkan data yang tersedia.
Model dilatih menggunakan data training dan kemudian diuji menggunakan data testing untuk mengetahui seberapa baik model dalam melakukan klasifikasi.

## Dataset
Dataset yang digunakan merupakan dataset **Predict People Personality Types** yang berisi data karakteristik seseorang yang digunakan sebagai fitur untuk memprediksi tipe kepribadian.
Data terdiri dari beberapa atribut, termasuk data kategorikal seperti:
- **Gender**
- **Interest**
serta atribut lain yang digunakan sebagai fitur dalam proses klasifikasi.

## Workflow
Workflow yang digunakan dalam project ini terdiri dari beberapa tahapan berikut:

### 1. Data Loading
Dataset dimuat menggunakan library **Pandas** untuk membaca dan menampilkan data.
Tahapan ini dilakukan untuk mengetahui struktur awal dataset, jumlah data, nama kolom, tipe data, serta informasi lainnya yang diperlukan sebelum masuk ke tahap berikutnya.

### 2. Data Cleaning
Pada tahap Data Cleaning dilakukan pemeriksaan terhadap kondisi dataset, seperti:
- Memeriksa struktur dan tipe data.
- Memeriksa missing value.
- Memeriksa data yang tidak sesuai atau bermasalah.
- Memastikan data siap digunakan untuk tahap berikutnya.
Berdasarkan hasil pemeriksaan, dataset yang digunakan tidak memiliki missing value sehingga tidak diperlukan proses pengisian atau penghapusan data berdasarkan missing value.

### 3. Data Splitting
Dataset dibagi menjadi data fitur (**X**) dan target (**y**).
Selanjutnya data dibagi menjadi:
- **X_train dan y_train** sebagai data training.
- **X_test dan y_test** sebagai data testing.
Pembagian data dilakukan agar model dilatih menggunakan data training dan performanya dapat diuji menggunakan data yang belum digunakan pada proses pelatihan.

### 4. Data Processing
Pada tahap ini dilakukan preprocessing terhadap fitur kategorikal agar dapat digunakan oleh algoritma Machine Learning.

#### Encoding Gender
Kolom **Gender** diubah dari data kategorikal menjadi data numerik menggunakan **One Hot Encoding**.
Encoding dilakukan menggunakan:
```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, dtype=int)
```
Hasil encoding kemudian digunakan sebagai fitur pengganti kolom Gender sebelumnya.

#### Encoding Interest
Kolom **Interest** juga diproses menggunakan **One Hot Encoding** karena memiliki beberapa kategori.
One Hot Encoding mengubah setiap kategori menjadi kolom baru dengan nilai 0 dan 1 sehingga data dapat diproses oleh model Machine Learning.

### 5. Modeling
Algoritma yang digunakan pada tahap modeling adalah **Random Forest Classification** dan **Logistic Regression**
Random Forest merupakan algoritma klasifikasi yang menggunakan beberapa Decision Tree dan menggabungkan hasil dari tree tersebut untuk menghasilkan prediksi. dan Logistic Regression algoritma Machine Learning yang digunakan untuk masalah klasifikasi, yaitu untuk memprediksi kategori atau kelas dari suatu data.~
Model dilatih menggunakan:

```text
X_train
y_train
```

Kemudian digunakan untuk melakukan prediksi terhadap:

```text
X_test
```

### 6. Prediction
Setelah model selesai dilatih, model digunakan untuk memprediksi tipe kepribadian pada data testing.
Hasil prediksi kemudian dibandingkan dengan nilai aktual pada **y_test** untuk mengetahui kemampuan model dalam melakukan klasifikasi.

### 7. Model Evaluation
Evaluasi dilakukan untuk mengetahui performa model Random Forest dalam melakukan klasifikasi.
Evaluasi dapat dilakukan menggunakan beberapa metrik, seperti:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Hasil evaluasi digunakan untuk mengetahui seberapa baik model dalam memprediksi tipe kepribadian.

## Algoritma
Algoritma Machine Learning yang digunakan:
**Random Forest Classification**
Random Forest dipilih karena dapat digunakan untuk permasalahan klasifikasi dan mampu menangani berbagai fitur dalam dataset.

## Preprocessing
Tahapan preprocessing yang digunakan dalam project:

```text
Data
 ↓
Data Cleaning
 ↓
Data Splitting
 ↓
Data Processing
 ↓
One Hot Encoding Gender
 ↓
One Hot Encoding Interest
 ↓
Random Forest & Logistic Regression
 ↓
Prediction
 ↓
Evaluation
```

## Library yang Digunakan
Project ini menggunakan beberapa library Python:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
```

## Tools
- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook / Google Colab
- Visual Studio Code

## Hasil
Hasil akhir dari project berupa model **Random Forest Classification** yang digunakan untuk memprediksi tipe kepribadian berdasarkan fitur pada dataset.
Performa model kemudian diketahui melalui hasil evaluasi menggunakan data testing dan metrik evaluasi klasifikasi.

## Kesimpulan
Project ini menerapkan tahapan Machine Learning mulai dari **Data Loading, Data Cleaning, Data Splitting, Data Processing, Modeling, Prediction, hingga Evaluation**.
Data kategorikal seperti **Gender** dan **Interest** diproses menggunakan **One Hot Encoding**, kemudian data yang telah diproses digunakan untuk membangun model **Random Forest Classification** dan **Logistic Regression** untuk melakukan prediksi tipe kepribadian.