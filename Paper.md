**Nama 			    : Siti Anisa Damayanti**

**NIM			    : 241011402712**

**Kelas			    : 05TPLP020**

**Mata Kuliah		: Machine Learning**

**Dosen Pengampu	: Asep Erlan Maulana, S.Kom., M.Kom**

**MAE (Mean Absolute Error) & MSE (Mean Squared Error)**
MAE (Mean Absolute Error) dan MSE (Mean Squared Error) merupakan metrik untuk mengukur besarnya kesalahan prediksi terhadap nilai aktual. Oleh karena itu, nilai MAE dan MSE yang semakin mendekati nol menunjukkan kesalahan prediksi yang semakin kecil. Nilai idealnya adalah nol, yang berarti tidak terdapat kesalahan prediksi pada observasi yang dievaluasi.

Pendapat ini didukung oleh Syahreza, Ningrum, dan Syahrazy (2024) dalam Edumatic: Jurnal Pendidikan Informatika, yang secara eksplisit menjelaskan bahwa MAE dan MSE yang mendekati nol menunjukkan performa model yang semakin baik. Penelitian tersebut juga melaporkan bahwa XGBoost menjadi model terbaik di antara Random Forest dan SVR, dengan MAE sebesar 0,3744, MSE sebesar 0,2278, dan R² sebesar 0,8183.

**R² (Coefficient of Determination)**
R² (Coefficient of Determination) mempunyai arah interpretasi yang berbeda dari MAE dan MSE. R² mengukur proporsi variasi nilai aktual yang dapat dijelaskan oleh model. Dengan demikian, nilai R² yang semakin mendekati satu umumnya menunjukkan kemampuan model yang semakin baik dalam menjelaskan variasi data.

Hal tersebut juga didukung oleh Khabani, Aesyi, dan Himawan (2024) dalam JSI: Jurnal Sistem Informasi. Penelitian mereka menyatakan bahwa semakin tinggi nilai R-Squared, semakin tinggi pula tingkat akurasi model. Pada penelitian prediksi tingkat pengangguran terbuka di Provinsi Banten, diperoleh R² sebesar 84,1%, MSE sebesar 0,042%, dan MAE sebesar 0,160%, yang oleh penulis dinilai menunjukkan performa model yang sangat baik.


Berdasarkan kedua bukti tersebut, saya menyimpulkan bahwa MAE dan MSE yang baik adalah yang mendekati nol, sedangkan R² yang baik umumnya mendekati satu. Ketiganya tidak dapat dinilai menggunakan arah yang sama karena mengukur aspek yang berbeda.
Meskipun demikian, saya memahami bahwa nilai yang dianggap baik juga bergantung pada skala data, karakteristik permasalahan, dan hasil pengujian pada data yang tidak digunakan untuk melatih model. Oleh karena itu, penilaian model sebaiknya mempertimbangkan ketiga metrik secara bersama-sama, bukan hanya berfokus pada satu nilai.

**Referensi :**
Syahreza, A., Ningrum, N. K., & Syahrazy, M. A. (2024). Perbandingan kinerja model prediksi cuaca: Random Forest, Support Vector Regression, dan XGBoost. Edumatic: Jurnal Pendidikan Informatika, 8(2), 526–534. 
https://doi.org/10.29408/edumatic.v8i2.27640
Khabani, M. Z., Aesyi, U. S., & Himawan, A. (2024). Prediksi tingkat pengangguran berdasarkan data time series menggunakan metode regresi linear (Studi kasus: Provinsi Banten). JSI: Jurnal Sistem Informasi (E-Journal), 16(2), 263–274.


