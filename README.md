# 📈 Retail Sales Forecast Dashboard

Dashboard ini digunakan untuk melakukan prediksi **Weekly_Sales** pada data `test.csv` yang tidak memiliki target. Dashboard dibangun menggunakan **Streamlit** dan menerapkan model terbaik hasil analisis, yaitu **XGBoost–LSTM Tuning**.

Penelitian ini secara keseluruhan menggunakan pendekatan **Hybrid Machine Learning** dengan melibatkan tiga model utama, yaitu **XGBoost**, **LSTM**, dan **Prophet**. Ketiga model tersebut digunakan pada tahap analisis di Google Colab untuk membandingkan performa model tunggal dan model hybrid. Berdasarkan hasil evaluasi, model **XGBoost–LSTM Tuning** dipilih sebagai model terbaik dan digunakan pada dashboard untuk melakukan prediksi data test ke depan.

Dashboard dibuat dalam bentuk **single-page Streamlit** agar lebih sederhana, ringan, dan mudah digunakan. Proses analisis, preprocessing, training model, hyperparameter tuning, evaluasi model, dan penyimpanan model dilakukan melalui **Google Colab**, sedangkan dashboard digunakan sebagai media implementasi untuk menjalankan prediksi pada data baru secara interaktif.

---

## ✨ Fitur Utama

Dashboard memiliki beberapa fitur utama sebagai berikut:

* Membaca otomatis data utama dari folder `data/`.
* Menampilkan status kesiapan data utama:

  * `train.csv`
  * `features.csv`
  * `stores.csv`
* Upload file `test.csv` sebagai data baru yang akan diprediksi.
* Menjalankan model **XGBoost–LSTM Tuning** untuk menghasilkan prediksi `Weekly_Sales`.
* Menampilkan log proses prediksi secara bertahap.
* Menampilkan grafik gabungan antara:

  * data aktual dari `train.csv`;
  * data prediksi dari `test.csv`.
* Menampilkan ringkasan hasil prediksi.
* Menampilkan tabel hasil prediksi.
* Menyediakan fitur download hasil prediksi lengkap.
* Menyediakan fitur download format submission.

---

## 📁 Struktur Folder

Pastikan struktur folder project seperti berikut:

```text
Dashboard/
├── app.py
├── requirements.txt
├── data/
│   ├── train.csv
│   ├── features.csv
│   ├── stores.csv
│   └── residual_history.csv
├── model/
│   ├── hybrid_xgboost_lstm_model_1_xgboost_20260626_040552.pkl
│   └── tuning_xgboost_lstm_model_lstm_residual_20260628_072659.keras
├── resource/
│   └── artefak_tuning_xgboost_lstm_20260628_072659.pkl
└── output/
```

Keterangan folder:

| Folder      | Fungsi                                                                                             |
| ----------- | -------------------------------------------------------------------------------------------------- |
| `data/`     | Menyimpan data utama seperti `train.csv`, `features.csv`, `stores.csv`, dan `residual_history.csv` |
| `model/`    | Menyimpan model XGBoost dan model LSTM residual                                                    |
| `resource/` | Menyimpan artefak pendukung seperti scaler residual dan konfigurasi model                          |
| `output/`   | Menyimpan hasil prediksi yang dihasilkan oleh dashboard                                            |

---

## 📌 File yang Dibutuhkan

### 1. File utama pada folder `data/`

File berikut harus tersedia di folder `data/`:

```text
train.csv
features.csv
stores.csv
```

Keterangan:

| File           | Fungsi                                                                                            |
| -------------- | ------------------------------------------------------------------------------------------------- |
| `train.csv`    | Data historis penjualan yang memiliki target `Weekly_Sales`                                       |
| `features.csv` | Data faktor eksternal seperti suhu, harga bahan bakar, CPI, unemployment, hari libur, dan promosi |
| `stores.csv`   | Data karakteristik toko seperti tipe toko dan ukuran toko                                         |

File berikut bersifat pendukung:

```text
residual_history.csv
```

File `residual_history.csv` digunakan untuk membantu model LSTM residual melakukan koreksi prediksi berdasarkan pola residual historis. Jika file ini tidak tersedia, dashboard tetap dapat menjalankan prediksi, tetapi koreksi residual LSTM dapat menjadi kurang kuat karena tidak memiliki riwayat residual sebelumnya.

### 2. File yang diupload melalui dashboard

File berikut diupload langsung melalui dashboard:

```text
test.csv
```

File `test.csv` digunakan sebagai data masa depan yang akan diprediksi. File ini tidak memiliki kolom `Weekly_Sales`, sehingga tidak digunakan untuk menghitung metrik evaluasi seperti MAE, RMSE, sMAPE, dan R².

---

## 🔗 Google Colab Analisis

Proses analisis data, preprocessing, feature engineering, pelatihan model, hyperparameter tuning, evaluasi model, serta penyimpanan model dilakukan melalui Google Colab.

Link Google Colab: [Colab Analisis Skripsi](https://colab.research.google.com/drive/1StxdU_H91jrYqjX972VbsljJQ6g7s0Jf?usp=sharing)

Keterangan penggunaan Google Colab:

| Bagian              | Fungsi                                                                                                |
| ------------------- | ----------------------------------------------------------------------------------------------------- |
| Google Colab        | Digunakan untuk analisis data, preprocessing, training model, tuning, evaluasi, dan penyimpanan model |
| Dashboard Streamlit | Digunakan untuk menjalankan prediksi pada `test.csv` tanpa target                                     |
| GitHub              | Digunakan untuk menyimpan kode dashboard, model, resource, dan dokumentasi project                    |
| Streamlit Cloud     | Digunakan untuk deployment dashboard agar dapat diakses secara online                                 |

Alur keterhubungan antara Google Colab dan dashboard adalah sebagai berikut:

```text
Google Colab
   ↓
Analisis data dan preprocessing
   ↓
Training model XGBoost, LSTM, Prophet, dan model hybrid
   ↓
Hyperparameter tuning
   ↓
Evaluasi model
   ↓
Penentuan model terbaik
   ↓
Penyimpanan model dan artefak
   ↓
Dashboard Streamlit
   ↓
Upload test.csv
   ↓
Prediksi Weekly_Sales pada data test
```

Google Colab berperan sebagai tempat utama pengembangan model, sedangkan dashboard Streamlit berperan sebagai media implementasi untuk menampilkan hasil prediksi secara lebih interaktif.

---

## 🧠 Model Hybrid dan Penentuan Model Terbaik

Penelitian ini menggunakan pendekatan **Hybrid Machine Learning** dengan melibatkan tiga model utama, yaitu **XGBoost**, **LSTM**, dan **Prophet**. Ketiga model tersebut digunakan karena memiliki karakteristik yang saling melengkapi.

* **XGBoost** digunakan untuk menangkap hubungan nonlinier pada data tabular.
* **LSTM** digunakan untuk mempelajari pola sekuensial atau deret waktu.
* **Prophet** digunakan untuk menangkap pola tren, musiman, dan pengaruh periode waktu tertentu.

Pada tahap analisis di Google Colab, model tidak hanya dibangun sebagai model tunggal, tetapi juga dikembangkan menjadi beberapa kombinasi hybrid. Kombinasi tersebut digunakan untuk membandingkan kemampuan masing-masing pasangan model dalam memprediksi penjualan ritel mingguan.

Model hybrid yang dianalisis melibatkan kombinasi dari model:

```text
XGBoost
LSTM
Prophet
```

Berdasarkan hasil evaluasi terhadap seluruh model, model terbaik yang diperoleh adalah:

```text
XGBoost–LSTM Tuning
```

Model **XGBoost–LSTM Tuning** dipilih karena menghasilkan nilai **sMAPE terendah pada data Test internal** dibandingkan model lainnya. Pada model ini, XGBoost berperan sebagai model utama untuk menghasilkan prediksi awal, sedangkan LSTM digunakan untuk mempelajari residual atau sisa kesalahan prediksi dari XGBoost. Prediksi akhir diperoleh dari gabungan prediksi XGBoost dan koreksi residual LSTM.

Formula prediksi akhir:

```text
Prediksi Akhir = Prediksi XGBoost + Prediksi Residual LSTM
```

---

## 📊 Hasil Evaluasi Model Terbaik

Hasil evaluasi model terbaik adalah sebagai berikut:

| Peringkat | Tahap  | Model Terbaik       | Data Acuan |       MAE |      RMSE |   sMAPE |     R² | Dasar Penentuan                        |
| --------- | ------ | ------------------- | ---------- | --------: | --------: | ------: | -----: | -------------------------------------- |
| 1         | Tuning | XGBoost–LSTM Tuning | Test       | 1365.1854 | 2983.9209 | 16.4627 | 0.9816 | sMAPE terendah pada data Test internal |

Interpretasi metrik evaluasi:

| Metrik |     Nilai | Interpretasi                                                                                                                                                                                                  |
| ------ | --------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MAE    | 1365.1854 | Rata-rata selisih absolut antara nilai aktual dan nilai prediksi adalah sekitar 1365.1854 satuan penjualan. Semakin kecil nilai MAE, semakin baik hasil prediksi model.                                       |
| RMSE   | 2983.9209 | RMSE menunjukkan tingkat kesalahan prediksi dengan penalti lebih besar pada error yang tinggi. Nilai ini membantu melihat apakah model masih menghasilkan kesalahan besar pada beberapa titik data.           |
| sMAPE  |   16.4627 | sMAPE menunjukkan tingkat kesalahan relatif dalam bentuk persentase. Model ini dipilih karena memiliki nilai sMAPE terendah pada data Test internal.                                                          |
| R²     |    0.9816 | R² menunjukkan bahwa model mampu menjelaskan sekitar 98.16% variasi data aktual pada data Test internal. Nilai ini menunjukkan bahwa model memiliki kemampuan yang kuat dalam mengikuti pola data yang diuji. |

Berdasarkan hasil tersebut, model **XGBoost–LSTM Tuning** digunakan pada dashboard Streamlit sebagai model utama untuk melakukan prediksi pada `test.csv`. Meskipun penelitian melibatkan kombinasi model XGBoost, LSTM, dan Prophet, dashboard hanya mengimplementasikan model terbaik agar proses prediksi lebih fokus, efisien, dan sesuai dengan hasil evaluasi akhir.

---

## 🔄 Alur Kerja Prediksi pada Dashboard

Alur prediksi pada dashboard adalah sebagai berikut:

```text
test.csv
   ↓
Integrasi dengan features.csv dan stores.csv
   ↓
Feature engineering
   ↓
Pembentukan fitur lag dan rolling dari train.csv
   ↓
Prediksi awal menggunakan XGBoost
   ↓
Koreksi residual menggunakan LSTM
   ↓
Prediksi akhir Weekly_Sales
```

Penjelasan alur:

| Tahap                 | Penjelasan                                                                                                                             |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Upload `test.csv`     | User mengupload data test tanpa target melalui dashboard                                                                               |
| Integrasi data        | `test.csv` digabungkan dengan `features.csv` dan `stores.csv`                                                                          |
| Feature engineering   | Sistem membentuk fitur promosi, waktu, hari libur, dan variabel pendukung lain                                                         |
| Lag dan rolling       | Sistem menggunakan `train.csv` untuk membentuk fitur historis seperti `Sales_Lag1`, `Sales_Lag4`, `Rolling_Mean4`, dan `Rolling_Mean8` |
| Prediksi XGBoost      | Model XGBoost menghasilkan prediksi awal                                                                                               |
| Koreksi LSTM residual | Model LSTM residual mempelajari sisa kesalahan dari XGBoost                                                                            |
| Prediksi akhir        | Hasil akhir diperoleh dari prediksi XGBoost ditambah prediksi residual LSTM                                                            |

Pada dashboard ini, `train.csv`, `features.csv`, dan `stores.csv` digunakan sebagai data utama untuk membentuk konteks historis dan fitur pendukung. Sementara itu, `test.csv` digunakan sebagai data baru tanpa target yang akan diprediksi oleh model terbaik.

Karena `test.csv` tidak memiliki kolom `Weekly_Sales`, dashboard tidak menghitung metrik evaluasi seperti MAE, RMSE, sMAPE, dan R² pada data tersebut. Evaluasi model dilakukan pada Google Colab menggunakan data Test internal yang masih memiliki nilai aktual.

---

## 📈 Visualisasi Train dan Test

Dashboard menampilkan satu grafik utama yang menggabungkan data historis dan data prediksi.

* **Aktual Train** ditampilkan sebagai garis gelap.
* **Prediksi Test** ditampilkan sebagai garis hijau putus-putus.

Tujuan visualisasi ini adalah untuk memperlihatkan alur kronologis dari data historis menuju hasil prediksi ke depan. Melalui grafik tersebut, pengguna dapat melihat bagaimana pola penjualan aktual pada data train dilanjutkan dengan hasil prediksi pada data test.

---

## ⚙️ Instalasi

Buka terminal pada folder utama project, lalu jalankan:

```bash
pip install -r requirements.txt
```

Contoh isi `requirements.txt`:

```txt
streamlit>=1.37,<2
pandas==2.2.2
numpy==1.26.4
plotly==5.24.1
joblib==1.4.2
scikit-learn==1.5.2
xgboost==2.1.1
tensorflow-cpu==2.16.2
keras==3.4.1
```

Catatan:

* `tensorflow-cpu` digunakan karena dashboard hanya melakukan proses prediksi, bukan pelatihan ulang model.
* Gunakan **Python 3.11** agar kompatibel dengan TensorFlow dan Keras.
* Hindari memakai `requirements.txt` dari hasil full `pip freeze` Colab karena biasanya terlalu besar dan dapat menyebabkan deploy gagal.

---

## 🚀 Cara Menjalankan Dashboard

Jalankan perintah berikut dari folder utama project:

```bash
streamlit run app.py
```

Setelah perintah dijalankan, dashboard akan terbuka otomatis di browser.

Apabila browser tidak terbuka otomatis, buka alamat berikut secara manual:

```text
http://localhost:8501
```

---

## 🧭 Cara Menggunakan Dashboard

1. Pastikan file utama sudah tersedia di folder `data/`:

   ```text
   train.csv
   features.csv
   stores.csv
   ```

2. Pastikan file model sudah tersedia di folder `model/`:

   ```text
   hybrid_xgboost_lstm_model_1_xgboost_20260626_040552.pkl
   tuning_xgboost_lstm_model_lstm_residual_20260628_072659.keras
   ```

3. Pastikan file artefak sudah tersedia di folder `resource/`:

   ```text
   artefak_tuning_xgboost_lstm_20260628_072659.pkl
   ```

4. Jalankan dashboard:

   ```bash
   streamlit run app.py
   ```

5. Upload file `test.csv` pada bagian upload data.

6. Klik tombol:

   ```text
   Run Prediction
   ```

7. Tunggu proses prediksi selesai.

8. Dashboard akan menampilkan:

   * ringkasan hasil prediksi;
   * grafik aktual train dan prediksi test;
   * tabel hasil prediksi;
   * tombol download hasil prediksi.

---

## 📊 Output yang Dihasilkan

Setelah prediksi selesai, hasil akan disimpan otomatis ke folder `output/`.

```text
output/
├── hasil_prediksi_test.csv
└── submission.csv
```

Keterangan output:

| File                      | Fungsi                                         |
| ------------------------- | ---------------------------------------------- |
| `hasil_prediksi_test.csv` | Hasil prediksi lengkap beserta fitur pendukung |
| `submission.csv`          | Format ringkas berisi `Id` dan `Weekly_Sales`  |

---

## ☁️ Deploy ke Streamlit Cloud

Untuk melakukan deployment ke Streamlit Cloud, ikuti langkah berikut:

1. Push seluruh project ke GitHub.
2. Buka Streamlit Community Cloud.
3. Pilih repository project.
4. Pastikan file utama yang dijalankan adalah:

```text
app.py
```

5. Pada bagian **Advanced Settings**, pilih versi Python:

```text
Python 3.11
```

6. Klik **Deploy**.

Struktur minimal repository untuk deployment:

```text
Dashboard/
├── app.py
├── requirements.txt
├── data/
├── model/
├── resource/
└── output/
```

---

## 🔧 Troubleshooting

### 1. Error saat instalasi dependency

Apabila muncul error seperti berikut:

```text
installer returned a non-zero exit code
```

Pastikan versi Python yang digunakan adalah **Python 3.11**.

Selain itu, pastikan `requirements.txt` tidak terlalu banyak berisi library dari hasil `pip freeze` Colab, karena file tersebut biasanya terlalu besar dan dapat menyebabkan proses instalasi di Streamlit Cloud menjadi gagal.

---

### 2. Model LSTM tidak dapat dimuat

Apabila muncul error terkait Keras atau TensorFlow, pastikan `requirements.txt` menggunakan versi berikut:

```txt
tensorflow-cpu==2.16.2
keras==3.4.1
```

Selain itu, pastikan file model LSTM residual tersedia pada folder `model/`.

---

### 3. Hasil prediksi berbeda dengan notebook

Periksa beberapa hal berikut:

* `train.csv` sudah tersedia di folder `data/`;
* `features.csv` dan `stores.csv` sesuai dengan dataset asli;
* `residual_history.csv` tersedia apabila ingin koreksi LSTM lebih kuat;
* versi library di dashboard mendekati versi library saat model dilatih;
* urutan dan nama kolom fitur tidak berubah;
* model XGBoost dan LSTM residual yang digunakan adalah model hasil training terbaru.

---

### 4. Dashboard berjalan lama saat prediksi

Proses prediksi dapat berjalan lama apabila jumlah baris `test.csv` besar. Untuk mempercepat proses, dashboard menggunakan strategi prediksi berbasis batch per tanggal, sehingga model tidak dipanggil satu per satu untuk setiap baris.

Strategi ini membuat proses prediksi lebih efisien karena data Walmart memiliki banyak baris, tetapi jumlah periode minggu pada data test relatif lebih sedikit.

---

## 🧾 Catatan Penggunaan

Dashboard ini digunakan untuk tahap implementasi atau deployment model. Oleh karena itu, dashboard tidak menghitung metrik evaluasi pada `test.csv`, karena file tersebut tidak memiliki target `Weekly_Sales`.

Evaluasi model seperti MAE, RMSE, sMAPE, dan R² dilakukan pada Google Colab menggunakan data Test internal yang berasal dari pembagian `train.csv`.

---

## 👤 Pengembang

Dashboard ini dikembangkan untuk mendukung penelitian skripsi terkait prediksi penjualan ritel menggunakan pendekatan **Hybrid Machine Learning** dengan mengombinasikan tiga model utama, yaitu **XGBoost**, **LSTM**, dan **Prophet**. Berdasarkan hasil evaluasi akhir, model **XGBoost–LSTM Tuning** dipilih sebagai model terbaik dan digunakan pada dashboard untuk melakukan prediksi `Weekly_Sales` pada data `test.csv`.
