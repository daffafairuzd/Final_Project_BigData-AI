# Final_Project_BigData&AI

Sistem deteksi pencucian uang (*Anti-Money Laundering* / AML Fraud Detection) skala besar menggunakan **Apache Spark** untuk praproses/rekayasa fitur dan algoritme ensemble machine learning (**LightGBM, XGBoost, Random Forest**) dengan optimasi **Risk-Based Approach (RBA)**.

---

## 📌 Ringkasan Kasus (Case Study)
Pencucian uang (*money laundering*) merupakan salah satu ancaman terbesar dalam sektor finansial global. Proyek ini bertujuan untuk membangun sistem deteksi transaksi fraud/pencucian uang secara *end-to-end* yang efisien, handal, dan *scalable*. 

Tantangan utama dalam kasus ini adalah **ketimpangan kelas yang sangat ekstrem** (*extreme class imbalance*) di mana jumlah transaksi normal jauh mendominasi transaksi mencurigakan. Untuk mengatasi hal tersebut, proyek ini menerapkan:
1. **Feature Engineering Modular & Tanpa Kebocoran Data (Leakage-Free)**: Menghitung parameter agregasi velocity (frekuensi & volume transaksi dalam 24 jam terakhir) secara terdistribusi menggunakan Apache Spark.
2. **Downsampling Cerdas**: Melakukan penyeimbangan data latih dengan rasio 1:20 (1 fraud banding 20 normal) untuk melatih model secara optimal.
3. **Data-Driven Risk-Based Approach (RBA)**: Menggunakan batas threshold keputusan dinamis berbasis nominal transaksi untuk memotong tingkat alarm palsu (*false alert noise*) tanpa meloloskan transaksi fraud bernilai tinggi.

---

## 📊 Dataset yang Digunakan
Dataset yang digunakan dalam proyek ini adalah dataset transaksi keuangan berskala besar:
* **Nama Berkas**: `HI-Medium_Trans.csv` (berukuran **3.03 GB** dengan total **~31.9 juta baris** transaksi).
* **Kolom Kunci**: `Timestamp`, `From_Account`, `To_Account`, `Amount_Paid`, `Payment_Format`, dan target label binary `Is_Laundering`.
* **Karakteristik**: Mengandung ketimpangan kelas ekstrem dengan rasio sekitar **1:197** (hanya sekitar ~0.5% transaksi bernilai positif pencucian uang).

---

## 📈 Hasil Evaluasi & Performa Model
Evaluasi model dilakukan pada data uji skala besar yang sangat tidak seimbang (~6.38 juta baris). Berikut adalah tabel performansi lengkap dari ketiga model machine learning yang diuji:

| Model & Skenario | PR-AUC | ROC-AUC | Accuracy | Precision | Recall | F1-Score | False Positives (Alert Noise) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Tuned 0.99)** | **0.3862** | **0.9831** | **99.90%** | **54.26%** | 32.71% | **40.81%** | **1,902** |
| **XGBoost (Dynamic RBA)** | **0.3862** | **0.9831** | 99.70% | 19.46% | 55.47% | 28.82% | 15,831 |
| **XGBoost (Default 0.50)** | **0.3862** | **0.9831** | 93.49% | 1.46% | 88.91% | 2.87% | 414,271 |
| **LightGBM (Tuned 0.99)** | 0.2772 | 0.9824 | 99.87% | 39.40% | 32.43% | 35.58% | 3,441 |
| **LightGBM (Dynamic RBA)** | 0.2772 | 0.9824 | 99.46% | 11.58% | 59.87% | 19.41% | 31,521 |
| **LightGBM (Default 0.50)** | 0.2772 | 0.9824 | 88.90% | 0.92% | **95.06%** | 1.82% | 708,020 |
| **Random Forest (Tuned 0.95)** | 0.1343 | 0.9802 | 99.82% | 18.33% | 19.78% | 19.03% | 6,078 |
| **Random Forest (Default 0.50)** | 0.1343 | 0.9802 | 89.94% | 1.00% | 93.56% | 1.97% | 641,560 |

### Temuan Utama & Progress Proyek:
1. **XGBoost Unggul Mutlak**: Model XGBoost mencapai nilai PR-AUC tertinggi sebesar **0.3862** dengan nilai F1-Score optimal sebesar **40.81%** pada threshold statis 0.99.
2. **Kekuatan Dynamic RBA**: Dibandingkan dengan threshold statis hasil tuning (0.95 - 0.99) yang memiliki kelemahan meloloskan terlalu banyak transaksi fraud bernilai tinggi, skenario **Dynamic Risk-Based Approach (RBA)** berhasil mendongkrak tingkat kepekaan (*Recall*) secara signifikan (naik ke **55.47%** untuk XGBoost dan **59.87%** untuk LightGBM) sambil tetap memotong alarm palsu hingga lebih dari **90%** dibandingkan dengan skenario default.
3. **Analisis Velocity Terbukti Efektif**: Berdasarkan grafik kepentingan fitur (*Feature Importance*), akumulasi volume transaksi dalam 24 jam terakhir (`sender_amount_sum_24h` & `receiver_amount_sum_24h`) menjadi prediktor paling signifikan, memvalidasi keberhasilan modul *feature engineering* terdistribusi Spark.

---

## 🛠️ Struktur Repositori & Kode
* `main.py` : Orkestrator utama pipeline untuk mengeksekusi preprocessing, EDA, pelatihan model, evaluasi, dan plotting secara end-to-end.
* `src/` : Folder berisi modul fungsional proyek:
  * `config.py` : Pengaturan konfigurasi, path, dan hyperparameter terpusat.
  * `spark_session.py` : Pengelolaan dan konfigurasi aman inisialisasi sesi Apache Spark di OS Windows.
  * `preprocess.py` & `preprocessing.py` : Ingestion CSV mentah, schema validation, data cleaning, dan penulisan berkas Parquet awal.
  * `feature_engineering.py` : Proses pemecahan data latih/uji (stratified split), kalkulasi velocity lookup (velocity & global activity), downsampling, dan proses label encoding.
  * `eda.py` & `eda_spark.py` : Analisis eksplorasi data statistik terdistribusi menggunakan Apache Spark SQL dan analisis quantile bucket RBA.
  * `train.py` : Skrip pelatihan model machine learning (Random Forest, LightGBM, XGBoost) dengan mekanisme alokasi GPU dan fallback otomatis ke CPU.
  * `evaluate.py` : Modul sweep threshold keputusan optimal dan konstruksi tiers dynamic RBA berbasis data.
  * `visualize.py` : Pembuatan grafik evaluasi kinerja model dan hasil visualisasi EDA.
  * `utils.py` : Fungsi utilitas pembantu (logging, timing, file handler).
* `models/` : Berkas parameter encoding dan bobot model terlatih terbaik yang siap pakai.
* `plots/` : Berkas gambar visualisasi kurva evaluasi model dan EDA hasil eksekusi.

---

## 🚀 Cara Menjalankan Pipeline secara Lokal

### 1. Prasyarat (Prerequisites)
Pastikan sistem Anda telah memiliki Java (JDK 8 atau JDK 11) terinstal dan terdaftar di PATH untuk menjalankan Apache Spark.

### 2. Instalasi Dependensi
Buat virtual environment Python dan pasang pustaka yang diperlukan:
```bash
# Membuat virtual environment
python -m venv .venv

# Mengaktifkan virtual environment (Windows)
.venv\Scripts\activate

# Memasang pustaka dependensi
pip install pyspark pandas numpy lightgbm xgboost scikit-learn optuna joblib matplotlib seaborn
```

### 3. Mempersiapkan Dataset Mentah
1. Dapatkan dataset transaksi `HI-Medium_Trans.csv` (misal dari platform IBM/Kaggle).
2. Letakkan file `HI-Medium_Trans.csv` langsung di dalam **direktori root** proyek ini (sejajar dengan `main.py`).

### 4. Eksekusi Pipeline Utama
Jalankan file orkestrator utama untuk mengeksekusi seluruh pipeline dari preprocessing hingga visualisasi:
```bash
python main.py
```
> **Catatan**: Pipeline dilengkapi fitur penyimpanan Parquet bertahap. Jika file Parquet hasil pemrosesan sudah terbentuk di `data/processed/`, langkah preprocessing & EDA akan dilewati secara otomatis untuk menghemat waktu eksekusi.

### 5. Menjalankan Modul Mandiri (Opsional)
* **EDA Spark SQL**: Untuk menjalankan query analisis data terdistribusi secara manual:
  ```bash
  python src/eda_spark.py
  ```
* **Evaluasi & Visualisasi Ulang**: Untuk menguji ambang batas baru atau mengubah visualisasi grafik:
  ```bash
  python src/visualize.py
  ```
