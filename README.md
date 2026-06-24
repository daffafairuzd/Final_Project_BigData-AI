# 🛡️ Anti-Money Laundering (AML) Fraud Detection System

Sistem deteksi pencucian uang (*Anti-Money Laundering* / AML Fraud Detection) menggunakan **Apache Spark (PySpark)** untuk pemrosesan paralel terdistribusi, pembersihan data, eksplorasi data, dan rekayasa fitur, dikombinasikan dengan algoritme ensemble machine learning (**XGBoost, LightGBM, Random Forest**) serta dioptimalkan menggunakan **Risk-Based Approach (RBA)** dinamis untuk meminimalkan *alert noise*.

---

## 📌 Ringkasan Kasus (Case Study)

Pencucian uang (*money laundering*) merupakan salah satu ancaman kejahatan kerah putih terbesar dalam sektor finansial global. Proyek ini bertujuan untuk membangun sistem deteksi transaksi mencurigakan secara *end-to-end* yang efisien, handal, dan *scalable* pada dataset berskala besar (juta-an baris transaksi) yang disimulasikan oleh IBM.

Tantangan teknis utama dalam pemodelan data transaksi AML meliputi:
1. **Ketimpangan Kelas Ekstrem (*Extreme Class Imbalance*)**: Di mana jumlah transaksi normal jauh mendominasi transaksi pencucian uang (kurang dari 0.5% dari total data).
2. **Kebocoran Data (*Data Leakage*)**: Rentannya sistem membocorkan informasi masa depan atau data uji ke dalam proses rekayasa fitur (seperti menghitung rata-rata aktivitas transaksi akun pengirim secara global).
3. **Beban Operasional Alert Palsu (*False Alert Noise*)**: Penggunaan threshold klasifikasi default (0.50) menghasilkan ratusan ribu alarm palsu yang tidak mungkin ditinjau secara manual oleh tim kepatuhan (*compliance* bank).

Proyek ini memecahkan ketiga tantangan tersebut dengan membagi pipeline secara modular, menerapkan pemisahan data berbasis stratifikasi di awal, rekayasa fitur velocity lookup terisolasi, downsampling terkontrol, ordinal encoding dengan penanganan nilai tidak dikenal (*out-of-vocabulary*), serta kalibrasi threshold keputusan berbasis tingkat risiko nominal transaksi (RBA).

---

## ⚙️ Detail Pemrosesan Data & Desain Sistem

### 1. Pembagian Data (*Data Splitting Scheme*)
Untuk menjamin evaluasi model yang adil dan realistis tanpa adanya kebocoran informasi (*data leakage*):
* **Metode**: **Stratified Random Split** terdistribusi menggunakan PySpark.
* **Proporsi**: **80% Data Latih (Train Set)** dan **20% Data Uji (Test Set)**.
* **Alur Kerja**:
  1. Dataset awal disaring menjadi dua DataFrame terpisah: kelas positif (`Is_Laundering == 1`) dan kelas negatif (`Is_Laundering == 0`).
  2. Fungsi `.randomSplit([0.8, 0.2], seed=42)` dipanggil secara independen pada masing-masing DataFrame.
  3. Hasil pemisahan masing-masing kelas digabungkan kembali menggunakan operasi `.union()`.
* **Tujuan**: Menjamin rasio ketimpangan label target `Is_Laundering` tetap sama persis (~0.5% fraud) baik di partisi data latih maupun data uji sebelum fitur-fitur agregat dihitung. Hal ini sangat krusial agar model diuji pada data yang merepresentasikan distribusi dunia nyata.

---

### 2. Skema Downsampling Data Latih (*Downsampling Scheme*)
Karena rasio imbalanced bawaan dataset sangat ekstrem (**1:197** atau ~0.5% fraud), model machine learning akan kesulitan mengenali transaksi fraud jika dilatih langsung pada data mentah. Model akan cenderung memprediksi semua transaksi sebagai normal demi meminimalkan tingkat kesalahan global.
* **Data Latih**: Kami melakukan downsampling acak pada kelas mayoritas (transaksi normal) hingga mencapai target rasio penyeimbangan kelas **1:20** (1 transaksi fraud berbanding 20 transaksi normal) berdasarkan konfigurasi `IMBALANCE_RATIO = 20` di [src/config.py](src/config.py).
* **Alur Kerja Downsampling**:
  1. Hitung jumlah transaksi fraud di data latih: `num_train_fraud`.
  2. Hitung jumlah transaksi normal di data latih: `num_train_normal`.
  3. Hitung jumlah target transaksi normal yang diinginkan: `target_normal_count = num_train_fraud * 20`.
  4. Hitung fraksi pengambilan sampel: `sample_fraction = min(target_normal_count / num_train_normal, 1.0)`.
  5. Panggil fungsi `.sample(withReplacement=False, fraction=sample_fraction, seed=42)` pada DataFrame kelas normal.
  6. Satukan kembali baris normal hasil sampling dengan seluruh baris fraud latih melalui `.union()`.
* **Data Uji**: Tetap dipertahankan pada kondisi aslinya (**1:197**) tanpa downsampling guna menguji model pada kondisi operasional bank sesungguhnya tanpa memanipulasi bias performansi.

---

### 3. Skema Encoding Variabel Kategorikal (*Categorical Feature Encoding*)
Variabel kategorikal dalam data transaksi finansial harus dikonversi menjadi format angka sebelum dimasukkan ke model machine learning.
* **Variabel yang Di-encode**: `Payment_Format`, `Receiving_Currency`, `Payment_Currency`, `From_Bank`, `To_Bank`.
* **Metode**: **Ordinal Encoding** menggunakan kelas `OrdinalEncoder` dari pustaka `scikit-learn`.
* **Pencegahan Leakage & Crash**:
  * Encoder dilatih (*fitted*) **hanya** menggunakan Data Latih (`train_balanced.parquet`).
  * Encoder yang telah dilatih disimpan dalam bentuk biner beralamat di [data/processed/feature_encoders.joblib](data/processed/feature_encoders.joblib).
  * Data latih dan data uji kemudian ditransformasikan secara terpisah menggunakan encoder yang telah disimpan tersebut.
  * Kami mengaktifkan parameter `handle_unknown='use_encoded_value'` dengan `unknown_value=-1`. Ini memastikan jika ada mata uang baru, format pembayaran baru, atau bank baru yang muncul di data uji (atau di data produksi di masa depan), sistem tidak akan crash melainkan memetakan nilai tersebut secara aman ke indeks `-1`.

---

### 4. Rekayasa Fitur & Pencegahan *Leakage* (*Feature Engineering*)
Seluruh fitur berbasis agregasi dihitung **hanya dari Data Latih (Train Set)**, kemudian baru digabungkan (*left-joined*) kembali ke masing-masing partisi data latih dan data uji. Hal ini penting untuk mencegah model mengetahui statistik masa depan dari Data Uji. Fitur yang diekstrak meliputi:
* **Fitur Temporal**: `Hour` (jam transaksi) dan `Day_of_Week` (hari transaksi) yang diekstrak langsung dari kolom `Timestamp` menggunakan fungsi bawaan Spark SQL.
* **Fitur Konversi**: `Is_Currency_Conversion` (bernilai 1 jika mata uang pengirim berbeda dengan penerima, bernilai 0 jika sama).
* **Daily Velocity (24 Jam)**: 
  * `sender_tx_count_24h` / `receiver_tx_count_24h`: Frekuensi transaksi pengirim/penerima dalam 24 jam terakhir.
  * `sender_amount_sum_24h` / `receiver_amount_sum_24h`: Akumulasi volume nominal transaksi pengirim/penerima dalam 24 jam terakhir.
  * *Mekanisme OOM-safe*: Dihitung secara harian dengan mengelompokkan data berdasarkan akun dan tanggal. Kemudian dilakukan self-join dengan offset tanggal kemarin (`Prev_Date = Date - 1`) untuk menjumlahkan transaksi hari ini dan kemarin demi menyimulasikan akumulasi jendela waktu 24 jam secara *memory-efficient* tanpa lag windowing berat.
* **Global Activity**: Rata-rata nominal (`sender_avg_amount` / `receiver_avg_amount`) dan jumlah transaksi seumur hidup (`sender_tx_count` / `receiver_tx_count`) akun pengirim/penerima.
* **Graph Degree**: 
  * `out_degree`: Derajat keluar, mengukur seberapa banyak akun penerima unik yang dikirimi uang oleh akun pengirim (mendeteksi pola penyebaran uang/*structuring*).
  * `in_degree`: Derajat masuk, mengukur seberapa banyak akun pengirim unik yang menyetor uang ke akun penerima (mendeteksi pola pengumpulan uang/*layering & integration*).

---

## 🔍 Eksplorasi Data Analisis (EDA) dengan Spark SQL

Modul [src/eda.py](src/eda.py) melakukan eksplorasi data secara paralel dan terdistribusi pada berkas `cleaned.parquet` menggunakan modul Spark SQL. Komponen analisis EDA yang dijalankan meliputi:

1. **Q1: Class Distribution**: Menghitung sebaran absolut dan persentase transaksi normal vs fraud untuk mengukur tingkat imbalance awal. Output divisualisasikan dalam grafik donat dan barplot pada [plots/eda/01_class_imbalance.png](plots/eda/01_class_imbalance.png).
2. **Q2: Amount Statistics**: Agregasi statistik nominal uang (`Amount_Paid`) per kelas (rata-rata, nilai minimum, kuartil bawah Q1, median/Q2, kuartil atas Q3, nilai maksimum, dan standar deviasi) untuk melihat perbedaan skala transaksi. Output divisualisasikan dalam boxplot log-scale pada [plots/fraud_amount_distribution.png](plots/fraud_amount_distribution.png).
3. **Q3: Payment Format**: Analisis persentase penggunaan metode pembayaran (ACH, Cheque, Credit Card, dll.) pada transaksi normal vs fraud. Terlihat bahwa transaksi fraud memiliki preferensi tinggi pada metode pembayaran tertentu seperti Wire Transfer. Output divisualisasikan dalam clustered bar chart pada [plots/payment_format_comparison.png](plots/payment_format_comparison.png).
4. **Q4: Daily Timeline**: Analisis deret waktu volume harian transaksi fraud untuk mengidentifikasi tanggal puncak aktivitas pencucian uang secara historis. Output divisualisasikan pada [plots/eda/02_daily_fraud_timeline.png](plots/eda/02_daily_fraud_timeline.png).
5. **Q5: Top Accounts (Graph Behavior)**: Mengidentifikasi 15 akun pengirim paling aktif beserta jumlah penerima unik dan volume fraud yang dialirkan. Output divisualisasikan pada [plots/eda/04_tier_segmentation.png](plots/eda/04_tier_segmentation.png).
6. **Q6: Hourly Distribution**: Menghitung jam-jam terjadinya transaksi fraud dalam kurun waktu 24 jam sehari untuk mendeteksi jam operasional aktivitas mencurigakan. Output divisualisasikan pada [plots/eda/05_hourly_heatmap.png](plots/eda/05_hourly_heatmap.png).
7. **Q7: RBA NTILE Analysis**: Mengelompokkan nominal `Amount_Paid` transaksi ke dalam 20 bucket quantile menggunakan `approxQuantile` dan `Bucketizer` terdistribusi PySpark untuk memetakan *fraud rate* per bucket nominal secara aman dari memory overflow. Hasil ini langsung digunakan untuk merancang batas nominal Low/Medium/High Risk Tier pada RBA. Output divisualisasikan pada [plots/rba_fraud_rate_buckets.png](plots/rba_fraud_rate_buckets.png).
8. **Correlation Matrix**: Penghitungan matriks korelasi Pearson antar seluruh fitur numerik menggunakan Pandas pada data latih yang telah seimbang untuk melacak korelasi fitur tanpa membebani memori driver Spark. Output divisualisasikan pada [plots/eda/03_correlation_heatmap.png](plots/eda/03_correlation_heatmap.png).

---

## 🛡️ Desain Risk-Based Approach (RBA) Dinamis

Metode klasifikasi konvensional menerapkan threshold keputusan statis (misal $t = 0.50$ atau $t = 0.99$) untuk semua transaksi tanpa mempedulikan nominal transaksi. Hal ini tidak efisien secara bisnis dan operasional:
* Meloloskan transaksi fraud bernilai raksasa (misal $10,000,000) adalah bencana regulasi dan finansial yang fatal bagi bank.
* Menghasilkan ribuan alert palsu pada transaksi bernilai mikro (misal $5) membuang-buang waktu peninjau kepatuhan secara sia-sia.

Untuk mengatasi ini, kami merancang **Dynamic Risk-Based Approach (RBA)** berbasis 3 Risk Tiers nominal transaksi yang dibentuk secara otomatis dari hasil analisis quantile EDA:
1. **Low Risk Tier (Nominal < $690.87)**:
   * **Logika**: Transaksi harian bernilai mikro dengan tingkat fraud rate di bawah baseline rata-rata.
   * **Kebijakan**: Threshold keputusan ketat (**0.99**). Mengurangi jutaan alert palsu pada transaksi mikro harian.
2. **Medium Risk Tier ($690.87 - $4,484.14)**:
   * **Logika**: Transaksi dengan frekuensi dan tingkat fraud rate standar.
   * **Kebijakan**: Threshold optimal standar (**0.98 - 0.99**).
3. **High Risk Tier (Nominal >= $4,484.14)**:
   * **Logika**: Transaksi bernilai nominal besar di mana aktivitas pencucian uang terkonsentrasi sangat padat.
   * **Kebijakan**: Threshold keputusan sensitif (**0.95**). Mengoptimalkan tingkat kepekaan (*Recall*) untuk memastikan tidak ada transaksi bernilai besar yang lolos dari pemantauan.

Batas nominal dan threshold optimal per tier dicari secara dinamis saat evaluasi berjalan ([src/evaluate.py](src/evaluate.py)) dengan meminimalkan error alert noise di bawah batasan presisi minimum per tier.

---

## 📈 Hasil Evaluasi & Performa Model

Evaluasi model dilakukan pada data uji skala besar yang sangat tidak seimbang (~6.38 juta baris). Berikut adalah tabel performansi lengkap dari ketiga model machine learning yang diuji:

| Model & Skenario | PR-AUC | ROC-AUC | Accuracy | Precision | Recall | F1-Score | False Positives (Alert Noise) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Tuned 0.99)** | **0.3862** | **0.9831** | **99.90%** | **54.26%** | 32.71% | **40.81%** | **1,902** |
| **XGBoost (Dynamic RBA)** | **0.3862** | **0.9831** | 99.70% | 19.46% | 55.47% | 28.82% | 15,831 |
| **XGBoost (Default 0.50)** | **0.3862** | **0.9831** | 93.49% | 1.46% | 88.91% | 2.87% | 414,271 |
| **LightGBM (Tuned 0.99)** | 0.2772 | 0.9824 | 99.87% | 39.40% | 32.43% | 35.58% | 3,441 |
| **LightGBM (Dynamic RBA)** | 0.2772 | 0.9824 | 99.46% | 11.58% | **59.87%** | 19.41% | 31,521 |
| **LightGBM (Default 0.50)** | 0.2772 | 0.9824 | 88.90% | 0.92% | **95.06%** | 1.82% | 708,020 |
| **Random Forest (Tuned 0.95)** | 0.1343 | 0.9802 | 99.82% | 18.33% | 19.78% | 19.03% | 6,078 |
| **Random Forest (Default 0.50)** | 0.1343 | 0.9802 | 89.94% | 1.00% | 93.56% | 1.97% | 641,560 |

### Analisis Kunci:
* **XGBoost Unggul Mutlak**: Mencapai PR-AUC tertinggi sebesar **0.3862** dengan F1-Score optimal sebesar **40.81%** pada threshold statis 0.99.
* **Kekuatan Dynamic RBA**: Dibandingkan dengan threshold statis hasil tuning (0.95 - 0.99) yang berisiko meloloskan fraud bernilai tinggi, skenario **Dynamic Risk-Based Approach (RBA)** berhasil mendongkrak tingkat kepekaan (*Recall*) secara signifikan (naik ke **55.47%** untuk XGBoost dan **59.87%** untuk LightGBM) sambil tetap memotong alert palsu hingga lebih dari **95%** dibandingkan dengan skenario default 0.50.
* **Analisis Velocity Sangat Efektif**: Berdasarkan grafik kepentingan fitur (*Feature Importance*), akumulasi volume transaksi dalam 24 jam terakhir (`sender_amount_sum_24h` & `receiver_amount_sum_24h`) menjadi prediktor paling signifikan, membuktikan bahwa rekayasa fitur velocity lookup berhasil menangkap anomali pencucian uang dengan akurat.

---

## 🛠️ Struktur Repositori & Kode

* [main.py](main.py) : Orkestrator utama pipeline untuk mengeksekusi preprocessing, EDA, pelatihan model, evaluasi, dan plotting secara end-to-end.
* `src/` : Folder berisi modul fungsional proyek:
  * [src/config.py](src/config.py) : Pengaturan konfigurasi, path, dan hyperparameter terpusat.
  * [src/spark_session.py](src/spark_session.py) : Ininisialisasi sesi Apache Spark yang aman, adaptif, dan OOM-safe di OS Windows.
  * [src/preprocessing.py](src/preprocessing.py) : Ingestion CSV mentah, schema validation, data cleaning, dan orkestrasi lengkap preprocessing.
  * [src/feature_engineering.py](src/feature_engineering.py) : Pemisahan data stratified split, rekayasa fitur (velocity, global, graph degree), downsampling kelas normal, fit dan transform Ordinal Encoder.
  * [src/eda.py](src/eda.py) : Analisis eksplorasi data statistik terdistribusi menggunakan Apache Spark SQL dan analisis quantile bucket RBA.
  * [src/train.py](src/train.py) : Skrip pelatihan model machine learning (Random Forest, LightGBM, XGBoost) dengan mekanisme alokasi GPU dan fallback otomatis ke CPU.
  * [src/evaluate.py](src/evaluate.py) : Modul sweep threshold keputusan optimal dan konstruksi tiers dynamic RBA berbasis data.
  * [src/visualize.py](src/visualize.py) : Pembuatan grafik evaluasi kinerja model dan hasil visualisasi EDA.
  * [src/utils.py](src/utils.py) : Fungsi utilitas pembantu (logging, timing, file handler).
* [models/](models/) : Berkas parameter encoding dan bobot model terlatih terbaik yang siap pakai (dalam format biner `.joblib`).
* [plots/](plots/) : Berkas gambar visualisasi kurva evaluasi model dan EDA hasil eksekusi.

---

## 🚀 Cara Menjalankan Pipeline secara Lokal

### 1. Prasyarat (Prerequisites)
* **Python**: Versi 3.8 hingga 3.11.
* **Java**: JDK 8 atau JDK 11 terinstal dan terdaftar di variabel lingkungan `PATH` & `JAVA_HOME` (Apache Spark membutuhkan runtime Java).

### 2. Instalasi Dependensi
Buat virtual environment Python dan pasang pustaka yang diperlukan:
```bash
# Membuat virtual environment
python -m venv .venv

# Mengaktifkan virtual environment (Windows)
.venv\Scripts\activate

# Memasang pustaka dependensi
pip install pyspark pandas numpy lightgbm xgboost scikit-learn joblib matplotlib seaborn
```

### 3. Mempersiapkan Dataset Mentah
1. Unduh dataset **IBM Transactions for Anti-Money Laundering (AML)** dari Kaggle di tautan berikut: [Kaggle Dataset Link](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).
2. Ekstrak berkas zip tersebut dan ambil berkas bernama **`HI-Medium_Trans.csv`**.
3. Letakkan berkas **`HI-Medium_Trans.csv`** langsung di dalam **direktori root** proyek ini (sejajar dengan `main.py`).

### 4. Konfigurasi Hadoop Binaries (Penting: Khusus Pengguna Windows)
Karena Apache Spark membutuhkan binari Hadoop (`winutils.exe` dan `hadoop.dll`) untuk melakukan proses pembacaan/penulisan berkas Parquet di sistem operasi Windows:
1. Unduh berkas `winutils.exe` dan `hadoop.dll` untuk versi Hadoop terkait (seperti Hadoop 3.0.0 atau 3.3.0) dari repositori publik tepercaya (misalnya di [cdarlint/winutils](https://github.com/cdarlint/winutils)).
2. Buat direktori baru bernama `hadoop` di root proyek ini, lalu buat sub-direktori `bin` di dalamnya sehingga terbentuk jalur **`hadoop/bin/`**.
3. Masukkan file `winutils.exe` dan `hadoop.dll` ke dalam folder `hadoop/bin/` tersebut.
*(Catatan: Jika rekan Anda menjalankan proyek ini di sistem operasi Linux atau macOS, langkah konfigurasi Hadoop ini dapat dilewati secara aman).*

### 5. Eksekusi Pipeline Utama
Jalankan file orkestrator utama untuk mengeksekusi seluruh pipeline dari preprocessing hingga visualisasi:
```bash
python main.py
```
> **Catatan**: Pipeline dilengkapi fitur penyimpanan Parquet bertahap. Jika file Parquet hasil pemrosesan sudah terbentuk di `data/processed/`, langkah preprocessing & EDA akan dilewati secara otomatis untuk menghemat waktu eksekusi.

### 6. Menjalankan Modul Mandiri (Opsional)
* **EDA Spark SQL**: Untuk menjalankan query analisis data terdistribusi secara manual:
  ```bash
  python src/eda.py
  ```
* **Evaluasi & Visualisasi Ulang**: Untuk menguji ambang batas baru atau mengubah visualisasi grafik:
  ```bash
  python src/visualize.py
  ```
