"""
eda.py — Modul Exploratory Data Analysis (EDA)
Tanggung jawab:
1. Menjalankan query analisis statistik pada cleaned.parquet (distribusi kelas asli yang imbalanced).
2. Mengekspor hasil EDA ke format JSON (class distribution, amount stats, temporal distribution, dll.).
3. Menghitung fraud rate per quantile bucket menggunakan Bucketizer terdistribusi (OOM-safe).
4. Menghitung korelasi fitur numerik menggunakan Pandas pada train_balanced.parquet (efisien, menghindari O(n²) Spark jobs).
"""
import os
import json
import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, to_date, avg, min as spark_min, max as spark_max, sum as spark_sum
from pyspark.ml.feature import Bucketizer

from src import config
from src.spark_session import get_spark_session
from src.utils import get_logger, timer, ensure_dirs

logger = get_logger("eda")

def run_spark_queries(spark: SparkSession, cleaned_path: str, output_dir: str) -> dict:
    """
    Menjalankan query analisis statistik Spark SQL pada cleaned.parquet.
    """
    logger.info(f"Memuat cleaned data dari: {cleaned_path}...")
    df = spark.read.parquet(cleaned_path)
    
    # Registrasikan sebagai view SQL
    df.createOrReplaceTempView("transactions")
    logger.info("Temporary view 'transactions' berhasil didaftarkan.")
    
    timing = {}
    
    # ── QUERY 1: Distribusi Kelas (Class Imbalance) ──
    logger.info("[EDA Query 1] Menghitung distribusi kelas...")
    t0 = time.time()
    class_dist_df = spark.sql("""
        SELECT
            Is_Laundering,
            COUNT(*) AS jumlah,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 4) AS persentase
        FROM transactions
        GROUP BY Is_Laundering
        ORDER BY Is_Laundering
    """)
    class_dist_rows = class_dist_df.collect()
    timing["q1_class_distribution"] = round(time.time() - t0, 3)
    
    class_dist_res = []
    for row in class_dist_rows:
        label = "Fraud" if row["Is_Laundering"] == 1 else "Normal"
        class_dist_res.append({
            "label": label,
            "is_laundering": int(row["Is_Laundering"]),
            "jumlah": int(row["jumlah"]),
            "persentase": float(row["persentase"])
        })
    with open(os.path.join(output_dir, "class_distribution.json"), "w") as f:
        json.dump(class_dist_res, f, indent=4)
        
    # ── QUERY 2: Statistik Nominal Transaksi per Kelas ──
    logger.info("[EDA Query 2] Menghitung statistik nominal transaksi per kelas...")
    t0 = time.time()
    amount_stats_df = spark.sql("""
        SELECT
            Is_Laundering,
            ROUND(AVG(Amount_Paid), 2)             AS avg_amount,
            ROUND(MIN(Amount_Paid), 2)             AS min_amount,
            ROUND(MAX(Amount_Paid), 2)             AS max_amount,
            ROUND(STDDEV(Amount_Paid), 2)           AS std_amount,
            ROUND(PERCENTILE_APPROX(Amount_Paid, 0.25), 2) AS q1_amount,
            ROUND(PERCENTILE_APPROX(Amount_Paid, 0.50), 2) AS median_amount,
            ROUND(PERCENTILE_APPROX(Amount_Paid, 0.75), 2) AS q3_amount
        FROM transactions
        GROUP BY Is_Laundering
        ORDER BY Is_Laundering
    """)
    amount_stats_rows = amount_stats_df.collect()
    timing["q2_amount_statistics"] = round(time.time() - t0, 3)
    
    amount_stats_res = []
    for row in amount_stats_rows:
        label = "Fraud" if row["Is_Laundering"] == 1 else "Normal"
        amount_stats_res.append({
            "label": label,
            "is_laundering": int(row["Is_Laundering"]),
            "avg_amount": float(row["avg_amount"] or 0),
            "min_amount": float(row["min_amount"] or 0),
            "max_amount": float(row["max_amount"] or 0),
            "std_amount": float(row["std_amount"] or 0),
            "q1_amount":  float(row["q1_amount"] or 0),
            "median_amount": float(row["median_amount"] or 0),
            "q3_amount": float(row["q3_amount"] or 0)
        })
    with open(os.path.join(output_dir, "amount_stats.json"), "w") as f:
        json.dump(amount_stats_res, f, indent=4)

    # ── QUERY 3: Distribusi Metode Pembayaran per Kelas ──
    logger.info("[EDA Query 3] Menghitung distribusi format pembayaran...")
    t0 = time.time()
    payment_df = spark.sql("""
        SELECT
            Is_Laundering,
            Payment_Format,
            COUNT(*) AS jumlah,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY Is_Laundering), 2) AS pct_dalam_kelas
        FROM transactions
        GROUP BY Is_Laundering, Payment_Format
        ORDER BY Is_Laundering, jumlah DESC
    """)
    payment_rows = payment_df.collect()
    timing["q3_payment_format"] = round(time.time() - t0, 3)
    
    payment_res = []
    for row in payment_rows:
        payment_res.append({
            "is_laundering": int(row["Is_Laundering"]),
            "label": "Fraud" if row["Is_Laundering"] == 1 else "Normal",
            "payment_format": str(row["Payment_Format"]),
            "jumlah": int(row["jumlah"]),
            "pct_dalam_kelas": float(row["pct_dalam_kelas"] or 0)
        })
    with open(os.path.join(output_dir, "payment_format_eda.json"), "w") as f:
        json.dump(payment_res, f, indent=4)

    # ── QUERY 4: Timeline Transaksi Fraud Harian ──
    logger.info("[EDA Query 4] Menghitung timeline fraud harian...")
    t0 = time.time()
    df_with_date = df.withColumn("Date", to_date(col("Timestamp")))
    df_with_date.createOrReplaceTempView("transactions_dated")
    
    daily_df = spark.sql("""
        SELECT
            CAST(Date AS STRING)        AS tanggal,
            Is_Laundering,
            COUNT(*)                    AS jumlah_transaksi,
            ROUND(SUM(Amount_Paid), 2)  AS total_amount
        FROM transactions_dated
        WHERE Date IS NOT NULL
        GROUP BY Date, Is_Laundering
        ORDER BY Date, Is_Laundering
    """)
    daily_rows = daily_df.collect()
    timing["q4_daily_timeline"] = round(time.time() - t0, 3)
    
    daily_res = []
    for row in daily_rows:
        daily_res.append({
            "tanggal": str(row["tanggal"]),
            "is_laundering": int(row["Is_Laundering"]),
            "label": "Fraud" if row["Is_Laundering"] == 1 else "Normal",
            "jumlah_transaksi": int(row["jumlah_transaksi"]),
            "total_amount": float(row["total_amount"] or 0)
        })
    with open(os.path.join(output_dir, "daily_fraud_timeline.json"), "w") as f:
        json.dump(daily_res, f, indent=4)

    # ── QUERY 5: Top 15 Akun Pengirim (Graph Behavior) ──
    logger.info("[EDA Query 5] Menghitung akun pengirim paling aktif...")
    t0 = time.time()
    top_accounts_df = spark.sql("""
        SELECT
            From_Account,
            COUNT(*)                    AS tx_count_sent,
            ROUND(SUM(Amount_Paid), 2)  AS total_sent,
            COUNT(DISTINCT To_Account)  AS unique_recipients,
            SUM(Is_Laundering)          AS fraud_tx_count
        FROM transactions
        GROUP BY From_Account
        ORDER BY tx_count_sent DESC
        LIMIT 15
    """)
    top_rows = top_accounts_df.collect()
    timing["q5_top_accounts"] = round(time.time() - t0, 3)
    
    top_accounts_res = []
    for row in top_rows:
        top_accounts_res.append({
            "account": str(row["From_Account"]),
            "tx_count_sent": int(row["tx_count_sent"]),
            "total_sent": float(row["total_sent"] or 0),
            "unique_recipients": int(row["unique_recipients"]),
            "fraud_tx_count": int(row["fraud_tx_count"] or 0)
        })
    with open(os.path.join(output_dir, "top_accounts.json"), "w") as f:
        json.dump(top_accounts_res, f, indent=4)

    # ── QUERY 6: Transaksi Fraud Berdasarkan Jam dalam Sehari ──
    logger.info("[EDA Query 6] Menghitung distribusi jam transaksi...")
    t0 = time.time()
    hourly_df = spark.sql("""
        SELECT
            Hour,
            Is_Laundering,
            COUNT(*) AS jumlah_transaksi
        FROM transactions
        GROUP BY Hour, Is_Laundering
        ORDER BY Hour, Is_Laundering
    """)
    hourly_rows = hourly_df.collect()
    timing["q6_hourly_distribution"] = round(time.time() - t0, 3)
    
    hourly_res = []
    for row in hourly_rows:
        hourly_res.append({
            "hour": int(row["Hour"]),
            "is_laundering": int(row["Is_Laundering"]),
            "label": "Fraud" if row["Is_Laundering"] == 1 else "Normal",
            "jumlah_transaksi": int(row["jumlah_transaksi"])
        })
    with open(os.path.join(output_dir, "hourly_distribution.json"), "w") as f:
        json.dump(hourly_res, f, indent=4)

    return timing

def run_rba_ntile_analysis(df, n_buckets: int, output_path: str) -> None:
    """
    Menghitung fraud rate per quantile bucket secara terdistribusi menggunakan approxQuantile + Bucketizer.
    Menghindari OOM driver yang disebabkan oleh window function NTILE tanpa PARTITION BY.
    """
    logger.info(f"Menghitung {n_buckets} quantile buckets untuk analisis fraud rate RBA...")
    
    # 1. Hitung pembagi quantile (splits) secara terdistribusi
    fractions = [i / float(n_buckets) for i in range(n_buckets + 1)]
    # approxQuantile: toleransi error 0.001
    quantiles = df.approxQuantile("Amount_Paid", fractions, 0.001)
    
    # Buat splits aman dari overflow/underflow
    splits = [-float("inf")] + quantiles[1:-1] + [float("inf")]
    
    # 2. Kelompokkan ke dalam bucket menggunakan Bucketizer
    bucketizer = Bucketizer(splits=splits, inputCol="Amount_Paid", outputCol="bucket_id")
    df_bucketed = bucketizer.transform(df)
    
    # 3. Agregasi statistik per bucket
    fraud_rate_df = df_bucketed.groupBy("bucket_id").agg(
        spark_min("Amount_Paid").alias("bucket_min"),
        spark_max("Amount_Paid").alias("bucket_max"),
        count("*").alias("total_tx"),
        spark_sum("Is_Laundering").alias("fraud_tx")
    ).withColumn("fraud_rate_pct", col("fraud_tx") * 100.0 / col("total_tx")) \
     .orderBy("bucket_id")
     
    rows = fraud_rate_df.collect()
    
    fraud_rate_res = []
    for row in rows:
        fraud_rate_res.append({
            "bucket_id": int(row["bucket_id"] + 1),  # 1-indexed
            "bucket_min": float(row["bucket_min"] or 0),
            "bucket_max": float(row["bucket_max"] or 0),
            "total_tx": int(row["total_tx"]),
            "fraud_tx": int(row["fraud_tx"] or 0),
            "fraud_rate_pct": round(float(row["fraud_rate_pct"] or 0), 6)
        })
        
    with open(output_path, "w") as f:
        json.dump(fraud_rate_res, f, indent=4)
    logger.info(f"Hasil RBA Quantile Buckets disimpan di: {output_path}")

def run_correlation_pandas(train_balanced_path: str, output_path: str) -> None:
    """
    Menghitung matriks korelasi menggunakan Pandas pada train_balanced.parquet.
    Sangat cepat dan efisien.
    """
    logger.info("Menghitung matriks korelasi menggunakan Pandas...")
    
    # Load dataset training seimbang (kecil, ~588K baris)
    df_pandas = pd.read_parquet(train_balanced_path)
    
    key_features = [
        "sender_tx_count_24h", "sender_amount_sum_24h",
        "receiver_tx_count_24h", "receiver_amount_sum_24h",
        "sender_tx_count", "out_degree", "in_degree",
        "Amount_Paid", "Is_Laundering"
    ]
    
    # Filter kolom yang benar-benar ada
    available_cols = [c for c in key_features if c in df_pandas.columns]
    
    corr_df = df_pandas[available_cols].corr()
    corr_dict = corr_df.to_dict()
    
    result = {
        "features": available_cols,
        "matrix": {f1: {f2: float(corr_dict[f1][f2]) for f2 in available_cols} for f1 in available_cols}
    }
    
    ensure_dirs(os.path.dirname(output_path))
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)
    logger.info("Matriks korelasi berhasil disimpan.")

def run_full_eda(parquet_dir: str, eda_results_dir: str) -> None:
    """
    Fungsi orkestrasi lengkap EDA.
    """
    ensure_dirs(eda_results_dir)
    
    cleaned_parquet = os.path.join(parquet_dir, "cleaned.parquet")
    train_balanced_parquet = os.path.join(parquet_dir, "train_balanced.parquet")
    
    # 1. Jalankan query-query Spark SQL utama
    spark = get_spark_session()
    timing_spark = {}
    try:
        # Cek ketersediaan cleaned.parquet
        if not os.path.exists(cleaned_parquet):
            raise FileNotFoundError(
                f"cleaned.parquet tidak ditemukan di {cleaned_parquet}. "
                "Jalankan preprocessing terlebih dahulu!"
            )
            
        with timer("Spark SQL EDA Queries", logger):
            timing_spark = run_spark_queries(spark, cleaned_parquet, eda_results_dir)
            
        # 2. Analisis NTILE Fraud Rate RBA
        with timer("RBA NTILE Fraud Rate Bucket Analysis", logger):
            df_cleaned = spark.read.parquet(cleaned_parquet)
            run_rba_ntile_analysis(df_cleaned, config.RBA_N_BUCKETS, os.path.join(eda_results_dir, "fraud_rate_by_bucket.json"))
            
    finally:
        spark.stop()
        logger.info("Spark Session dihentikan.")
        
    # 3. Hitung korelasi via Pandas (tidak membutuhkan Spark)
    if os.path.exists(train_balanced_parquet):
        with timer("Pandas Correlation Calculation", logger):
            run_correlation_pandas(train_balanced_parquet, os.path.join(eda_results_dir, "correlation_matrix.json"))
    else:
        logger.warning(
            f"train_balanced.parquet tidak ditemukan di {train_balanced_parquet}. "
            "Korelasi matriks dilewati dan perlu dijalankan setelah feature engineering selesai."
        )
        
    # Simpan benchmark timing dummy untuk backward compatibility visualisasi
    timing_spark["load_parquet"] = 0.0
    benchmark_result = {
        "catatan": "Apache Spark SQL EDA timing logs (DuckDB dihapus untuk audit compliance).",
        "spark_seconds": timing_spark,
        "duckdb_seconds": {},
        "total_spark": round(sum(timing_spark.values()), 3),
        "total_duckdb": 0.0
    }
    with open(os.path.join(eda_results_dir, "benchmark_timing.json"), "w") as f:
        json.dump(benchmark_result, f, indent=4)
        
    logger.info("Seluruh alur EDA selesai!")

if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(src_dir, ".."))
    
    p_dir = os.path.join(project_root, config.PROCESSED_DIR)
    e_dir = os.path.join(project_root, config.EDA_DIR)
    
    run_full_eda(p_dir, e_dir)
