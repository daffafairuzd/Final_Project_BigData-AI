"""
feature_engineering.py — Modul Feature Engineering Deteksi AML
Tanggung jawab:
1. Split data secara stratified (80% train / 20% test) SEBELUM menghitung agregasi.
2. Hitung statistik agregasi (velocity & global activity) HANYA dari data train.
3. Terapkan (join) statistik tersebut ke data train dan test.
4. Lakukan downsampling pada data train (rasio 1:20).
5. Fit OrdinalEncoder pada data train, simpan, dan transform data train + test.
"""
import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, date_sub, count, avg
import pyspark.sql.functions as F

from src import config
from src.spark_session import get_spark_session
from src.utils import get_logger, timer, ensure_dirs

logger = get_logger("feature_engineering")

def stratified_spark_split(df, target_col: str, test_size: float, seed: int):
    """
    Membagi DataFrame secara stratified berdasarkan target_col.
    """
    logger.info(f"Melakukan stratified split dengan test_size={test_size}...")
    fraud_df = df.filter(col(target_col) == 1)
    normal_df = df.filter(col(target_col) == 0)
    
    train_fraud, test_fraud = fraud_df.randomSplit([1.0 - test_size, test_size], seed=seed)
    train_normal, test_normal = normal_df.randomSplit([1.0 - test_size, test_size], seed=seed)
    
    train_df = train_fraud.union(train_normal)
    test_df = test_fraud.union(test_normal)
    
    return train_df, test_df

def compute_velocity_features(train_df):
    """
    Menghitung daily send/receive aggregates dari data train saja.
    """
    logger.info("Menghitung agregasi daily velocity dari data train...")
    train_df_with_date = train_df.withColumn("Date", to_date(col("Timestamp")))
    
    daily_send = train_df_with_date.groupBy("From_Account", "Date").agg(
        count("Timestamp").alias("daily_send_count"),
        F.sum("Amount_Paid").alias("daily_send_amount")
    ).cache()
    
    daily_recv = train_df_with_date.groupBy("To_Account", "Date").agg(
        count("Timestamp").alias("daily_recv_count"),
        F.sum("Amount_Received").alias("daily_recv_amount")
    ).cache()
    
    return daily_send, daily_recv

def compute_global_features(train_df):
    """
    Menghitung statistik global dan graph degree dari data train saja.
    """
    logger.info("Menghitung statistik global dan graph degree dari data train...")
    
    sender_stats = train_df.groupBy("From_Account").agg(
        count("Timestamp").alias("sender_tx_count"),
        avg("Amount_Paid").alias("sender_avg_amount")
    ).cache()
    
    receiver_stats = train_df.groupBy("To_Account").agg(
        count("Timestamp").alias("receiver_tx_count"),
        avg("Amount_Received").alias("receiver_avg_amount")
    ).cache()
    
    out_degree = train_df.groupBy("From_Account", "To_Account").count() \
                         .groupBy("From_Account").agg(count("To_Account").alias("out_degree")).cache()
                         
    in_degree = train_df.groupBy("From_Account", "To_Account").count() \
                        .groupBy("To_Account").agg(count("From_Account").alias("in_degree")).cache()
                        
    return sender_stats, receiver_stats, out_degree, in_degree

def join_features(df, daily_send, daily_recv, sender_stats, receiver_stats, out_degree, in_degree):
    """
    Menggabungkan fitur lookup ke DataFrame target secara aman untuk menghindari kolusi nama kolom.
    """
    df = df.withColumn("Date", to_date(col("Timestamp")))
    df = df.withColumn("Prev_Date", date_sub(col("Date"), 1))
    
    # 1. Join daily_send untuk Hari Ini
    df = df.join(
        daily_send.withColumnRenamed("From_Account", "lookup_send_from")
                  .withColumnRenamed("Date", "lookup_send_date")
                  .withColumnRenamed("daily_send_count", "send_count_today")
                  .withColumnRenamed("daily_send_amount", "send_amount_today"),
        (col("From_Account") == col("lookup_send_from")) & (col("Date") == col("lookup_send_date")),
        how="left"
    ).drop("lookup_send_from", "lookup_send_date")
    
    # 2. Join daily_send untuk Kemarin
    df = df.join(
        daily_send.withColumnRenamed("From_Account", "lookup_send_from")
                  .withColumnRenamed("Date", "lookup_send_date")
                  .withColumnRenamed("daily_send_count", "send_count_yesterday")
                  .withColumnRenamed("daily_send_amount", "send_amount_yesterday"),
        (col("From_Account") == col("lookup_send_from")) & (col("Prev_Date") == col("lookup_send_date")),
        how="left"
    ).drop("lookup_send_from", "lookup_send_date")
    
    # 3. Join daily_recv untuk Hari Ini
    df = df.join(
        daily_recv.withColumnRenamed("To_Account", "lookup_recv_to")
                  .withColumnRenamed("Date", "lookup_recv_date")
                  .withColumnRenamed("daily_recv_count", "recv_count_today")
                  .withColumnRenamed("daily_recv_amount", "recv_amount_today"),
        (col("To_Account") == col("lookup_recv_to")) & (col("Date") == col("lookup_recv_date")),
        how="left"
    ).drop("lookup_recv_to", "lookup_recv_date")
    
    # 4. Join daily_recv untuk Kemarin
    df = df.join(
        daily_recv.withColumnRenamed("To_Account", "lookup_recv_to")
                  .withColumnRenamed("Date", "lookup_recv_date")
                  .withColumnRenamed("daily_recv_count", "recv_count_yesterday")
                  .withColumnRenamed("daily_recv_amount", "recv_amount_yesterday"),
        (col("To_Account") == col("lookup_recv_to")) & (col("Prev_Date") == col("lookup_recv_date")),
        how="left"
    ).drop("lookup_recv_to", "lookup_recv_date")
    
    # Isi null untuk 24h features
    df = df.fillna({
        "send_count_today": 0, "send_amount_today": 0.0,
        "send_count_yesterday": 0, "send_amount_yesterday": 0.0,
        "recv_count_today": 0, "recv_amount_today": 0.0,
        "recv_count_yesterday": 0, "recv_amount_yesterday": 0.0
    })
    
    # Agregasi 24h
    df = df.withColumn("sender_tx_count_24h", col("send_count_today") + col("send_count_yesterday"))
    df = df.withColumn("sender_amount_sum_24h", col("send_amount_today") + col("send_amount_yesterday"))
    df = df.withColumn("receiver_tx_count_24h", col("recv_count_today") + col("recv_count_yesterday"))
    df = df.withColumn("receiver_amount_sum_24h", col("recv_amount_today") + col("recv_amount_yesterday"))
    
    # Hapus kolom antara & kolom tanggal temporer
    df = df.drop(
        "Date", "Prev_Date",
        "send_count_today", "send_amount_today",
        "send_count_yesterday", "send_amount_yesterday",
        "recv_count_today", "recv_amount_today",
        "recv_count_yesterday", "recv_amount_yesterday"
    )
    
    # 5. Join global sender stats
    df = df.join(
        sender_stats.withColumnRenamed("From_Account", "lookup_sender"),
        col("From_Account") == col("lookup_sender"),
        how="left"
    ).drop("lookup_sender")
    
    # 6. Join global receiver stats
    df = df.join(
        receiver_stats.withColumnRenamed("To_Account", "lookup_receiver"),
        col("To_Account") == col("lookup_receiver"),
        how="left"
    ).drop("lookup_receiver")
    
    # 7. Join out degree
    df = df.join(
        out_degree.withColumnRenamed("From_Account", "lookup_out_deg"),
        col("From_Account") == col("lookup_out_deg"),
        how="left"
    ).drop("lookup_out_deg")
    
    # 8. Join in degree
    df = df.join(
        in_degree.withColumnRenamed("To_Account", "lookup_in_deg"),
        col("To_Account") == col("lookup_in_deg"),
        how="left"
    ).drop("lookup_in_deg")
    
    # Isi null untuk global features
    df = df.fillna({
        "sender_tx_count": 0, "sender_avg_amount": 0.0,
        "receiver_tx_count": 0, "receiver_avg_amount": 0.0,
        "out_degree": 0, "in_degree": 0
    })
    
    return df

def downsample_for_training(train_features_df, ratio: int, seed: int):
    """
    Downsampling data normal agar seimbang dengan target ratio 1:ratio.
    """
    logger.info(f"Melakukan downsampling pada training features dengan rasio 1:{ratio}...")
    fraud_df = train_features_df.filter(col("Is_Laundering") == 1).cache()
    normal_df = train_features_df.filter(col("Is_Laundering") == 0)
    
    num_train_fraud = fraud_df.count()
    num_train_normal = normal_df.count()
    
    target_normal_count = num_train_fraud * ratio
    sample_fraction = min(target_normal_count / num_train_normal, 1.0)
    
    logger.info(f"Downsampling normal fraction: {sample_fraction:.6f} ({target_normal_count} target normal rows)")
    
    train_normal_sampled = normal_df.sample(withReplacement=False, fraction=sample_fraction, seed=seed)
    train_balanced = fraud_df.union(train_normal_sampled)
    
    return train_balanced

def fit_and_apply_encoding(train_input_path: str, test_input_path: str,
                           train_output_path: str, test_output_path: str,
                           cat_cols: list, encoder_output_path: str):
    """
    Melakukan fitting OrdinalEncoder pada data training saja, menyimpannya,
    dan menerapkannya pada data training dan testing menggunakan Pandas.
    """
    import pandas as pd
    from sklearn.preprocessing import OrdinalEncoder
    import joblib
    
    logger.info("Memuat data ke Pandas untuk proses Ordinal Encoding...")
    train_df = pd.read_parquet(train_input_path)
    test_df = pd.read_parquet(test_input_path)
    
    logger.info(f"Melakukan fit OrdinalEncoder pada kolom: {cat_cols}")
    # handle_unknown='use_encoded_value' untuk menangani kategori baru di test set secara aman
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    encoder.fit(train_df[cat_cols])
    
    os.makedirs(os.path.dirname(encoder_output_path), exist_ok=True)
    joblib.dump(encoder, encoder_output_path)
    logger.info(f"Encoder berhasil disimpan ke: {encoder_output_path}")
    
    logger.info("Transformasi data train & test menggunakan encoder...")
    train_df[cat_cols] = encoder.transform(train_df[cat_cols])
    test_df[cat_cols] = encoder.transform(test_df[cat_cols])
    
    logger.info("Menyimpan kembali data ter-encode ke berkas final...")
    
    # Hapus file/folder output jika sudah ada untuk menghindari konflik
    for path in [train_output_path, test_output_path]:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
                
    train_df.to_parquet(train_output_path, index=False)
    test_df.to_parquet(test_output_path, index=False)
    logger.info("Ordinal Encoding dan penyimpanan berkas final selesai.")

def run_feature_engineering(spark: SparkSession, cleaned_path: str, output_dir: str, seed: int, ratio: int) -> dict:
    """
    Orkestrasi Feature Engineering.
    """
    ensure_dirs(output_dir)
    
    # Path output temporer Spark
    train_temp = os.path.join(output_dir, "train_balanced_temp.parquet")
    test_temp = os.path.join(output_dir, "test_features_temp.parquet")
    
    # Path output akhir
    train_final = os.path.join(output_dir, "train_balanced.parquet")
    test_final = os.path.join(output_dir, "test_features.parquet")
    encoder_path = os.path.join(output_dir, "feature_encoders.joblib")
    
    logger.info(f"Memuat cleaned data dari: {cleaned_path}")
    df_cleaned = spark.read.parquet(cleaned_path)
    
    # 1. Stratified Split
    train_raw, test_raw = stratified_spark_split(df_cleaned, config.TARGET_COL, config.TEST_SIZE, seed)
    
    # Cache split awal agar join lebih cepat
    train_raw = train_raw.cache()
    test_raw = test_raw.cache()
    
    # 2. Hitung aggregasi hanya pada train_raw
    daily_send, daily_recv = compute_velocity_features(train_raw)
    sender_stats, receiver_stats, out_degree, in_degree = compute_global_features(train_raw)
    
    # 3. Join lookup ke data train dan test secara terpisah
    logger.info("Menggabungkan lookup fitur ke data train...")
    train_features = join_features(train_raw, daily_send, daily_recv, sender_stats, receiver_stats, out_degree, in_degree)
    
    logger.info("Menggabungkan lookup fitur ke data test...")
    test_features = join_features(test_raw, daily_send, daily_recv, sender_stats, receiver_stats, out_degree, in_degree)
    
    # 4. Downsampling data train
    train_balanced = downsample_for_training(train_features, ratio, seed)
    
    # 5. Tulis hasil Spark ke folder temporer Parquet
    logger.info("Menyimpan dataset train balanced dan test features (temporer)...")
    with timer("Penyimpanan dataset fitur Spark", logger):
        train_balanced.write.mode("overwrite").parquet(train_temp)
        test_features.write.mode("overwrite").parquet(test_temp)
        
    # Hapus cache
    daily_send.unpersist()
    daily_recv.unpersist()
    sender_stats.unpersist()
    receiver_stats.unpersist()
    out_degree.unpersist()
    in_degree.unpersist()
    train_raw.unpersist()
    test_raw.unpersist()
    
    # 6. Fit & apply encoder dengan Pandas
    with timer("Proses Ordinal Encoding", logger):
        fit_and_apply_encoding(
            train_input_path=train_temp,
            test_input_path=test_temp,
            train_output_path=train_final,
            test_output_path=test_final,
            cat_cols=config.CATEGORICAL_COLS,
            encoder_output_path=encoder_path
        )
        
    # Hapus folder temp
    logger.info("Membersihkan folder temporer...")
    for temp_path in [train_temp, test_temp]:
        if os.path.exists(temp_path) and os.path.isdir(temp_path):
            shutil.rmtree(temp_path)
            
    logger.info("Seluruh alur Feature Engineering selesai dengan sukses!")
    
    return {
        "train_balanced": train_final,
        "test_features": test_final,
        "encoder": encoder_path
    }

if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(src_dir, ".."))
    
    cleaned_parquet = os.path.join(project_root, config.PROCESSED_DIR, "cleaned.parquet")
    output_folder = os.path.join(project_root, config.PROCESSED_DIR)
    
    spark = get_spark_session()
    try:
        run_feature_engineering(
            spark, 
            cleaned_parquet, 
            output_folder, 
            seed=config.RANDOM_SEED, 
            ratio=config.IMBALANCE_RATIO
        )
    except Exception as e:
        logger.error(f"Feature Engineering Gagal: {str(e)}", exc_info=True)
    finally:
        spark.stop()
        logger.info("Spark Session dihentikan.")
