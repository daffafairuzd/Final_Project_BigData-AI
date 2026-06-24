"""
preprocessing.py — Modul Preprocessing Data AML
Tanggung jawab: Ingestion raw CSV, cleaning, row-level feature creation, schema validation,
dan penyimpanan ke cleaned.parquet.
(Tidak ada feature engineering berbasis agregasi/grafik di sini untuk mencegah feature leakage).
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, hour, dayofweek, when
import pyspark.sql.types as T

from src import config
from src.spark_session import get_spark_session
from src.utils import get_logger, timer, ensure_dirs

logger = get_logger("preprocessing")

def validate_raw_schema(df) -> None:
    """
    Validasi bahwa kolom-kolom utama yang dibutuhkan ada dalam DataFrame mentah.
    """
    cols = df.columns
    # Cek minimal beberapa kolom kunci untuk memastikan CSV valid
    required_keywords = ["Timestamp", "Bank", "Account", "Amount", "Currency", "Format", "Laundering"]
    
    found = False
    for req in required_keywords:
        if any(req in c for c in cols):
            found = True
            break
            
    if not found:
        raise ValueError(
            f"Schema CSV tidak valid. Kolom yang ditemukan: {cols}. "
            f"Dibutuhkan kolom-kolom transaksi standard IBM AML."
        )
    logger.info("Verifikasi skema data mentah berhasil.")

def clean_data(df) -> SparkSession:
    """
    Melakukan rename kolom, parsing timestamp, casting tipe data, dan membuat fitur level baris.
    """
    logger.info("Memulai pembersihan data dan pembuatan fitur level baris...")
    
    # 1. Rename kolom secara dinamis dan aman
    columns = df.columns
    
    # Mengatasi variasi penamaan duplicate column 'Account'
    if "Account2" in columns:
        df = df.withColumnRenamed("Account2", "From_Account")
    if "Account4" in columns:
        df = df.withColumnRenamed("Account4", "To_Account")
    if "Account" in columns:
        df = df.withColumnRenamed("Account", "From_Account")
    if "Account.1" in columns:
        df = df.withColumnRenamed("Account.1", "To_Account")
    if "Account_1" in columns:
        df = df.withColumnRenamed("Account_1", "To_Account")
        
    # Mapping sisanya dari config
    for raw_col, clean_col in config.RAW_COLUMN_MAP.items():
        # Jangan timpa jika sudah di-rename di atas
        if raw_col in df.columns and clean_col not in df.columns:
            df = df.withColumnRenamed(raw_col, clean_col)
            
    # Pastikan kolom From_Account dan To_Account sudah ter-rename
    if "From_Account" not in df.columns or "To_Account" not in df.columns:
        # Fallback manual jika format kolom di luar perkiraan
        remap = {}
        account_cols = [c for c in df.columns if "Account" in c]
        if len(account_cols) >= 2:
            remap[account_cols[0]] = "From_Account"
            remap[account_cols[1]] = "To_Account"
            for k, v in remap.items():
                df = df.withColumnRenamed(k, v)
                logger.warning(f"Fallback rename duplicate account column: {k} -> {v}")

    # 2. Parsing tipe data & Pembuatan fitur temporal
    logger.info("Mengonversi Timestamp ke format datetime dan membuat fitur temporal...")
    df = df.withColumn("Timestamp", to_timestamp(col("Timestamp"), "yyyy/MM/dd HH:mm"))
    df = df.withColumn("Hour", hour(col("Timestamp")).cast(T.IntegerType()))
    df = df.withColumn("Day_of_Week", dayofweek(col("Timestamp")).cast(T.IntegerType()))
    
    # 3. Fitur konversi mata uang (Currency Conversion)
    logger.info("Membuat fitur konversi mata uang...")
    df = df.withColumn("Is_Currency_Conversion", 
                       when(col("Receiving_Currency") != col("Payment_Currency"), 1).otherwise(0).cast(T.IntegerType()))
    
    # 4. Cast tipe data numerik utama ke Double/Integer
    logger.info("Casting kolom numerik (Amount_Paid, Amount_Received, Is_Laundering)...")
    df = df.withColumn("Amount_Paid", col("Amount_Paid").cast(T.DoubleType()))
    df = df.withColumn("Amount_Received", col("Amount_Received").cast(T.DoubleType()))
    df = df.withColumn("Is_Laundering", col("Is_Laundering").cast(T.IntegerType()))
    
    # Drop kolom temporer jika ada yang tidak terpakai
    return df

def save_cleaned_parquet(df, output_path: str) -> None:
    """
    Menyimpan DataFrame yang dibersihkan ke format Parquet.
    """
    logger.info(f"Menyimpan data yang telah dibersihkan ke format Parquet di: {output_path} ...")
    df.write.mode("overwrite").parquet(output_path)
    logger.info("Penyimpanan cleaned.parquet selesai.")

def run_preprocessing(spark: SparkSession, csv_path: str, output_dir: str) -> str:
    """
    Fungsi orkestrasi preprocessing.
    """
    ensure_dirs(output_dir)
    cleaned_output_path = os.path.join(output_dir, "cleaned.parquet")
    
    logger.info(f"Membaca file data mentah dari: {csv_path}")
    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    
    validate_raw_schema(df)
    
    df_cleaned = clean_data(df)
    
    with timer("Pembersihan dan Penyimpanan Parquet", logger):
        save_cleaned_parquet(df_cleaned, cleaned_output_path)
        
    return cleaned_output_path

if __name__ == "__main__":
    # Script entry point untuk eksekusi mandiri
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(src_dir, ".."))
    
    csv_file = os.path.join(project_root, config.RAW_CSV_FILENAME)
    output_folder = os.path.join(project_root, config.PROCESSED_DIR)
    
    spark = get_spark_session()
    try:
        run_preprocessing(spark, csv_file, output_folder)
    except Exception as e:
        logger.error(f"Preprocessing Gagal: {str(e)}", exc_info=True)
    finally:
        spark.stop()
        logger.info("Spark Session dihentikan.")
