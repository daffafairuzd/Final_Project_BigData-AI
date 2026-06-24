"""
preprocess.py — Wrapper untuk kompatibilitas pipeline lama.
Mengimpor dan menyatukan fungsi-fungsi baru dari spark_session.py,
preprocessing.py, dan feature_engineering.py.
"""
from src.spark_session import get_spark_session
from src.preprocessing import run_preprocessing
from src.feature_engineering import run_feature_engineering
from src import config
from src.utils import get_logger

logger = get_logger("preprocess_legacy_wrapper")

def init_spark():
    """
    Inisialisasi SparkSession menggunakan spark_session.py.
    """
    return get_spark_session()

def preprocess_data(spark, csv_path, output_dir):
    """
    Orkestrasi preprocessing baru (cleaning + feature engineering tanpa leakage).
    """
    logger.info("--- [Wrapper] Memulai Preprocessing & Feature Engineering Modular ---")
    
    # Langkah 1: Preprocessing (cleaning & row-level features) -> cleaned.parquet
    cleaned_parquet_path = run_preprocessing(spark, csv_path, output_dir)
    
    # Langkah 2: Feature Engineering (split-first, aggregates from train only)
    run_feature_engineering(
        spark=spark,
        cleaned_path=cleaned_parquet_path,
        output_dir=output_dir,
        seed=config.RANDOM_SEED,
        ratio=config.IMBALANCE_RATIO
    )
    
    logger.info("--- [Wrapper] Seluruh Preprocessing & Feature Engineering Selesai ---")

if __name__ == "__main__":
    import os
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(src_dir, ".."))
    
    csv_file = os.path.join(project_root, config.RAW_CSV_FILENAME)
    output_folder = os.path.join(project_root, config.PROCESSED_DIR)
    
    spark = init_spark()
    try:
        preprocess_data(spark, csv_file, output_folder)
    finally:
        spark.stop()
        logger.info("Spark Session dihentikan.")
