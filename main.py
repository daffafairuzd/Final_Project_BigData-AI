"""
main.py — Orkestrator Utama AML Fraud Detection Pipeline
Tanggung jawab:
Menjalankan seluruh alur proyek secara end-to-end dengan skema modular:
[Step 1] Preprocessing & Feature Engineering (Spark, fix leakage)
[Step 2] Exploratory Data Analysis (Spark SQL + approxQuantile RBA buckets)
[Step 3] Pelatihan Model AI (RF Baseline + LightGBM + XGBoost)
[Step 4] Evaluasi Model AI (Standardized Metrics + Data-Driven RBA)
[Step 5] Visualisasi (Model Evaluation + Spark SQL EDA plots)
"""
import os
import sys

# Tambahkan project directory ke PYTHONPATH pencarian import Python
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from src import config
from src.spark_session import get_spark_session
from src.preprocess import preprocess_data
from src.eda import run_full_eda
from src.train import train_all_models
from src.evaluate import run_evaluation
from src.visualize import generate_visualizations, generate_eda_visualizations
from src.utils import get_logger, ensure_dirs

logger = get_logger("main_pipeline")

def main():
    logger.info("======================================================================")
    logger.info("      PIPELINE SISTEM DETEKSI FRAUD (MONEY LAUNDERING) - BIG DATA     ")
    logger.info("======================================================================\n")
    
    # 1. Definisikan folder dan file paths
    processed_dir = os.path.join(project_dir, config.PROCESSED_DIR)
    eda_dir = os.path.join(project_dir, config.EDA_DIR)
    models_dir = os.path.join(project_dir, config.MODELS_DIR)
    plots_dir = os.path.join(project_dir, config.PLOTS_DIR)
    
    csv_file = os.path.join(project_dir, config.RAW_CSV_FILENAME)
    
    ensure_dirs(processed_dir, eda_dir, models_dir, plots_dir)
    
    # Check outputs yang ada
    train_balanced = os.path.join(processed_dir, "train_balanced.parquet")
    test_features = os.path.join(processed_dir, "test_features.parquet")
    
    # ──────────────────────────────────────────────────────────────────────────
    # [LANGKAH 1] PREPROCESSING & FEATURE ENGINEERING
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("[LANGKAH 1] Preprocessing & Feature Engineering (Apache Spark)...")
    
    # Check apakah sudah pernah diproses untuk skip
    if os.path.exists(train_balanced) and os.path.exists(test_features):
        logger.info("-> File Parquet hasil preprocessing & feature engineering sudah ada.")
        logger.info("-> Melewati tahap ini untuk menghemat waktu (Skip).")
    else:
        logger.info("-> Dataset fitur Parquet tidak ditemukan. Menjalankan pipeline Spark...")
        spark = get_spark_session()
        try:
            preprocess_data(spark, csv_file, processed_dir)
        except Exception as e:
            logger.error(f"Gagal pada tahap Preprocessing & Feature Engineering: {str(e)}", exc_info=True)
            sys.exit(1)
        finally:
            spark.stop()
            logger.info("Spark Session dihentikan.\n")
            
    # ──────────────────────────────────────────────────────────────────────────
    # [LANGKAH 2] EXPLORATORY DATA ANALYSIS (Spark SQL)
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("[LANGKAH 2] Exploratory Data Analysis (EDA) dengan Apache Spark SQL...")
    
    fraud_buckets_path = os.path.join(eda_dir, "fraud_rate_by_bucket.json")
    if os.path.exists(fraud_buckets_path):
        logger.info("-> Hasil EDA dan RBA buckets sudah ada di data/eda_results/.")
        logger.info("-> Melewati tahap query Spark SQL EDA (Skip).")
    else:
        logger.info("-> Menjalankan analisis Spark SQL EDA pada cleaned.parquet...")
        try:
            run_full_eda(processed_dir, eda_dir)
        except Exception as e:
            logger.error(f"Gagal pada tahap EDA: {str(e)}", exc_info=True)
            sys.exit(1)
            
    # ──────────────────────────────────────────────────────────────────────────
    # [LANGKAH 3] PELATIHAN MODEL AI (RF, LightGBM, XGBoost)
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("[LANGKAH 3] Pelatihan Model AI (RF Baseline + LightGBM + XGBoost)...")
    try:
        train_all_models(processed_dir, models_dir)
    except Exception as e:
        logger.error(f"Gagal pada tahap Pelatihan Model AI: {str(e)}", exc_info=True)
        sys.exit(1)
        
    # ──────────────────────────────────────────────────────────────────────────
    # [LANGKAH 4] EVALUASI MODEL AI (Standardized Metrics & RBA Tiers)
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("[LANGKAH 4] Evaluasi Model AI & Kalibrasi Data-Driven RBA...")
    try:
        run_evaluation(processed_dir, models_dir, eda_dir)
    except Exception as e:
        logger.error(f"Gagal pada tahap Evaluasi Model AI: {str(e)}", exc_info=True)
        sys.exit(1)

    # ──────────────────────────────────────────────────────────────────────────
    # [LANGKAH 5] PEMBUATAN GRAFIK VISUALISASI
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("[LANGKAH 5] Pembuatan Grafik Visualisasi (Evaluasi & EDA)...")
    try:
        # Generate plot EDA
        generate_eda_visualizations(eda_dir, plots_dir)
        # Generate plot model
        generate_visualizations(processed_dir, models_dir, plots_dir, test_features)
    except Exception as e:
        logger.error(f"Gagal pada tahap Pembuatan Grafik: {str(e)}", exc_info=True)
        sys.exit(1)
        
    logger.info("\n======================================================================")
    logger.info(" PIPELINE BERHASIL DISAJIKAN SECARA END-TO-END!                       ")
    logger.info(" -> Grafik model evaluation tersimpan di folder 'plots/'             ")
    logger.info(" -> Grafik EDA tersimpan di folder 'plots/eda/'                      ")
    logger.info(" -> Model tersimpan di folder 'models/'                              ")
    logger.info(" -> Hasil EDA Spark tersimpan di 'data/eda_results/'                 ")
    logger.info("======================================================================\n")

if __name__ == "__main__":
    main()
