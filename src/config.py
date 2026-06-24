"""
config.py — Konfigurasi Terpusat Proyek AML Fraud Detection
Semua konstanta, path, dan hyperparameter dikumpulkan di sini.
Tidak ada magic number yang tersebar di file lain.
"""

# ─────────────────────────────────────────────────────────────
# PATHS (relatif dari root project)
# ─────────────────────────────────────────────────────────────
RAW_CSV_FILENAME    = "HI-Medium_Trans.csv"
PROCESSED_DIR       = "data/processed"
EDA_DIR             = "data/eda_results"
MODELS_DIR          = "models"
PLOTS_DIR           = "plots"
EDA_PLOTS_DIR       = "plots/eda"
LOG_DIR             = "logs"
SPARK_TEMP_DIR      = "spark-temp"
SPARK_WAREHOUSE_DIR = "spark-warehouse"

# ─────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE   = 0.20    # 80% train / 20% test

# ─────────────────────────────────────────────────────────────
# SPARK
# ─────────────────────────────────────────────────────────────
SPARK_APP_NAME        = "FraudDetectionAML"
SPARK_DRIVER_MEMORY   = "6g"
SPARK_EXECUTOR_MEMORY = "6g"
SPARK_SHUFFLE_PARTS   = 8

# ─────────────────────────────────────────────────────────────
# DATASET SCHEMA — mapping nama kolom CSV asli ke nama bersih
# ─────────────────────────────────────────────────────────────
RAW_COLUMN_MAP = {
    "Timestamp"           : "Timestamp",
    "Account"             : "From_Account",
    "Account.1"           : "To_Account",
    "From Bank"           : "From_Bank",
    "To Bank"             : "To_Bank",
    "Amount Paid"         : "Amount_Paid",
    "Payment Currency"    : "Payment_Currency",
    "Amount Received"     : "Amount_Received",
    "Receiving Currency"  : "Receiving_Currency",
    "Payment Format"      : "Payment_Format",
    "Is Laundering"       : "Is_Laundering",
}

# ─────────────────────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────────────────────
TARGET_COL = "Is_Laundering"

CATEGORICAL_COLS = [
    "Payment_Format",
    "Receiving_Currency",
    "Payment_Currency",
    "From_Bank",
    "To_Bank",
]

# Kolom identitas yang tidak masuk model (di-drop sebelum training)
ID_COLS  = ["From_Account", "To_Account", "Timestamp"]

FEATURE_COLS = [
    # Temporal
    "Hour", "Day_of_Week", "Is_Currency_Conversion",
    # Nominal
    "Amount_Paid", "Amount_Received",
    # Velocity 24h
    "sender_tx_count_24h", "sender_amount_sum_24h",
    "receiver_tx_count_24h", "receiver_amount_sum_24h",
    # Global activity
    "sender_tx_count", "sender_avg_amount",
    "receiver_tx_count", "receiver_avg_amount",
    # Graph degree
    "out_degree", "in_degree",
    # Kategorikal (akan di-encode)
    "Payment_Format", "Receiving_Currency", "Payment_Currency",
    "From_Bank", "To_Bank",
]

# ─────────────────────────────────────────────────────────────
# SAMPLING — menangani class imbalance pada training
# ─────────────────────────────────────────────────────────────
IMBALANCE_RATIO = 20   # 1 fraud : 20 normal pada training set

# ─────────────────────────────────────────────────────────────
# EDA
# ─────────────────────────────────────────────────────────────
RBA_N_BUCKETS = 20    # jumlah NTILE bucket untuk fraud rate analysis

# ─────────────────────────────────────────────────────────────
# MODEL — Random Forest
# ─────────────────────────────────────────────────────────────
RF_N_ESTIMATORS     = 300
RF_MAX_DEPTH        = 15
RF_MIN_SAMPLES_LEAF = 20
RF_N_JOBS           = -1

# ─────────────────────────────────────────────────────────────
# MODEL — LightGBM
# ─────────────────────────────────────────────────────────────
LGB_N_ESTIMATORS  = 300
LGB_MAX_DEPTH     = 8
LGB_LEARNING_RATE = 0.05
LGB_NUM_LEAVES    = 31

# ─────────────────────────────────────────────────────────────
# MODEL — XGBoost
# ─────────────────────────────────────────────────────────────
XGB_N_ESTIMATORS     = 300
XGB_MAX_DEPTH        = 8
XGB_LEARNING_RATE    = 0.05
XGB_SUBSAMPLE        = 0.8
XGB_COLSAMPLE_BYTREE = 0.7
XGB_EVAL_METRIC      = "aucpr"

# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────
EVAL_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                   0.7, 0.8, 0.9, 0.95, 0.98, 0.99]

# ─────────────────────────────────────────────────────────────
# RBA — konstruksi tier berbasis data (bukan hardcoded)
# ─────────────────────────────────────────────────────────────
RBA_LOW_MULTIPLIER  = 0.5   # fraud_rate < 0.5× baseline → Low Risk
RBA_HIGH_MULTIPLIER = 2.0   # fraud_rate ≥ 2.0× baseline → High Risk

# Constraint Precision minimum per tier saat threshold sweep
RBA_LOW_MIN_PRECISION  = 0.35   # Low: utamakan presisi (kurangi false alarm)
RBA_HIGH_MIN_PRECISION = 0.10   # High: utamakan recall (tangkap semua fraud)
