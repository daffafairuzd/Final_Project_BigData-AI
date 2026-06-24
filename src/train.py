"""
train.py — Modul Pelatihan Model AI
Tanggung jawab:
1. Memuat dataset training seimbang dan dataset testing.
2. Melatih 3 model (Random Forest, LightGBM, XGBoost) pada dataset yang sama persis.
3. Menyertakan mekanisme akselerasi GPU dengan fallback otomatis ke CPU jika terjadi kendala driver.
4. Menyimpan model yang dilatih ke folder models/.
"""
import os
import time
import json
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

# Mengimpor modul baru setelah instalasi selesai
try:
    import xgboost as xgb
except ImportError:
    xgb = None

from src import config
from src.utils import get_logger, ensure_dirs, get_scale_pos_weight

logger = get_logger("train")

def load_datasets(data_dir: str):
    """
    Memuat dataset train dan test ter-encode, dan memisahkannya menjadi X dan y.
    """
    train_path = os.path.join(data_dir, "train_balanced.parquet")
    test_path = os.path.join(data_dir, "test_features.parquet")
    
    logger.info(f"Memuat data latih dari: {train_path}")
    df_train = pd.read_parquet(train_path)
    
    logger.info(f"Memuat data uji dari: {test_path}")
    df_test = pd.read_parquet(test_path)
    
    logger.info(f"Dimensi Train: {df_train.shape[0]:,} baris | Dimensi Test: {df_test.shape[0]:,} baris")
    
    # Pisahkan label dan fitur identitas yang di-drop
    features_to_drop = config.ID_COLS + [config.TARGET_COL]
    
    X_train = df_train.drop(columns=features_to_drop, errors="ignore")
    y_train = df_train[config.TARGET_COL].astype(int)
    
    X_test = df_test.drop(columns=features_to_drop, errors="ignore")
    y_test = df_test[config.TARGET_COL].astype(int)
    
    feature_names = X_train.columns.tolist()
    logger.info(f"Fitur yang digunakan ({len(feature_names)}): {feature_names}")
    
    return X_train, y_train, X_test, y_test, feature_names

def train_rf(X_train, y_train, scale_pos_weight: float) -> RandomForestClassifier:
    """
    Melatih model Random Forest Baseline.
    """
    logger.info("--- Melatih Random Forest Baseline ---")
    class_weight = {0: 1.0, 1: scale_pos_weight}
    
    rf = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
        class_weight=class_weight,
        n_jobs=config.RF_N_JOBS,
        random_state=config.RANDOM_SEED
    )
    
    t0 = time.perf_counter()
    rf.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    logger.info(f"Random Forest selesai dilatih dalam {elapsed:.2f} detik.")
    return rf, elapsed

def train_lgb(X_train, y_train, scale_pos_weight: float, models_dir: str) -> lgb.LGBMClassifier:
    """
    Melatih model LightGBM dengan deteksi parameter terbaik dan fallback GPU -> CPU.
    """
    logger.info("--- Melatih LightGBM Classifier ---")
    
    # Default parameters
    lgb_params = {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 8,
        'scale_pos_weight': scale_pos_weight,
        'random_state': config.RANDOM_SEED,
        'verbose': -1,
        'n_jobs': -1
    }
    
    # Load parameters hasil tuning Optuna jika ada
    params_path = os.path.join(models_dir, "best_lgb_params.json")
    if os.path.exists(params_path):
        try:
            logger.info(f"Memuat best LightGBM params dari {params_path}...")
            with open(params_path, "r") as f:
                tuned_params = json.load(f)
            # Konversi float ke int untuk hyperparameter yang sesuai
            for k, v in tuned_params.items():
                if k in ['n_estimators', 'num_leaves', 'max_depth', 'min_child_samples']:
                    tuned_params[k] = int(v)
            lgb_params.update(tuned_params)
        except Exception as e:
            logger.warning(f"Gagal memuat {params_path}, menggunakan parameter bawaan. Error: {str(e)}")
            
    t0 = time.perf_counter()
    model = None
    
    # Coba GPU (OpenCL/CUDA)
    try:
        logger.info("Mencoba melatih LightGBM menggunakan GPU (RTX 3050)...")
        gpu_params = lgb_params.copy()
        gpu_params['device'] = 'gpu'
        model = lgb.LGBMClassifier(**gpu_params)
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        logger.info(f"LightGBM GPU training berhasil dalam {elapsed:.2f} detik.")
    except Exception as e:
        logger.warning(f"LightGBM GPU training gagal/tidak didukung: {str(e)}")
        logger.info("Melakukan fallback otomatis LightGBM ke CPU...")
        t0_cpu = time.perf_counter()
        cpu_params = lgb_params.copy()
        cpu_params['device'] = 'cpu'
        model = lgb.LGBMClassifier(**cpu_params)
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0_cpu
        logger.info(f"LightGBM CPU training berhasil dalam {elapsed:.2f} detik.")
        
    return model, elapsed

def train_xgb(X_train, y_train, scale_pos_weight: float) -> xgb.XGBClassifier:
    """
    Melatih model XGBoost dengan parameter terpusat dan fallback GPU -> CPU.
    """
    logger.info("--- Melatih XGBoost Classifier ---")
    if xgb is None:
        raise ImportError("XGBoost tidak terinstal di environment virtual ini!")
        
    xgb_params = {
        'n_estimators': config.XGB_N_ESTIMATORS,
        'max_depth': config.XGB_MAX_DEPTH,
        'learning_rate': config.XGB_LEARNING_RATE,
        'subsample': config.XGB_SUBSAMPLE,
        'colsample_bytree': config.XGB_COLSAMPLE_BYTREE,
        'scale_pos_weight': scale_pos_weight,
        'eval_metric': config.XGB_EVAL_METRIC,
        'random_state': config.RANDOM_SEED,
        'n_jobs': -1
    }
    
    t0 = time.perf_counter()
    model = None
    
    # Coba GPU (CUDA)
    try:
        logger.info("Mencoba melatih XGBoost menggunakan GPU (CUDA)...")
        gpu_params = xgb_params.copy()
        gpu_params['device'] = 'cuda'
        model = xgb.XGBClassifier(**gpu_params)
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        logger.info(f"XGBoost GPU training berhasil dalam {elapsed:.2f} detik.")
    except Exception as e:
        logger.warning(f"XGBoost GPU training gagal/tidak didukung: {str(e)}")
        logger.info("Melakukan fallback otomatis XGBoost ke CPU (hist)...")
        t0_cpu = time.perf_counter()
        cpu_params = xgb_params.copy()
        cpu_params['device'] = 'cpu'
        cpu_params['tree_method'] = 'hist'
        model = xgb.XGBClassifier(**cpu_params)
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0_cpu
        logger.info(f"XGBoost CPU training berhasil dalam {elapsed:.2f} detik.")
        
    return model, elapsed

def train_all_models(data_dir: str, models_dir: str) -> dict:
    """
    Fungsi orkestrasi utama untuk memuat data, melatih 3 model, dan menyimpan hasilnya.
    """
    ensure_dirs(models_dir)
    
    # 1. Load Data
    X_train, y_train, X_test, y_test, feature_names = load_datasets(data_dir)
    
    # Simpan nama fitur
    with open(os.path.join(models_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)
    logger.info("Nama fitur disimpan ke feature_names.json")
    
    # Hitung scale_pos_weight otomatis
    spw = get_scale_pos_weight(y_train)
    logger.info(f"Dihitung scale_pos_weight untuk penyeimbangan kelas: {spw:.4f}")
    
    durations = {}
    
    # 2. Train Random Forest
    rf_model, rf_time = train_rf(X_train, y_train, spw)
    joblib.dump(rf_model, os.path.join(models_dir, "rf_model.joblib"))
    durations["RandomForest"] = rf_time
    
    # 3. Train LightGBM
    lgb_model, lgb_time = train_lgb(X_train, y_train, spw, models_dir)
    # Simpan sebagai booster format (standard) dan joblib
    lgb_model.booster_.save_model(os.path.join(models_dir, "lgb_model.txt"))
    joblib.dump(lgb_model, os.path.join(models_dir, "lgb_model.joblib"))
    durations["LightGBM"] = lgb_time
    
    # 4. Train XGBoost
    xgb_model, xgb_time = train_xgb(X_train, y_train, spw)
    joblib.dump(xgb_model, os.path.join(models_dir, "xgb_model.joblib"))
    durations["XGBoost"] = xgb_time
    
    logger.info("Seluruh pelatihan model AI selesai!")
    logger.info(f"Durasi Training: RF={rf_time:.2f}s | LGB={lgb_time:.2f}s | XGB={xgb_time:.2f}s")
    
    return durations

if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(src_dir, ".."))
    
    d_dir = os.path.join(project_root, config.PROCESSED_DIR)
    m_dir = os.path.join(project_root, config.MODELS_DIR)
    
    try:
        train_all_models(d_dir, m_dir)
    except Exception as e:
        logger.error(f"Pelatihan Gagal: {str(e)}", exc_info=True)
