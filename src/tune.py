import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc
import lightgbm as lgb
import optuna

def objective(trial, X_train, y_train, X_val, y_val):
    # Search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 5.0, 40.0),
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }
    
    # Try GPU if available, else CPU
    try:
        gpu_params = params.copy()
        gpu_params['device'] = 'gpu'
        model = lgb.LGBMClassifier(**gpu_params)
        model.fit(X_train, y_train)
    except Exception:
        cpu_params = params.copy()
        cpu_params['device'] = 'cpu'
        model = lgb.LGBMClassifier(**cpu_params)
        model.fit(X_train, y_train)
        
    y_prob = model.predict_proba(X_val)[:, 1]
    precisions, recalls, _ = precision_recall_curve(y_val, y_prob)
    pr_auc = auc(recalls, precisions)
    
    return pr_auc

def main():
    # Gunakan path relatif dinamis berbasis lokasi file ini
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(src_dir, ".."))
    
    train_path = os.path.join(project_root, "data", "processed", "train_balanced.parquet")
    model_dir = os.path.join(project_root, "models")
    
    if not os.path.exists(train_path):
        print(f"Data latih {train_path} tidak ditemukan. Harap jalankan preprocess.py terlebih dahulu.")
        return
        
    print("Membaca data latih untuk tuning...")
    df = pd.read_parquet(train_path)
    
    # Drop unused columns
    features_to_drop = ["From_Account", "To_Account", "Timestamp", "Is_Laundering"]
    X = df.drop(columns=features_to_drop, errors='ignore')
    y = df["Is_Laundering"]
    
    # Encoding categorical
    categorical_cols = ["Payment_Format", "Receiving_Currency", "Payment_Currency", "From_Bank", "To_Bank"]
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_encoded = X.copy()
    X_encoded[categorical_cols] = encoder.fit_transform(X[categorical_cols])
    
    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, stratify=y, random_state=42)
    
    print(f"Ukuran Data Latih Tuning: {X_train.shape[0]:,} baris")
    print(f"Ukuran Data Validasi Tuning: {X_val.shape[0]:,} baris")
    
    # Setup Optuna study
    print("Memulai pencarian hyperparameter dengan Optuna (15 trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=15)
    
    print("\n=== HASIL TUNING HYPERPARAMETER ===")
    print(f"PR-AUC Terbaik: {study.best_value:.4f}")
    print("Parameter Terbaik:")
    best_params = study.best_params
    print(json.dumps(best_params, indent=4))
    
    # Simpan ke file config
    config_path = os.path.join(model_dir, "best_lgb_params.json")
    with open(config_path, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"Parameter terbaik disimpan ke {config_path}")

if __name__ == "__main__":
    main()
