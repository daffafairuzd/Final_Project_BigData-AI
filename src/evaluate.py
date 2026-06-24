"""
evaluate.py — Modul Evaluasi Model & Data-Driven RBA
Tanggung jawab:
1. Memuat model yang dilatih dan data uji (test_features.parquet).
2. Membangun risk tiers RBA berbasis data dari fraud_rate_by_bucket.json (bukan hardcoded).
3. Melakukan pencarian threshold optimal (static tuned) dan RBA (dynamic).
4. Mengevaluasi performa model secara terstandarisasi (PR-AUC, ROC-AUC, Precision, Recall, F1, CM).
5. Menyimpan ringkasan metrik evaluasi ke format JSON dan CSV.
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, accuracy_score, confusion_matrix, classification_report

from src import config
from src.utils import get_logger, timer, ensure_dirs, load_json, save_json

# Tambah import xgboost jika tersedia
try:
    import xgboost as xgb
except ImportError:
    xgb = None

logger = get_logger("evaluate")

def build_rba_tiers(bucket_json_path: str, y_prob: np.ndarray, y_true: np.ndarray, amount: np.ndarray) -> dict:
    """
    Konstruksi risk tiers otomatis dari hasil analisis fraud rate EDA.
    1. Membaca baseline fraud rate.
    2. Mengelompokkan bucket ke dalam Low/Medium/High Risk berdasarkan pengali baseline.
    3. Menentukan batas Amount_Paid dinamis berdasarkan transisi bucket.
    4. Melakukan sweep threshold per tier untuk mengoptimalkan presisi/recall.
    """
    logger.info(f"Membaca data quantile buckets dari {bucket_json_path}...")
    buckets = load_json(bucket_json_path)
    
    # Hitung baseline fraud rate rata-rata
    total_fraud = sum(b["fraud_tx"] for b in buckets)
    total_tx = sum(b["total_tx"] for b in buckets)
    
    if total_tx == 0:
        logger.warning("Total transaksi pada buckets bernilai 0. Menggunakan fallback baseline.")
        baseline = 0.005  # Fallback default ~0.5%
    else:
        baseline = total_fraud / total_tx
        
    logger.info(f"Baseline Fraud Rate: {baseline * 100:.4f}%")
    
    # Klasifikasi setiap bucket ke dalam tier
    low_buckets = []
    high_buckets = []
    
    for b in buckets:
        rate = b["fraud_rate_pct"] / 100.0
        # Pengali baseline
        multiplier = rate / max(baseline, 0.0001)
        
        if multiplier < config.RBA_LOW_MULTIPLIER:
            low_buckets.append(b)
        elif multiplier >= config.RBA_HIGH_MULTIPLIER:
            high_buckets.append(b)
            
    # Tentukan batas nominal
    low_boundary = max([b["bucket_max"] for b in low_buckets], default=5000.0)
    high_boundary = min([b["bucket_min"] for b in high_buckets], default=50000.0)
    
    # Validasi jika terjadi tumpang tindih
    if low_boundary >= high_boundary or low_boundary <= 0:
        logger.warning(f"Batas nominal tumpang tindih (Low: {low_boundary}, High: {high_boundary}). Menggunakan default fallback.")
        low_boundary = 10000.0
        high_boundary = 100000.0
        
    logger.info(f"Batas Nominal Dinamis Terbentuk: Low < {low_boundary:.2f} | High >= {high_boundary:.2f}")
    
    # Sweep threshold optimal per tier
    rba_config = {
        "low_boundary": float(low_boundary),
        "high_boundary": float(high_boundary),
        "baseline_rate": float(baseline),
        "tiers": {}
    }
    
    # Daftar masker untuk membagi data evaluasi berdasarkan tier
    masks = {
        "Low": amount < low_boundary,
        "Medium": (amount >= low_boundary) & (amount < high_boundary),
        "High": amount >= high_boundary
    }
    
    thresholds = config.EVAL_THRESHOLDS
    
    for tier_name, mask in masks.items():
        y_true_tier = y_true[mask]
        y_prob_tier = y_prob[mask]
        
        if len(y_true_tier) == 0 or np.sum(y_true_tier == 1) == 0:
            logger.warning(f"Data tier {tier_name} kosong atau tidak mengandung transaksi fraud. Menggunakan default threshold.")
            rba_config["tiers"][tier_name] = {"threshold": 0.50, "metric": "default"}
            continue
            
        best_t = 0.50
        
        if tier_name == "Low":
            # Utamakan presisi tinggi untuk mengurangi false alarm pada transaksi retail kecil
            best_f1 = -1.0
            best_prec = -1.0
            for t in thresholds:
                y_pred = (y_prob_tier >= t).astype(int)
                cm = confusion_matrix(y_true_tier, y_pred)
                if cm.shape == (2, 2):
                    tp, fp = cm[1, 1], cm[0, 1]
                else:
                    tp, fp = 0, 0
                    
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / np.sum(y_true_tier == 1)
                f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                
                # Cari F1 terbaik dengan constraint Precision >= RBA_LOW_MIN_PRECISION
                if prec >= config.RBA_LOW_MIN_PRECISION:
                    if f1 > best_f1:
                        best_f1 = f1
                        best_t = t
                # Fallback ke presisi tertinggi jika tidak ada yang melewati batas
                if prec > best_prec:
                    best_prec = prec
                    if best_f1 < 0:
                        best_t = t
                        
        elif tier_name == "High":
            # Utamakan Recall tinggi untuk menangkap sebanyak mungkin transaksi fraud bernilai tinggi
            best_rec = -1.0
            for t in thresholds:
                y_pred = (y_prob_tier >= t).astype(int)
                cm = confusion_matrix(y_true_tier, y_pred)
                if cm.shape == (2, 2):
                    tp, fp = cm[1, 1], cm[0, 1]
                else:
                    tp, fp = 0, 0
                    
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / np.sum(y_true_tier == 1)
                
                # Cari Recall tertinggi dengan constraint Precision >= RBA_HIGH_MIN_PRECISION
                if prec >= config.RBA_HIGH_MIN_PRECISION:
                    if rec > best_rec:
                        best_rec = rec
                        best_t = t
            # Fallback ke F1 optimal biasa
            if best_rec < 0:
                best_f1 = -1.0
                for t in thresholds:
                    y_pred = (y_prob_tier >= t).astype(int)
                    cm = confusion_matrix(y_true_tier, y_pred)
                    if cm.shape == (2, 2):
                        tp, fp = cm[1, 1], cm[0, 1]
                    else:
                        tp, fp = 0, 0
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    rec = tp / np.sum(y_true_tier == 1)
                    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                    if f1 > best_f1:
                        best_f1 = f1
                        best_t = t
        else:
            # Medium: Standard F1-Score optimal
            best_f1 = -1.0
            for t in thresholds:
                y_pred = (y_prob_tier >= t).astype(int)
                cm = confusion_matrix(y_true_tier, y_pred)
                if cm.shape == (2, 2):
                    tp, fp = cm[1, 1], cm[0, 1]
                else:
                    tp, fp = 0, 0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / np.sum(y_true_tier == 1)
                f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
                    
        rba_config["tiers"][tier_name] = {
            "threshold": float(best_t),
            "sample_count": int(len(y_true_tier)),
            "fraud_count": int(np.sum(y_true_tier == 1))
        }
        logger.info(f"Tier {tier_name}: Threshold={best_t} | Sampel={len(y_true_tier)} | Fraud={np.sum(y_true_tier == 1)}")
        
    return rba_config

def apply_rba(y_prob: np.ndarray, amount: np.ndarray, rba_config: dict) -> np.ndarray:
    """
    Menerapkan threshold dinamis berdasarkan nominal transaksi (RBA).
    """
    low_boundary = rba_config["low_boundary"]
    high_boundary = rba_config["high_boundary"]
    
    t_low = rba_config["tiers"]["Low"]["threshold"]
    t_med = rba_config["tiers"]["Medium"]["threshold"]
    t_high = rba_config["tiers"]["High"]["threshold"]
    
    y_pred = np.zeros_like(y_prob, dtype=int)
    
    cond_low = (amount < low_boundary) & (y_prob >= t_low)
    cond_med = (amount >= low_boundary) & (amount < high_boundary) & (y_prob >= t_med)
    cond_high = (amount >= high_boundary) & (y_prob >= t_high)
    
    y_pred[cond_low | cond_med | cond_high] = 1
    return y_pred

def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, amount: np.ndarray, 
                         model_name: str, rba_config: dict = None) -> dict:
    """
    Menghitung seluruh metrik evaluasi secara lengkap dan terstandarisasi.
    """
    logger.info(f"Mengevaluasi performa model: {model_name}...")
    
    # 1. Metrik independen threshold
    precisions, recalls, thresholds_curve = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recalls, precisions)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    # 2. Evaluasi pada threshold default 0.50
    y_pred_def = (y_prob >= 0.50).astype(int)
    report_def = classification_report(y_true, y_pred_def, output_dict=True, zero_division=0)
    cm_def = confusion_matrix(y_true, y_pred_def)
    
    # 3. Cari static threshold F1-Score optimal
    best_t = 0.5
    best_f1 = 0.0
    thresholds_to_test = config.EVAL_THRESHOLDS
    
    threshold_results = []
    
    for t in thresholds_to_test:
        y_pred_t = (y_prob >= t).astype(int)
        report_t = classification_report(y_true, y_pred_t, output_dict=True, zero_division=0)
        cm_t = confusion_matrix(y_true, y_pred_t)
        
        p = report_t["1"]["precision"]
        r = report_t["1"]["recall"]
        f1 = report_t["1"]["f1-score"]
        acc = accuracy_score(y_true, y_pred_t)
        
        tp = int(cm_t[1, 1]) if cm_t.shape == (2, 2) else 0
        fp = int(cm_t[0, 1]) if cm_t.shape == (2, 2) else 0
        
        threshold_results.append({
            "threshold": float(t),
            "precision": float(p),
            "recall": float(r),
            "f1_score": float(f1),
            "accuracy": float(acc),
            "true_positives": tp,
            "false_positives": fp
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            
    # Evaluasi pada threshold optimal
    y_pred_opt = (y_prob >= best_t).astype(int)
    report_opt = classification_report(y_true, y_pred_opt, output_dict=True, zero_division=0)
    cm_opt = confusion_matrix(y_true, y_pred_opt)
    
    metrics = {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "default_threshold_0.5": {
            "precision": float(report_def["1"]["precision"]),
            "recall": float(report_def["1"]["recall"]),
            "f1_score": float(report_def["1"]["f1-score"]),
            "accuracy": float(accuracy_score(y_true, y_pred_def)),
            "true_positives": int(cm_def[1, 1]) if cm_def.shape == (2, 2) else 0,
            "false_positives": int(cm_def[0, 1]) if cm_def.shape == (2, 2) else 0,
            "false_negatives": int(cm_def[1, 0]) if cm_def.shape == (2, 2) else 0,
            "true_negatives": int(cm_def[0, 0]) if cm_def.shape == (2, 2) else 0
        },
        "tuned_threshold_optimal": {
            "threshold": float(best_t),
            "precision": float(report_opt["1"]["precision"]),
            "recall": float(report_opt["1"]["recall"]),
            "f1_score": float(report_opt["1"]["f1-score"]),
            "accuracy": float(accuracy_score(y_true, y_pred_opt)),
            "true_positives": int(cm_opt[1, 1]) if cm_opt.shape == (2, 2) else 0,
            "false_positives": int(cm_opt[0, 1]) if cm_opt.shape == (2, 2) else 0,
            "false_negatives": int(cm_opt[1, 0]) if cm_opt.shape == (2, 2) else 0,
            "true_negatives": int(cm_opt[0, 0]) if cm_opt.shape == (2, 2) else 0
        }
    }
    
    # 4. Evaluasi RBA (Dynamic Threshold) jika konfigurasi diberikan
    if rba_config:
        y_pred_rba = apply_rba(y_prob, amount, rba_config)
        report_rba = classification_report(y_true, y_pred_rba, output_dict=True, zero_division=0)
        cm_rba = confusion_matrix(y_true, y_pred_rba)
        
        metrics["rba_threshold_dynamic"] = {
            "precision": float(report_rba["1"]["precision"]),
            "recall": float(report_rba["1"]["recall"]),
            "f1_score": float(report_rba["1"]["f1-score"]),
            "accuracy": float(accuracy_score(y_true, y_pred_rba)),
            "true_positives": int(cm_rba[1, 1]) if cm_rba.shape == (2, 2) else 0,
            "false_positives": int(cm_rba[0, 1]) if cm_rba.shape == (2, 2) else 0,
            "false_negatives": int(cm_rba[1, 0]) if cm_rba.shape == (2, 2) else 0,
            "true_negatives": int(cm_rba[0, 0]) if cm_rba.shape == (2, 2) else 0
        }
        
    return metrics, threshold_results

def run_evaluation(data_dir: str, models_dir: str, eda_results_dir: str) -> None:
    """
    Fungsi orkestrasi lengkap evaluasi model.
    """
    logger.info("Memulai evaluasi model AI terstandarisasi...")
    
    # Load test dataset
    test_path = os.path.join(data_dir, "test_features.parquet")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Dataset uji tidak ditemukan di: {test_path}")
        
    df_test = pd.read_parquet(test_path)
    
    # Pisahkan label dan amount_paid untuk RBA
    y_test = df_test[config.TARGET_COL].values.astype(int)
    amount_paid = df_test["Amount_Paid"].values
    
    # Siapkan fitur untuk prediksi (drop metadata/label)
    features_to_drop = config.ID_COLS + [config.TARGET_COL]
    X_test = df_test.drop(columns=features_to_drop, errors="ignore")
    
    # Load models
    logger.info("Memuat model-model yang telah dilatih...")
    rf_model = joblib.load(os.path.join(models_dir, "rf_model.joblib"))
    lgb_model = joblib.load(os.path.join(models_dir, "lgb_model.joblib"))
    
    xgb_model = None
    xgb_path = os.path.join(models_dir, "xgb_model.joblib")
    if os.path.exists(xgb_path) and xgb is not None:
        xgb_model = joblib.load(xgb_path)
    else:
        logger.warning("Model XGBoost tidak dapat dimuat (berkas tidak ada atau module tidak terinstal).")

    # Prediksi probabilitas
    logger.info("Melakukan prediksi probabilitas pada data uji...")
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
    
    y_prob_xgb = None
    if xgb_model is not None:
        y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
        
    # 1. Konstruksi RBA Tiers dari hasil EDA dan probabilitas model terbaik (LightGBM)
    bucket_json_path = os.path.join(eda_results_dir, "fraud_rate_by_bucket.json")
    rba_config = None
    if os.path.exists(bucket_json_path):
        rba_config = build_rba_tiers(bucket_json_path, y_prob_lgb, y_test, amount_paid)
        save_json(rba_config, os.path.join(eda_results_dir, "rba_tier_config.json"))
        logger.info("Konfigurasi RBA Tier disimpan ke rba_tier_config.json")
    else:
        logger.warning(f"File {bucket_json_path} tidak ditemukan. Evaluasi RBA akan dilewati.")
        
    # 2. Hitung metrik evaluasi tiap model
    results = {}
    threshold_eval_data = {}
    
    # Evaluasi Random Forest
    rf_metrics, rf_sweep = evaluate_predictions(y_test, y_prob_rf, amount_paid, "RandomForest")
    results["RandomForest"] = rf_metrics
    threshold_eval_data["RandomForest"] = rf_sweep
    
    # Evaluasi LightGBM (dengan RBA)
    lgb_metrics, lgb_sweep = evaluate_predictions(y_test, y_prob_lgb, amount_paid, "LightGBM", rba_config)
    results["LightGBM"] = lgb_metrics
    threshold_eval_data["LightGBM"] = lgb_sweep
    
    # Evaluasi XGBoost jika tersedia
    if y_prob_xgb is not None:
        xgb_metrics, xgb_sweep = evaluate_predictions(y_test, y_prob_xgb, amount_paid, "XGBoost", rba_config)
        results["XGBoost"] = xgb_metrics
        threshold_eval_data["XGBoost"] = xgb_sweep
        
    # 3. Simpan ringkasan prediksi untuk plotting (visualize.py)
    logger.info("Menyimpan dataset prediksi model...")
    pd.DataFrame({"y_true": y_test, "y_prob": y_prob_rf}).to_parquet(
        os.path.join(data_dir, "randomforest_predictions.parquet"), index=False
    )
    
    lgb_pred_df = pd.DataFrame({"y_true": y_test, "y_prob": y_prob_lgb, "Amount_Paid": amount_paid})
    if rba_config:
        lgb_pred_df["y_pred_dynamic"] = apply_rba(y_prob_lgb, amount_paid, rba_config)
    lgb_pred_df.to_parquet(os.path.join(data_dir, "lightgbm_predictions.parquet"), index=False)
    
    if y_prob_xgb is not None:
        xgb_pred_df = pd.DataFrame({"y_true": y_test, "y_prob": y_prob_xgb, "Amount_Paid": amount_paid})
        if rba_config:
            xgb_pred_df["y_pred_dynamic"] = apply_rba(y_prob_xgb, amount_paid, rba_config)
        xgb_pred_df.to_parquet(os.path.join(data_dir, "xgboost_predictions.parquet"), index=False)
        
    # Save metrics summaries
    save_json(results, os.path.join(data_dir, "metrics_summary.json"))
    save_json(threshold_eval_data, os.path.join(data_dir, "threshold_evaluation.json"))
    
    # 4. Buat tabel perbandingan ringkas (CSV)
    logger.info("Membuat tabel komparasi performa model...")
    comparison_rows = []
    for model_name, metrics in results.items():
        # Default 0.50
        comparison_rows.append({
            "Model": model_name,
            "Threshold_Type": "Default (0.50)",
            "Threshold_Value": 0.50,
            "PR_AUC": metrics["pr_auc"],
            "ROC_AUC": metrics["roc_auc"],
            "Precision": metrics["default_threshold_0.5"]["precision"],
            "Recall": metrics["default_threshold_0.5"]["recall"],
            "F1_Score": metrics["default_threshold_0.5"]["f1_score"],
            "False_Positives": metrics["default_threshold_0.5"]["false_positives"],
            "True_Positives": metrics["default_threshold_0.5"]["true_positives"]
        })
        # Tuned Optimal
        comparison_rows.append({
            "Model": model_name,
            "Threshold_Type": "Tuned (Static)",
            "Threshold_Value": metrics["tuned_threshold_optimal"]["threshold"],
            "PR_AUC": metrics["pr_auc"],
            "ROC_AUC": metrics["roc_auc"],
            "Precision": metrics["tuned_threshold_optimal"]["precision"],
            "Recall": metrics["tuned_threshold_optimal"]["recall"],
            "F1_Score": metrics["tuned_threshold_optimal"]["f1_score"],
            "False_Positives": metrics["tuned_threshold_optimal"]["false_positives"],
            "True_Positives": metrics["tuned_threshold_optimal"]["true_positives"]
        })
        # RBA Dynamic
        if "rba_threshold_dynamic" in metrics:
            comparison_rows.append({
                "Model": model_name,
                "Threshold_Type": "RBA (Dynamic)",
                "Threshold_Value": "Variable",
                "PR_AUC": metrics["pr_auc"],
                "ROC_AUC": metrics["roc_auc"],
                "Precision": metrics["rba_threshold_dynamic"]["precision"],
                "Recall": metrics["rba_threshold_dynamic"]["recall"],
                "F1_Score": metrics["rba_threshold_dynamic"]["f1_score"],
                "False_Positives": metrics["rba_threshold_dynamic"]["false_positives"],
                "True_Positives": metrics["rba_threshold_dynamic"]["true_positives"]
            })
            
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_csv_path = os.path.join(data_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    logger.info(f"Tabel komparasi disimpan ke: {comparison_csv_path}")
    
    # Cetak ringkasan ke logger
    logger.info("\n=== RINGKASAN PERFORMA MODEL (Tuned Static) ===")
    for model_name, metrics in results.items():
        opt = metrics["tuned_threshold_optimal"]
        logger.info(
            f"{model_name:<12} (t={opt['threshold']:.2f}) | "
            f"PR-AUC: {metrics['pr_auc']:.4f} | "
            f"Precision: {opt['precision']*100:.2f}% | "
            f"Recall: {opt['recall']*100:.2f}% | "
            f"F1-Score: {opt['f1_score']*100:.2f}% | "
            f"FP: {opt['false_positives']:,}"
        )
        
    if rba_config:
        logger.info("\n=== RINGKASAN DETEKSI RBA (Dynamic Tiers) ===")
        for model_name, metrics in results.items():
            if "rba_threshold_dynamic" in metrics:
                rba = metrics["rba_threshold_dynamic"]
                logger.info(
                    f"{model_name:<12} (RBA Tiers)     | "
                    f"Precision: {rba['precision']*100:.2f}% | "
                    f"Recall: {rba['recall']*100:.2f}% | "
                    f"F1-Score: {rba['f1_score']*100:.2f}% | "
                    f"FP: {rba['false_positives']:,} | "
                    f"Alerts: {rba['true_positives'] + rba['false_positives']:,}"
                )

if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(src_dir, ".."))
    
    d_dir = os.path.join(project_root, config.PROCESSED_DIR)
    m_dir = os.path.join(project_root, config.MODELS_DIR)
    e_dir = os.path.join(project_root, config.EDA_DIR)
    
    try:
        run_evaluation(d_dir, m_dir, e_dir)
    except Exception as e:
        logger.error(f"Evaluasi Gagal: {str(e)}", exc_info=True)
