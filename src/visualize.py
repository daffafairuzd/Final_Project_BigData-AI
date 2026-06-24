"""
visualize.py — Modul Pembuatan Grafik & Visualisasi
Tanggung jawab:
1. Menyajikan visualisasi evaluasi model AI secara komparatif (ROC, PR, Confusion Matrix).
2. Membuat visualisasi data-driven RBA tier dan fraud rate per nominal bucket.
3. Menyajikan visualisasi EDA hasil agregasi Spark SQL secara premium dan konsisten.
"""
import os
import json
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, roc_auc_score

from src import config
from src.utils import get_logger, ensure_dirs, load_json

logger = get_logger("visualize")

# ─────────────────────────────────────────────────────────────
# TEMA VISUAL KONSISTEN
# ─────────────────────────────────────────────────────────────
PALETTE_NORMAL = "#3A86FF"   # Biru cerah
PALETTE_FRAUD  = "#FF006E"   # Merah-pink
PALETTE_SPARK  = "#E25A1C"   # Oranye Spark
PALETTE_XGB    = "#34A853"   # Hijau Google / XGBoost
PALETTE_RF     = "#FBBC05"   # Kuning Google / Random Forest

def setup_plot_theme():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
        "figure.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

def generate_visualizations(predictions_dir: str, models_dir: str, plots_dir: str, test_parquet_path: str):
    """
    Menghasilkan grafik evaluasi performa model AI (RF vs LightGBM vs XGBoost).
    """
    logger.info("Memulai pembuatan grafik evaluasi model...")
    ensure_dirs(plots_dir)
    setup_plot_theme()
    
    # 1. Load metrics_summary.json
    summary_path = os.path.join(predictions_dir, "metrics_summary.json")
    if not os.path.exists(summary_path):
        logger.error("metrics_summary.json tidak ditemukan. Visualisasi dihentikan.")
        return
    metrics_summary = load_json(summary_path)
    
    # 2. Load Predictions
    models_to_plot = ["LightGBM", "RandomForest", "XGBoost"]
    predictions = {}
    
    for name in models_to_plot:
        pred_path = os.path.join(predictions_dir, f"{name.lower()}_predictions.parquet")
        if os.path.exists(pred_path):
            predictions[name] = pd.read_parquet(pred_path)
            logger.info(f"Prediksi {name} berhasil dimuat.")
            
    if not predictions:
        logger.error("Tidak ada data prediksi untuk divisualisasikan!")
        return
        
    # ────────────── PLOT 1: ROC CURVE COMPARISON ──────────────
    logger.info("Membuat grafik ROC Curve Comparison...")
    plt.figure(figsize=(9, 6.5))
    model_colors = {"LightGBM": PALETTE_NORMAL, "RandomForest": PALETTE_RF, "XGBoost": PALETTE_XGB}
    
    for name, pred_df in predictions.items():
        y_true = pred_df["y_true"]
        y_prob = pred_df["y_prob"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        
        plt.plot(fpr, tpr, label=f"{name} (ROC-AUC = {auc_score:.4f})", 
                 color=model_colors.get(name, "#888888"), lw=2.5)
                 
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Guess (0.50)")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR / Recall)")
    plt.title("ROC Curve Overlay - Perbandingan Model Klasifikasi")
    plt.legend(loc="lower right", fontsize=10)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "roc_curve_comparison.png"), dpi=200)
    plt.close()

    # ────────────── PLOT 2: PR CURVE COMPARISON ──────────────
    logger.info("Membuat grafik Precision-Recall Curve Comparison...")
    plt.figure(figsize=(9, 6.5))
    
    # Hitung baseline PR-AUC (fraud rate pada test set)
    y_any = list(predictions.values())[0]["y_true"]
    baseline_pr = np.sum(y_any == 1) / len(y_any)
    
    for name, pred_df in predictions.items():
        y_true = pred_df["y_true"]
        y_prob = pred_df["y_prob"]
        precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recalls, precisions)
        
        plt.plot(recalls, precisions, label=f"{name} (PR-AUC = {pr_auc:.4f})", 
                 color=model_colors.get(name, "#888888"), lw=2.5)
                 
    plt.axhline(y=baseline_pr, color='r', linestyle=':', lw=1.5, 
                label=f"Baseline (Imbalance Ratio = {baseline_pr:.4f})")
    plt.xlabel("Recall (Sensitivitas)")
    plt.ylabel("Precision (Presisi)")
    plt.title("Precision-Recall (PR) Curve Overlay")
    plt.legend(loc="lower left", fontsize=10)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "pr_curve_comparison.png"), dpi=200)
    plt.close()

    # ────────────── PLOT 3: MODEL COMPARISON BAR CHART ──────────────
    logger.info("Membuat grafik Model Comparison Bar Chart...")
    comparison_data = []
    
    for model_name, metrics in metrics_summary.items():
        opt = metrics["tuned_threshold_optimal"]
        # Tambahkan data hasil tuned static
        comparison_data.append({
            "Model": f"{model_name}\n(Tuned Static)",
            "PR-AUC": metrics["pr_auc"],
            "Precision": opt["precision"],
            "Recall": opt["recall"],
            "F1-Score": opt["f1_score"]
        })
        # Tambahkan data RBA jika ada (biasanya di LightGBM/XGBoost)
        if "rba_threshold_dynamic" in metrics:
            rba = metrics["rba_threshold_dynamic"]
            comparison_data.append({
                "Model": f"{model_name}\n(RBA Dynamic)",
                "PR-AUC": metrics["pr_auc"],
                "Precision": rba["precision"],
                "Recall": rba["recall"],
                "F1-Score": rba["f1_score"]
            })
            
    df_compare = pd.DataFrame(comparison_data)
    df_melted = df_compare.melt(id_vars="Model", var_name="Metric", value_name="Value")
    
    plt.figure(figsize=(12, 6.5))
    sns.barplot(data=df_melted, x="Model", y="Value", hue="Metric", palette="muted")
    plt.ylabel("Nilai Skor")
    plt.xlabel("")
    plt.title("Perbandingan Metrik Kunci antar Model & Metode Thresholding")
    plt.ylim([0, 1.05])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "model_comparison_bar.png"), dpi=200)
    plt.close()

    # ────────────── PLOT 4: SIDE-BY-SIDE CONFUSION MATRIX ──────────────
    logger.info("Membuat grafik Confusion Matrix 3 Models Side-by-Side...")
    num_models = len(predictions)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5.5))
    
    if num_models == 1:
        axes = [axes]
        
    for i, (name, pred_df) in enumerate(predictions.items()):
        y_true = pred_df["y_true"]
        y_prob = pred_df["y_prob"]
        
        # Ambil threshold tuned dari metrics_summary
        t_opt = metrics_summary[name]["tuned_threshold_optimal"]["threshold"]
        y_pred = (y_prob >= t_opt).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Color mapping yang berbeda per model agar menarik
        cmap_color = "Blues" if name == "LightGBM" else ("Oranges" if name == "RandomForest" else "Greens")
        
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap_color, ax=axes[i], cbar=False,
                    annot_kws={"size": 13, "weight": "bold"})
                    
        axes[i].set_title(f"{name} (t = {t_opt:.2f})", fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Label Prediksi")
        axes[i].set_ylabel("Label Aktual")
        axes[i].set_xticklabels(["Normal", "Fraud"])
        axes[i].set_yticklabels(["Normal", "Fraud"])
        
        total_fraud = np.sum(y_true == 1)
        tp = cm[1, 1] if cm.shape == (2, 2) else 0
        fp = cm[0, 1] if cm.shape == (2, 2) else 0
        rec = tp / total_fraud if total_fraud > 0 else 0
        
        axes[i].text(0.5, -0.15, f"Recall: {rec*100:.2f}% | Alerts: {tp+fp:,}\nFalse Positives (FP): {fp:,}", 
                     ha="center", transform=axes[i].transAxes, fontsize=10, color="darkred", weight="bold")
                     
    plt.suptitle("Confusion Matrix Perbandingan 3 Model (Threshold Tuned Optimal)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix_3models.png"), dpi=200)
    plt.close()

    # ────────────── PLOT 5: RBA FRAUD RATE BUCKETS & TIERS ──────────────
    # Plot histogram rate fraud per bucket dan garis tier boundaries
    rba_config_path = os.path.join(predictions_dir, "..", "eda_results", "rba_tier_config.json")
    buckets_path = os.path.join(predictions_dir, "..", "eda_results", "fraud_rate_by_bucket.json")
    
    if os.path.exists(rba_config_path) and os.path.exists(buckets_path):
        logger.info("Membuat grafik RBA Fraud Rate Buckets & Tiers...")
        rba_config = load_json(rba_config_path)
        buckets = load_json(buckets_path)
        
        df_buckets = pd.DataFrame(buckets)
        low_boundary = rba_config["low_boundary"]
        high_boundary = rba_config["high_boundary"]
        
        plt.figure(figsize=(12, 6))
        # Plot fraud rate per bucket
        sns.barplot(data=df_buckets, x="bucket_id", y="fraud_rate_pct", color=PALETTE_NORMAL, alpha=0.85)
        
        # Berikan warna latar belakang berbeda untuk tiap risk tier
        # Cari bucket id transisi
        low_max_bucket = 1
        high_min_bucket = 20
        
        for b in buckets:
            if b["bucket_max"] <= low_boundary:
                low_max_bucket = max(low_max_bucket, b["bucket_id"])
            if b["bucket_min"] >= high_boundary:
                high_min_bucket = min(high_min_bucket, b["bucket_id"])
                
        # Gambar garis pemisah visual
        plt.axvline(x=low_max_bucket - 0.5, color="darkgreen", linestyle="--", lw=2, label=f"Batas Low Risk: ${low_boundary:,.0f}")
        plt.axvline(x=high_min_bucket - 1.5, color="darkred", linestyle="--", lw=2, label=f"Batas High Risk: ${high_boundary:,.0f}")
        
        # Shading warna latar belakang tier
        plt.axvspan(-0.5, low_max_bucket - 0.5, color="green", alpha=0.1)
        plt.axvspan(low_max_bucket - 0.5, high_min_bucket - 1.5, color="orange", alpha=0.1)
        plt.axvspan(high_min_bucket - 1.5, 19.5, color="red", alpha=0.1)
        
        # Anotasi teks
        plt.text((low_max_bucket - 1) / 2, plt.gca().get_ylim()[1]*0.85, "LOW\nRISK", ha="center", color="green", weight="bold", fontsize=12)
        plt.text((low_max_bucket - 0.5 + high_min_bucket - 1.5) / 2, plt.gca().get_ylim()[1]*0.85, "MEDIUM\nRISK", ha="center", color="darkorange", weight="bold", fontsize=12)
        plt.text((high_min_bucket - 1.5 + 19.5) / 2, plt.gca().get_ylim()[1]*0.85, "HIGH\nRISK", ha="center", color="red", weight="bold", fontsize=12)
        
        plt.title("Analisis Fraud Rate per Quantile Bucket & Penentuan Risk Tiers RBA", fontsize=13, fontweight="bold")
        plt.xlabel("ID Bucket Quantile (Berdasarkan Kenaikan Nominal Amount_Paid)")
        plt.ylabel("Fraud Rate per Bucket (%)")
        plt.legend(loc="upper left", frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "rba_fraud_rate_buckets.png"), dpi=200)
        plt.close()

    # ────────────── PLOT 6: FEATURE IMPORTANCE (XGBoost & LightGBM) ──────────────
    # XGBoost Feature Importance
    xgb_path = os.path.join(models_dir, "xgb_model.joblib")
    features_json_path = os.path.join(models_dir, "feature_names.json")
    
    if os.path.exists(xgb_path) and os.path.exists(features_json_path):
        logger.info("Membuat grafik XGBoost Feature Importance...")
        try:
            model = joblib.load(xgb_path)
            with open(features_json_path, "r") as f:
                feature_names = json.load(f)
                
            importances = model.feature_importances_
            
            feat_imp_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)
            
            plt.figure(figsize=(10, 7.5))
            sns.barplot(data=feat_imp_df.head(12), x="Importance", y="Feature", palette="viridis")
            plt.title("12 Fitur Paling Berpengaruh dalam Deteksi Fraud (XGBoost Weight)", fontsize=13, fontweight="bold")
            plt.xlabel("Skor Kepentingan Relatif (Weight)")
            plt.ylabel("Nama Fitur")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "xgb_feature_importance.png"), dpi=200)
            plt.close()
            
            # Export to JSON
            feat_imp_df.to_json(os.path.join(predictions_dir, "feature_importance_xgb.json"), orient="records", indent=4)
        except Exception as e:
            logger.warning(f"Gagal memuat visualisasi feature importance XGBoost: {str(e)}")

    # Visualisasi LightGBM Feature Importance (dari gain)
    lgb_txt_path = os.path.join(models_dir, "lgb_model.txt")
    if os.path.exists(lgb_txt_path) and os.path.exists(features_json_path):
        logger.info("Membuat grafik LightGBM Feature Importance...")
        try:
            model = lgb.Booster(model_file=lgb_txt_path)
            importances = model.feature_importance(importance_type="gain")
            with open(features_json_path, "r") as f:
                feature_names = json.load(f)
                
            feat_imp_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)
            
            plt.figure(figsize=(10, 7.5))
            sns.barplot(data=feat_imp_df.head(12), x="Importance", y="Feature", palette="plasma")
            plt.title("12 Fitur Paling Berpengaruh dalam Deteksi Fraud (LightGBM Gain)", fontsize=13, fontweight="bold")
            plt.xlabel("Skor Kepentingan Relatif (Gain)")
            plt.ylabel("Nama Fitur")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "feature_importance.png"), dpi=200)
            plt.close()
            
            feat_imp_df.to_json(os.path.join(predictions_dir, "feature_importance.json"), orient="records", indent=4)
        except Exception as e:
            logger.warning(f"Gagal memuat visualisasi feature importance LightGBM: {str(e)}")

    # ────────────── PLOT 7: NOMINAL BOXPLOT & PAYMENT FORMAT ──────────────
    if os.path.exists(test_parquet_path):
        logger.info("Membaca test set untuk visualisasi boxplot dan payment format...")
        try:
            df_subset = pd.read_parquet(test_parquet_path, columns=["Amount_Paid", "Payment_Format", "Is_Laundering"])
            
            # 7a. Boxplot Nominal
            plt.figure(figsize=(8, 5))
            df_subset["Log_Amount_Paid"] = np.log10(df_subset["Amount_Paid"] + 1e-5)
            sns.boxplot(data=df_subset, x="Is_Laundering", y="Log_Amount_Paid", palette=[PALETTE_NORMAL, PALETTE_FRAUD])
            plt.title("Perbandingan Nominal Transaksi: Normal vs Fraud (Skala Log10)")
            plt.xlabel("Status Transaksi")
            plt.ylabel("Nominal Pembayaran (Log10)")
            plt.xticks([0, 1], ["Normal", "Fraud"])
            
            fraud_med = df_subset[df_subset["Is_Laundering"] == 1]["Amount_Paid"].median()
            normal_med = df_subset[df_subset["Is_Laundering"] == 0]["Amount_Paid"].median()
            
            plt.text(0.5, 0.05, f"Median Fraud: ${fraud_med:,.2f}\nMedian Normal: ${normal_med:,.2f}",
                     ha="center", transform=plt.gca().transAxes, fontsize=10, 
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"))
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "fraud_amount_distribution.png"), dpi=200)
            plt.close()
            
            # 7b. Barplot Format Pembayaran
            fmt_normal = df_subset[df_subset["Is_Laundering"] == 0]["Payment_Format"].value_counts(normalize=True).reset_index()
            fmt_normal["Class"] = "Normal"
            
            fmt_fraud = df_subset[df_subset["Is_Laundering"] == 1]["Payment_Format"].value_counts(normalize=True).reset_index()
            fmt_fraud["Class"] = "Fraud"
            
            fmt_df = pd.concat([fmt_normal, fmt_fraud], ignore_index=True)
            
            plt.figure(figsize=(10, 5))
            sns.barplot(data=fmt_df, x="Payment_Format", y="proportion", hue="Class", palette=[PALETTE_NORMAL, PALETTE_FRAUD])
            plt.title("Proporsi Penggunaan Format Metode Pembayaran")
            plt.xlabel("Format Pembayaran")
            plt.ylabel("Proporsi (%)")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "payment_format_comparison.png"), dpi=200)
            plt.close()
            
        except Exception as e:
            logger.warning(f"Gagal menggambar boxplot nominal/payment format: {str(e)}")

    logger.info("Semua grafik visualisasi evaluasi model berhasil dibuat!")

def generate_eda_visualizations(eda_dir: str, plots_dir: str):
    """
    Membuat visualisasi hasil EDA (class distribution, daily fraud timeline, hourly heatmap, dll.).
    """
    logger.info("Memulai pembuatan grafik visualisasi EDA...")
    eda_plots_dir = os.path.join(plots_dir, "eda")
    ensure_dirs(eda_plots_dir)
    setup_plot_theme()
    
    # ── PLOT 1: Class Distribution ──
    class_path = os.path.join(eda_dir, "class_distribution.json")
    if os.path.exists(class_path):
        logger.info("[VIZ EDA] Plot 1: Class Imbalance...")
        class_data = load_json(class_path)
        
        labels = [d["label"] for d in class_data]
        counts = [d["jumlah"] for d in class_data]
        pcts = [d["persentase"] for d in class_data]
        colors = [PALETTE_FRAUD if d["is_laundering"] == 1 else PALETTE_NORMAL for d in class_data]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        
        # Left Barplot
        bars = axes[0].bar(labels, counts, color=colors, width=0.45, edgecolor="white", lw=1.5)
        axes[0].set_title("Jumlah Transaksi per Kelas")
        axes[0].set_ylabel("Jumlah Transaksi")
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
        
        for bar, count, pct in zip(bars, counts, pcts):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.5,
                f"{count:,}\n({pct}%)", ha="center", va="center",
                fontsize=11, fontweight="bold", color="white"
            )
            
        # Right Donut Chart
        axes[1].pie(counts, labels=labels, colors=colors, autopct="%1.3f%%", startangle=90,
                   wedgeprops=dict(width=0.45, edgecolor="white", lw=2), pctdistance=0.75,
                   textprops={"fontsize": 11, "weight": "bold", "color": "black"})
        axes[1].set_title("Proporsi Kelas (%)")
        
        plt.suptitle("Distribusi Kelas Dataset - Class Imbalance Ekstrem", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(eda_plots_dir, "01_class_imbalance.png"), dpi=200)
        plt.close()

    # ── PLOT 2: Daily Timeline ──
    timeline_path = os.path.join(eda_dir, "daily_fraud_timeline.json")
    if os.path.exists(timeline_path):
        logger.info("[VIZ EDA] Plot 2: Daily Fraud Timeline...")
        timeline_data = load_json(timeline_path)
        df_t = pd.DataFrame(timeline_data)
        df_t["tanggal"] = pd.to_datetime(df_t["tanggal"])
        df_t = df_t.sort_values("tanggal")
        
        df_fraud = df_t[df_t["is_laundering"] == 1]
        
        plt.figure(figsize=(13, 5.5))
        plt.plot(df_fraud["tanggal"], df_fraud["jumlah_transaksi"], color=PALETTE_FRAUD, lw=2.5, marker="o", label="Transaksi Fraud")
        plt.title("Timeline Frekuensi Transaksi Pencucian Uang (Fraud) Harian")
        plt.ylabel("Jumlah Transaksi Fraud")
        plt.xlabel("Tanggal Transaksi")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(eda_plots_dir, "02_daily_fraud_timeline.png"), dpi=200)
        plt.close()

    # ── PLOT 3: Hourly Heatmap ──
    hourly_path = os.path.join(eda_dir, "hourly_distribution.json")
    if os.path.exists(hourly_path):
        logger.info("[VIZ EDA] Plot 3: Hourly Heatmap...")
        hourly_data = load_json(hourly_path)
        df_h = pd.DataFrame(hourly_data)
        
        df_h_fraud = df_h[df_h["is_laundering"] == 1].sort_values("hour")
        
        plt.figure(figsize=(12, 5))
        sns.barplot(data=df_h_fraud, x="hour", y="jumlah_transaksi", color=PALETTE_FRAUD, alpha=0.85)
        plt.title("Distribusi Waktu Transaksi Fraud Berdasarkan Jam dalam Sehari")
        plt.xlabel("Jam Kejadian (00.00 - 23.00)")
        plt.ylabel("Jumlah Transaksi Fraud")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Tambahkan anotasi pola jam sibuk
        peak_hour = df_h_fraud.loc[df_h_fraud["jumlah_transaksi"].idxmax()]["hour"]
        plt.axvline(x=peak_hour, color="darkred", linestyle=":", lw=2)
        plt.text(peak_hour + 0.3, plt.gca().get_ylim()[1]*0.9, f"Puncak Aktivitas: Jam {peak_hour:02d}.00\n(Berdasarkan Spark SQL Query)",
                 color="darkred", weight="bold", fontsize=10)
                 
        plt.tight_layout()
        plt.savefig(os.path.join(eda_plots_dir, "05_hourly_heatmap.png"), dpi=200)
        plt.close()

    # ── PLOT 4: Correlation Matrix Heatmap ──
    corr_path = os.path.join(eda_dir, "correlation_matrix.json")
    if os.path.exists(corr_path):
        logger.info("[VIZ EDA] Plot 4: Correlation Heatmap...")
        corr_data = load_json(corr_path)
        features = corr_data["features"]
        matrix = corr_data["matrix"]
        
        # Ubah format dictionary ke matrix
        n = len(features)
        corr_matrix = np.zeros((n, n))
        for i, f1 in enumerate(features):
            for j, f2 in enumerate(features):
                corr_matrix[i, j] = matrix[f1][f2]
                
        df_corr = pd.DataFrame(corr_matrix, index=features, columns=features)
        
        plt.figure(figsize=(11, 9))
        sns.heatmap(df_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1.0, vmax=1.0,
                    square=True, cbar_kws={"shrink": 0.8}, annot_kws={"size": 9})
        plt.title("Matriks Korelasi Pearson Fitur Transaksi & Label Laundering", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(eda_plots_dir, "03_correlation_heatmap.png"), dpi=200)
        plt.close()

    # ── PLOT 5: RBA Tier Segmentation ──
    tier_path = os.path.join(eda_dir, "tier_segmentation.json")
    if os.path.exists(tier_path):
        logger.info("[VIZ EDA] Plot 5: RBA Tier Segmentation...")
        tier_data = load_json(tier_path)
        df_tier = pd.DataFrame(tier_data)
        
        df_tier_fraud = df_tier[df_tier["is_laundering"] == 1]
        
        plt.figure(figsize=(10, 5.5))
        sns.barplot(data=df_tier_fraud, x="tier", y="jumlah", palette="OrRd")
        plt.title("Distribusi Transaksi Fraud Berdasarkan Tier Nominal Pembayaran")
        plt.xlabel("Risk Tiers (Standard Pembatasan)")
        plt.ylabel("Jumlah Kasus Fraud")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(eda_plots_dir, "04_tier_segmentation.png"), dpi=200)
        plt.close()

    logger.info("Semua grafik visualisasi EDA berhasil dibuat!")

if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(src_dir, ".."))
    
    e_dir = os.path.join(project_root, config.EDA_DIR)
    p_dir = os.path.join(project_root, config.PROCESSED_DIR)
    plots_folder = os.path.join(project_root, config.PLOTS_DIR)
    
    test_parquet = os.path.join(p_dir, "test_features.parquet")
    
    try:
        generate_eda_visualizations(e_dir, plots_folder)
        generate_visualizations(p_dir, os.path.join(project_root, config.MODELS_DIR), plots_folder, test_parquet)
    except Exception as e:
        logger.error(f"Gagal generate visualisasi: {str(e)}", exc_info=True)
