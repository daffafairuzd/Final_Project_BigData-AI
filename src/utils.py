"""
utils.py — Utilitas Bersama: Logger, Timer, File Helpers
Digunakan oleh semua modul lain untuk konsistensi logging dan I/O.
"""
import os
import json
import logging
import time
from contextlib import contextmanager


# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Buat logger dengan format: [TIMESTAMP] [LEVEL] [MODULE] message.
    Output ke console (INFO+) dan opsional ke file (DEBUG+).
    Menghindari duplikasi handler jika dipanggil berkali-kali.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler — level INFO ke atas
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (opsional) — level DEBUG ke atas
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────
# TIMER
# ─────────────────────────────────────────────────────────────

@contextmanager
def timer(label: str, logger=None):
    """
    Context manager untuk mengukur dan mencatat waktu eksekusi blok kode.

    Contoh:
        with timer("Training LightGBM", logger):
            model.fit(X, y)
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        msg = f"[TIMER] {label}: {elapsed:.2f} detik"
        if logger:
            logger.info(msg)
        else:
            print(msg)


# ─────────────────────────────────────────────────────────────
# DIRECTORY & FILE HELPERS
# ─────────────────────────────────────────────────────────────

def ensure_dirs(*paths: str) -> None:
    """Buat direktori (dan parent-nya) jika belum ada."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


def save_json(obj, path: str) -> None:
    """Simpan objek Python ke file JSON dengan indent=4 dan encoding UTF-8."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False, default=str)


def load_json(path: str):
    """
    Load file JSON. Raise FileNotFoundError dengan pesan yang jelas
    jika file tidak ditemukan (membantu debugging langkah mana yang belum dijalankan).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}\n"
            "Pastikan langkah pipeline sebelumnya sudah berhasil dijalankan."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
# CLASS WEIGHT HELPER
# ─────────────────────────────────────────────────────────────

def get_scale_pos_weight(y_series) -> float:
    """
    Hitung scale_pos_weight = count(kelas 0) / count(kelas 1).
    Digunakan oleh LightGBM dan XGBoost untuk menangani class imbalance.
    """
    import numpy as np
    y = np.asarray(y_series)
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    if pos == 0:
        raise ValueError("Tidak ada sampel positif (fraud) dalam data!")
    spw = float(neg / pos)
    return spw


def get_class_weight_dict(y_series) -> dict:
    """
    Hitung class_weight dict untuk Random Forest.
    Return: {0: 1.0, 1: scale_pos_weight}
    """
    spw = get_scale_pos_weight(y_series)
    return {0: 1.0, 1: spw}
