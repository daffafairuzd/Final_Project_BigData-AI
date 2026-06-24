"""
eda_spark.py — Wrapper untuk kompatibilitas pipeline lama.
Mengimpor dan menyatukan fungsi-fungsi baru dari eda.py.
"""
from src.eda import run_full_eda
from src.utils import get_logger

logger = get_logger("eda_legacy_wrapper")

if __name__ == "__main__":
    import os
    from src import config
    
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(src_dir, ".."))
    
    p_dir = os.path.join(project_root, config.PROCESSED_DIR)
    e_dir = os.path.join(project_root, config.EDA_DIR)
    
    logger.info("--- [Wrapper] Memulai EDA Spark Modular ---")
    run_full_eda(p_dir, e_dir)
    logger.info("--- [Wrapper] EDA Spark Selesai ---")
