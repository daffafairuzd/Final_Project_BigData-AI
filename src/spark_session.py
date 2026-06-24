"""
spark_session.py — Pengelolaan Inisialisasi Apache Spark Session
Modul ini mengonfigurasi environment variables yang diperlukan untuk PySpark di Windows
dan menginisialisasi SparkSession dengan parameter dari config.py.
"""
import os
import sys
from pyspark.sql import SparkSession
from src import config
from src.utils import get_logger

logger = get_logger("spark_session")

def setup_spark_windows_env():
    """
    Mengonfigurasi environment variables untuk menghindari masalah path di Windows.
    Menunjuk ke Spark yang terpasang di virtual environment (.venv) dan hadoop binaries.
    """
    try:
        # Tentukan path .venv relatif terhadap file ini (src/spark_session.py)
        src_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(src_dir, ".."))
        venv_dir = os.path.join(project_root, ".venv")
        hadoop_dir = os.path.join(project_root, "hadoop")
        
        # Setup HADOOP_HOME jika binari hadoop ada
        if os.path.exists(hadoop_dir):
            os.environ["HADOOP_HOME"] = hadoop_dir
            hadoop_bin = os.path.join(hadoop_dir, "bin")
            os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ["PATH"]
            logger.debug(f"Spark Windows env configured: HADOOP_HOME={hadoop_dir}")
        else:
            logger.warning("Direktori hadoop binaries tidak ditemukan. Spark write mungkin gagal di Windows.")
        
        if os.path.exists(venv_dir):
            venv_scripts = os.path.join(venv_dir, "Scripts")
            pyspark_home = os.path.join(venv_dir, "Lib", "site-packages", "pyspark")
            
            # Setup environment variables
            os.environ["PATH"] = venv_scripts + os.pathsep + os.environ["PATH"]
            os.environ["PYSPARK_PYTHON"] = sys.executable  # Gunakan python interpreter aktif
            os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
            os.environ["SPARK_HOME"] = pyspark_home
            
            logger.debug(f"Spark Windows env configured: SPARK_HOME={pyspark_home}")
        else:
            logger.warning("Direktori .venv tidak ditemukan di root project. Menggunakan default system environment.")
    except Exception as e:
        logger.error(f"Gagal mengatur environment Spark di Windows: {str(e)}")

def get_spark_session() -> SparkSession:
    """
    Membuat atau mengambil SparkSession yang dikonfigurasi terpusat.
    """
    setup_spark_windows_env()
    
    # Dapatkan path mutlak untuk direktori temporer Spark
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(src_dir, ".."))
    
    spark_temp = os.path.abspath(os.path.join(project_root, config.SPARK_TEMP_DIR))
    spark_warehouse = os.path.abspath(os.path.join(project_root, config.SPARK_WAREHOUSE_DIR))
    
    # Buat direktori jika belum ada
    os.makedirs(spark_temp, exist_ok=True)
    os.makedirs(spark_warehouse, exist_ok=True)
    
    logger.info("Menginisialisasi Apache Spark Session (Resource-Limited to prevent freezing)...")
    
    spark = SparkSession.builder \
        .master("local[2]") \
        .appName(config.SPARK_APP_NAME) \
        .config("spark.driver.memory", "3g") \
        .config("spark.executor.memory", "3g") \
        .config("spark.local.dir", spark_temp) \
        .config("spark.sql.warehouse.dir", spark_warehouse) \
        .config("spark.sql.shuffle.partitions", str(config.SPARK_SHUFFLE_PARTS)) \
        .config("spark.sql.broadcastTimeout", "600") \
        .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
        .config("spark.shuffle.sort.bypassMergeThreshold", "0") \
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")
    logger.info(f"Spark Session berhasil diinisialisasi. App Name: {config.SPARK_APP_NAME}")
    return spark
