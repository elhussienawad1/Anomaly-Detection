# main.py

import sys
import os

# ── make project root importable ──────────────────────────
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pyspark.sql import SparkSession
from config.paths import DATASET_1, DATASET_2, MODELS_DIR, VISUALIZATIONS_DIR

from src.webserver_preprocessing      import run_preprocessing
from src.webserver_feature_engineering import run_feature_engineering
from src.webserver_modeling               import run_model_training
from src.webserver_visualization                import run_visualization

# ══════════════════════════════════════════════════════════════
# SPARK SESSION
# Only created once here — every other file receives spark
# as a parameter instead of creating its own session.
# ══════════════════════════════════════════════════════════════
spark = SparkSession.builder \
    .appName("LogAnomalyDetection") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ══════════════════════════════════════════════════════════════
# STAGE 1 — PREPROCESSING
# Parses raw log files → unified clean DataFrame
# Output: df_all (14.8M rows, 11 columns)
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("STAGE 1 — PREPROCESSING")
print("═"*60)

df = run_preprocessing(spark, DATASET_1, DATASET_2)

# cache df — it's used by both feature engineering and visualization
df.cache()
df.count()  # materialize cache now so downstream stages are fast

# ══════════════════════════════════════════════════════════════
# STAGE 2 — FEATURE ENGINEERING
# Enriches rows + aggregates per IP per minute window
# Output: df_enriched (row-level), windowed (IP-minute aggregates)
# windowed is what the models train on
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("STAGE 2 — FEATURE ENGINEERING")
print("═"*60)

df_enriched, windowed = run_feature_engineering(df)

# cache windowed — used by modeling and visualization
windowed.cache()
windowed.count()

# ══════════════════════════════════════════════════════════════
# STAGE 3 — MODELING & EVALUATION
# Trains K-Means, Logistic Regression, KNN/RandomForest
# Evaluates on held-out test set
# Output: fitted models saved to output/models/
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("STAGE 3 — MODELING & EVALUATION")
print("═"*60)

results = run_model_training(windowed)

# ══════════════════════════════════════════════════════════════
# STAGE 4 — VISUALIZATION
# Generates all 20 plots from the 3 DataFrames
# Output: PNG files saved to output/visualizations/
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("STAGE 4 — VISUALIZATION")
print("═"*60)

run_visualization(df, df_enriched, windowed, VISUALIZATIONS_DIR)

# ══════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("PIPELINE COMPLETE")
print("═"*60)
print(f"Models saved to      : {MODELS_DIR}")
print(f"Visualizations saved : {VISUALIZATIONS_DIR}")
print(f"\nModel summary:")
print(f"  LR  — F1: {results['lr_metrics'].get('f1', 0):.4f}  Accuracy: {results['lr_metrics'].get('accuracy', 0):.4f}")
print(f"  KNN — F1: {results['knn_metrics'].get('f1', 0):.4f}  Accuracy: {results['knn_metrics'].get('accuracy', 0):.4f}")
print(f"  K-Means — Silhouette: {results['km_metrics']['silhouette']:.4f}  Purity: {results['km_metrics']['purity']:.4f}")

spark.stop()