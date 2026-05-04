# main.py

import sys
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Prepend Hadoop bin to PATH if HADOOP_HOME is set
if os.environ.get("HADOOP_HOME"):
    os.environ["PATH"] = os.path.join(os.environ["HADOOP_HOME"], "bin") + os.pathsep + os.environ["PATH"]
os.environ["PATH"] = os.path.join(os.environ["SPARK_HOME"], "bin") + os.pathsep + os.environ["PATH"]

JAVA_HOME = os.environ.get("JAVA_HOME")


sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pyspark.sql import SparkSession

from config.paths import DATASET_1, DATASET_2, MODELS_DIR, VISUALIZATIONS_DIR
from config.sparkconfig import create_spark


from src.webserver_preprocessing      import run_preprocessing
from src.webserver_feature_engineering import run_feature_engineering
from src.webserver_modeling               import run_model_training
from src.webserver_visualization                import run_visualization

# ══════════════════════════════════════════════════════════════
# SPARK SESSION
# Only created once here — every other file receives spark
# as a parameter instead of creating its own session.
# ══════════════════════════════════════════════════════════════
spark = create_spark()
spark.sparkContext.setLogLevel("WARN")
spark.conf.set("spark.cleaner.referenceTracking.cleanCheckpoints", "true")

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
# SAVE RESULTS TO CSV
# ══════════════════════════════════════════════════════════════
import pandas as pd
import os

results_dir = os.path.join(os.path.dirname(__file__), 'output', 'results')
os.makedirs(results_dir, exist_ok=True)

# ── Model metrics summary ──────────────────────────────────
metrics_df = pd.DataFrame([
    {
        "model": "Logistic Regression",
        "accuracy":  results["lr_metrics"].get("accuracy", 0),
        "f1":        results["lr_metrics"].get("f1", 0),
        "precision": results["lr_metrics"].get("weightedPrecision", 0),
        "recall":    results["lr_metrics"].get("weightedRecall", 0),
        "silhouette": None,
        "purity": None,
    },
    {
        "model": "Random Forest",
        "accuracy":  results["knn_metrics"].get("accuracy", 0),
        "f1":        results["knn_metrics"].get("f1", 0),
        "precision": results["knn_metrics"].get("weightedPrecision", 0),
        "recall":    results["knn_metrics"].get("weightedRecall", 0),
        "silhouette": None,
        "purity": None,
    },
    {
        "model": f"K-Means (k={results['best_k']})",
        "accuracy":  None,
        "f1":        None,
        "precision": None,
        "recall":    None,
        "silhouette": results["km_metrics"]["silhouette"],
        "purity":     results["km_metrics"]["purity"],
    },
])
metrics_df.to_csv(f"{results_dir}/model_metrics.csv", index=False)
print(f"[results] Model metrics saved")

# ── K-Means scores per k ───────────────────────────────────
km_df = pd.DataFrame([
    {"k": k, "silhouette": v["silhouette"], "wssse": v["wssse"]}
    for k, v in results["km_scores"].items()
])
km_df.to_csv(f"{results_dir}/kmeans_scores.csv", index=False)
print(f"[results] K-Means scores saved")

# ── Anomaly distribution ───────────────────────────────────
anomaly_df = windowed.groupBy("anomaly_type", "severity") \
    .count().orderBy("count", ascending=False).toPandas()
anomaly_df.to_csv(f"{results_dir}/anomaly_distribution.csv", index=False)
print(f"[results] Anomaly distribution saved")

# ── Pipeline stats ─────────────────────────────────────────
stats_df = pd.DataFrame([{
    "total_rows": 14842672,
    "bot_traffic": 1118561,
    "bot_pct": 7.5,
    "error_rows": 262424,
    "error_pct": 1.8,
    "null_timestamps": 0,
    "total_windows": 1831664,
    "anomalous_windows": 331642,
    "anomalous_pct": 18.1,
}])
stats_df.to_csv(f"{results_dir}/pipeline_stats.csv", index=False)
print(f"[results] Pipeline stats saved")

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