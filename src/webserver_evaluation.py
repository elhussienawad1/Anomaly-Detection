import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    ClusteringEvaluator
)
from pyspark.sql.functions import col
from pyspark.sql import functions as F

LABEL_COL        = "label"
LABEL_STR_COL    = "anomaly_type"
FEATURES_COL     = "features"

# ══════════════════════════════════════════════════════════════════════════════
# 7.  EVALUATION — SUPERVISED MODELS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_supervised(model, test_transformed, label_names, model_name):
    """
    Prints accuracy, weighted F1, weighted precision/recall on the test set.
    Also shows per-class breakdown via confusion-matrix style groupBy.
    Returns a dict of metrics.
    """
    preds = model.transform(test_transformed)

    ev = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction"
    )

    metrics = {}
    for metric in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
        val = ev.setMetricName(metric).evaluate(preds)
        metrics[metric] = val
        print(f"  [{model_name}] {metric:<22}: {val:.4f}")

    # Per-class F1 (computed manually from confusion matrix)
    print(f"\n  [{model_name}] Per-class F1 (precision / recall / F1 per class):")
    print(f"    {'Class':<20} {'TP':>7} {'FP':>7} {'FN':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print(f"    {'-'*62}")
    for i, name in enumerate(label_names):
        tp = preds.filter((col(LABEL_COL) == i) & (col("prediction") == i)).count()
        fp = preds.filter((col(LABEL_COL) != i) & (col("prediction") == i)).count()
        fn = preds.filter((col(LABEL_COL) == i) & (col("prediction") != i)).count()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_i = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"    {name:<20} {tp:>7} {fp:>7} {fn:>7} {prec:>7.4f} {rec:>7.4f} {f1_i:>7.4f}")

    # Confusion matrix
    print(f"\n  [{model_name}] Confusion matrix (label vs prediction count):")
    preds.groupBy(LABEL_COL, "prediction") \
         .count() \
         .orderBy(LABEL_COL, "prediction") \
         .show(50)

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 8.  EVALUATION — K-MEANS (unsupervised)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_kmeans(model, test_transformed, label_col=LABEL_COL):
    """
    Silhouette score + cluster purity (how well clusters align with true labels).
    Purity = sum over clusters of the majority class count / total points.
    """
    preds = model.transform(test_transformed)

    evaluator = ClusteringEvaluator(
        featuresCol=FEATURES_COL,
        predictionCol="prediction",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )
    sil = evaluator.evaluate(preds)
    print(f"  [KMeans] Test silhouette: {sil:.4f}")

    # Cluster composition
    print("  [KMeans] Cluster vs anomaly_type distribution:")
    preds.groupBy("prediction", LABEL_STR_COL) \
         .count() \
         .orderBy("prediction", "count", ascending=[True, False]) \
         .show(50)

    # Purity calculation
    total = preds.count()
    majority_counts = (
        preds.groupBy("prediction", LABEL_STR_COL)
             .count()
             .groupBy("prediction")
             .agg(F.max("count").alias("majority_count"))
    )
    purity = majority_counts.agg(F.sum("majority_count")).collect()[0][0] / total
    print(f"  [KMeans] Cluster purity: {purity:.4f}  (1.0 = perfect alignment)")

    return {"silhouette": sil, "purity": purity}


# ══════════════════════════════════════════════════════════════════════════════
# 9.  COMPARE AGAINST RULE-BASED BASELINE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_rule_based_baseline(test_transformed, label_names):
    """
    The rule-based anomaly_type column IS the baseline 'prediction'.
    Re-index it using the same label mapping and compute F1.
    This shows how much the ML models improve over hard-coded rules.
    """
    # Map anomaly_type string → same integer index used in ML models
    # (relies on LABEL_COL already present from StringIndexer in feature pipeline)
    preds = test_transformed.withColumn("prediction", col(LABEL_COL))  # perfect — upper bound
    # For a real baseline, we'd re-index anomaly_type; here we note rule-based
    # IS the label source, so "baseline accuracy" == model accuracy on rule labels.
    print("\n[Baseline] Rule-based labels are the ground truth for supervised models.")
    print("  Baseline F1 = 1.0 by definition (labels come from rules).")
    print("  The value of ML models is generalisation + probability scores,")
    print("  not beating a baseline that IS the label.")
    print("  For unsupervised K-Means, cluster purity is the meaningful metric.\n")
