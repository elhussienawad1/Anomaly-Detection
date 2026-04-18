import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.webserver_feature_engineering import windowed
from src.webserver_evaluation import (
    evaluate_supervised,
    evaluate_kmeans,
    evaluate_rule_based_baseline,
)

# ══════════════════════════════════════════════════════════════════════════════
# model_training.py  —  MLlib Anomaly Detection Pipeline
# Covers: Feature assembly, Logistic Regression, KNN (via ALS approx),
#         K-Means clustering, train/val/test split, and full evaluation.
# Depends on: windowed DataFrame produced by preprocessing.py
# ══════════════════════════════════════════════════════════════════════════════

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler, StringIndexer,
    OneHotEncoder, IndexToString
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    ClusteringEvaluator
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col, when, lit, count
from pyspark.sql import functions as F
import os

# ──────────────────────────────────────────────────────────────────────────────
# NOTE on KNN in MLlib
# Native Spark MLlib does not ship a KNNClassifier.  Two production options:
#   Option A (recommended): spark-knn  — pip install spark-knn
#                           from pyspark.ml.classification import KNNClassifier
#   Option B (fallback, pure MLlib): approximate KNN via BallTree on each
#           executor using a UDF + BroadcastNestedLoopJoin, or simply replace
#           with RandomForestClassifier which is a strong baseline.
# This file uses Option A with an automatic fallback to RandomForest.
# ──────────────────────────────────────────────────────────────────────────────

try:
    from pyspark.ml.classification import KNNClassifier
    _HAS_KNN = True
except ImportError:
    from pyspark.ml.classification import RandomForestClassifier as KNNClassifier
    _HAS_KNN = False
    print("[WARN] spark-knn not found — substituting RandomForestClassifier for KNN step.")

# ══════════════════════════════════════════════════════════════════════════════
# 0.  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

LABEL_COL        = "label"          # integer label produced by StringIndexer
LABEL_STR_COL    = "anomaly_type"   # original string column
FEATURES_COL     = "features"       # scaled feature vector
RAW_FEATURES_COL = "raw_features"   # pre-scale assembly

NUMERIC_FEATURES = [
    "request_count",
    "error_count",
    "avg_bytes",
    "unique_endpoints",
    "bot_requests",
    "off_hours_requests",
    "fatal_count",
    "medium_count",
    "low_count",
    "error_rate",
]

SEED = 42

OUTPUT_DIR = "/home/abdallah/Anomaly-Detection/models"

# ══════════════════════════════════════════════════════════════════════════════
# 1.  PREPARE DATA FROM windowed
#     windowed is the DataFrame produced at the end of preprocessing.py.
#     We add a numeric label and handle any nulls before modelling.
# ══════════════════════════════════════════════════════════════════════════════

def prepare_data(windowed):
    """
    Clean the windowed aggregate DataFrame and return a modelling-ready DF.
    Drops rows with null timestamps (window edges) and fills numeric nulls.
    """
    df = windowed.select(
        *[col(c) for c in NUMERIC_FEATURES],
        col(LABEL_STR_COL),
        col("severity"),
        col("severity_score"),
        col("ip"),
        col("window"),
    )

    # fill numeric nulls (rare, but window edges can produce them)
    df = df.fillna(0, subset=NUMERIC_FEATURES)

    # cap error_rate at 1.0 to be safe
    df = df.withColumn("error_rate", when(col("error_rate") > 1.0, 1.0)
                                     .otherwise(col("error_rate")))

    print(f"[prepare_data] Total windows: {df.count()}")
    df.groupBy(LABEL_STR_COL).count().orderBy("count", ascending=False).show()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE PIPELINE
#     StringIndexer → VectorAssembler → StandardScaler
#     The scaler output becomes the canonical FEATURES_COL used by all models.
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_pipeline():
    """
    Returns a fitted Pipeline stage list (not yet a fitted model — call
    feature_pipeline.fit(df) on the training split).
    """

    # 2a. Encode target label as integer index
    label_indexer = StringIndexer(
        inputCol=LABEL_STR_COL,
        outputCol=LABEL_COL,
        handleInvalid="keep",   # unseen labels → extra index rather than error
        stringOrderType="frequencyDesc",
    )

    # 2b. Assemble all numeric features into one dense vector
    assembler = VectorAssembler(
        inputCols=NUMERIC_FEATURES,
        outputCol=RAW_FEATURES_COL,
        handleInvalid="skip",
    )

    # 2c. Z-score normalise (important for LR convergence and KNN distance)
    scaler = StandardScaler(
        inputCol=RAW_FEATURES_COL,
        outputCol=FEATURES_COL,
        withMean=True,
        withStd=True,
    )

    return Pipeline(stages=[label_indexer, assembler, scaler])


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TRAIN / VALIDATION / TEST SPLIT
#     60 % train  |  20 % validation  |  20 % test
# ══════════════════════════════════════════════════════════════════════════════

def split_data(df):
    train_val, test  = df.randomSplit([0.80, 0.20], seed=SEED)
    train,     val   = train_val.randomSplit([0.75, 0.25], seed=SEED)  # 60/20/20

    print(f"[split] train={train.count()}  val={val.count()}  test={test.count()}")
    return train, val, test


# ══════════════════════════════════════════════════════════════════════════════
# 4.  LOGISTIC REGRESSION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_lr_pipeline(feature_pipeline_model):
    """
    Builds and returns an unfitted LR estimator that expects the output of
    the already-fitted feature_pipeline_model (i.e., data already has FEATURES_COL).
    """
    lr = LogisticRegression(
        featuresCol=FEATURES_COL,
        labelCol=LABEL_COL,
        maxIter=100,
        regParam=0.01,          # L2 regularisation
        elasticNetParam=0.0,    # pure L2
        family="multinomial",   # multi-class
        standardization=False,  # already scaled
    )
    return lr


def train_logistic_regression(train_transformed, val_transformed, num_classes):
    """
    Trains LR with a small hyperparameter grid on the validation set.
    Returns the best fitted LR model.
    """
    lr = LogisticRegression(
        featuresCol=FEATURES_COL,
        labelCol=LABEL_COL,
        family="multinomial",
        standardization=False,
        maxIter=100,
    )

    param_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam,        [0.001, 0.01, 0.1])
        .addGrid(lr.elasticNetParam, [0.0,   0.5])
        .build()
    )

    evaluator = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL,
        predictionCol="prediction",
        metricName="f1",
    )

    # Manual val-set selection (faster than full CrossValidator for large data)
    best_model  = None
    best_f1     = -1.0
    best_params = {}

    for params in param_grid:
        model = lr.copy(params).fit(train_transformed)
        preds = model.transform(val_transformed)
        f1    = evaluator.evaluate(preds)
        rp    = params[lr.regParam]
        en    = params[lr.elasticNetParam]
        print(f"  LR regParam={rp:.3f}  elasticNet={en:.1f}  val_F1={f1:.4f}")
        if f1 > best_f1:
            best_f1     = f1
            best_model  = model
            best_params = {"regParam": rp, "elasticNetParam": en}

    print(f"[LR] Best val F1={best_f1:.4f}  params={best_params}")
    return best_model


# ══════════════════════════════════════════════════════════════════════════════
# 5.  KNN (or RandomForest fallback) PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def train_knn(train_transformed, val_transformed):
    """
    Trains KNN (spark-knn) or RandomForest fallback.
    Returns the fitted model.
    """
    if _HAS_KNN:
        knn = KNNClassifier(
            featuresCol=FEATURES_COL,
            labelCol=LABEL_COL,
            k=5,
            distanceType="euclidean",
        )
        param_grid = (
            ParamGridBuilder()
            .addGrid(knn.k, [3, 5, 11])
            .build()
        )
    else:
        # RandomForest fallback — interpretable, handles multi-class natively
        from pyspark.ml.classification import RandomForestClassifier
        knn = RandomForestClassifier(
            featuresCol=FEATURES_COL,
            labelCol=LABEL_COL,
            numTrees=100,
            maxDepth=10,
            seed=SEED,
        )
        param_grid = (
            ParamGridBuilder()
            .addGrid(knn.numTrees, [50, 100])
            .addGrid(knn.maxDepth, [5,  10])
            .build()
        )

    evaluator = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL,
        predictionCol="prediction",
        metricName="f1",
    )

    best_model = None
    best_f1    = -1.0

    for params in param_grid:
        model = knn.copy(params).fit(train_transformed)
        preds = model.transform(val_transformed)
        f1    = evaluator.evaluate(preds)
        print(f"  KNN/RF params={params}  val_F1={f1:.4f}")
        if f1 > best_f1:
            best_f1    = f1
            best_model = model

    print(f"[KNN/RF] Best val F1={best_f1:.4f}")
    return best_model


# ══════════════════════════════════════════════════════════════════════════════
# 6.  K-MEANS CLUSTERING (unsupervised — uses FEATURES_COL, ignores label)
# ══════════════════════════════════════════════════════════════════════════════

def train_kmeans(train_transformed, val_transformed, k_range=range(3, 9)):
    """
    Trains K-Means for each k in k_range, picks the best k by silhouette score
    on the validation set, returns (best_model, best_k, silhouette_scores_dict).
    """
    evaluator = ClusteringEvaluator(
        featuresCol=FEATURES_COL,
        predictionCol="prediction",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )

    scores    = {}
    best_k    = None
    best_sil  = -1.0
    best_model = None

    for k in k_range:
        km = KMeans(
            featuresCol=FEATURES_COL,
            predictionCol="prediction",
            k=k,
            maxIter=30,
            seed=SEED,
            initMode="k-means||",   # robust initialisation
        )
        model  = km.fit(train_transformed)
        preds  = model.transform(val_transformed)
        sil    = evaluator.evaluate(preds)
        wssse  = model.summary.trainingCost
        scores[k] = {"silhouette": sil, "wssse": wssse}
        print(f"  K-Means k={k}  silhouette={sil:.4f}  WSSSE={wssse:.2f}")

        if sil > best_sil:
            best_sil   = sil
            best_k     = k
            best_model = model

    print(f"[KMeans] Best k={best_k}  silhouette={best_sil:.4f}")
    return best_model, best_k, scores


# ══════════════════════════════════════════════════════════════════════════════
# 10.  SAVE MODELS
# ══════════════════════════════════════════════════════════════════════════════

def save_models(feature_model, lr_model, knn_model, km_model, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    feature_model.write().overwrite().save(f"{output_dir}/feature_pipeline")
    lr_model.write().overwrite().save(f"{output_dir}/logistic_regression")
    knn_model.write().overwrite().save(f"{output_dir}/knn_or_rf")
    km_model.write().overwrite().save(f"{output_dir}/kmeans")
    print(f"[save] All models written to {output_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# 11.  MAIN ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def run_model_training(windowed):
    """
    End-to-end training entry point.
    Pass in the `windowed` DataFrame from preprocessing.py.

    Returns dict with all fitted models and test metrics.
    """

    # ── 11.1  Prepare ────────────────────────────────────────────────────────
    df = prepare_data(windowed)

    # ── 11.2  Split (before fitting anything to avoid leakage) ───────────────
    train, val, test = split_data(df)

    # ── 11.3  Fit feature pipeline ON TRAIN ONLY ─────────────────────────────
    feature_pipeline = build_feature_pipeline()
    print("[Pipeline] Fitting feature pipeline on train split…")
    feature_model    = feature_pipeline.fit(train)

    train_t = feature_model.transform(train)
    val_t   = feature_model.transform(val)
    test_t  = feature_model.transform(test)

    # Retrieve the learned label mapping (needed for per-class reporting)
    label_indexer_model = feature_model.stages[0]          # StringIndexer stage
    label_names         = label_indexer_model.labels        # list of strings in index order
    num_classes         = len(label_names)
    print(f"[Pipeline] {num_classes} classes: {label_names}")

    # ── 11.4  Logistic Regression ─────────────────────────────────────────────
    print("\n" + "═"*60)
    print("TRAINING — Logistic Regression")
    print("═"*60)
    lr_model = train_logistic_regression(train_t, val_t, num_classes)

    # ── 11.5  KNN / RandomForest ──────────────────────────────────────────────
    print("\n" + "═"*60)
    print("TRAINING — KNN (or RandomForest fallback)")
    print("═"*60)
    knn_model = train_knn(train_t, val_t)

    # ── 11.6  K-Means ─────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("TRAINING — K-Means Clustering  (k = 3 … 8)")
    print("═"*60)
    km_model, best_k, km_scores = train_kmeans(
        train_t, val_t, k_range=range(3, 9)
    )

    # ── 11.7  Evaluation on held-out TEST set ─────────────────────────────────
    print("\n" + "═"*60)
    print("EVALUATION — Test Set")
    print("═"*60)

    evaluate_rule_based_baseline(test_t, label_names)

    print("\n--- Logistic Regression ---")
    lr_metrics  = evaluate_supervised(lr_model,  test_t, label_names, "LR")

    print("\n--- KNN / RandomForest ---")
    knn_metrics = evaluate_supervised(knn_model, test_t, label_names, "KNN")

    print("\n--- K-Means ---")
    km_metrics  = evaluate_kmeans(km_model, test_t)

    # ── 11.8  Summary table ───────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("SUMMARY")
    print("═"*60)
    print(f"{'Model':<25} {'Accuracy':>9} {'F1 (wtd)':>10} {'Precision':>10} {'Recall':>10}")
    print("-"*65)
    for name, m in [("Logistic Regression", lr_metrics), ("KNN / RF", knn_metrics)]:
        print(
            f"{name:<25} "
            f"{m.get('accuracy', 0):>9.4f} "
            f"{m.get('f1', 0):>10.4f} "
            f"{m.get('weightedPrecision', 0):>10.4f} "
            f"{m.get('weightedRecall', 0):>10.4f}"
        )
    print(f"\nK-Means  best_k={best_k}  silhouette={km_metrics['silhouette']:.4f}  purity={km_metrics['purity']:.4f}")

    # ── 11.9  Save ────────────────────────────────────────────────────────────
    save_models(feature_model, lr_model, knn_model, km_model)

    return {
        "feature_model":  feature_model,
        "lr_model":       lr_model,
        "knn_model":      knn_model,
        "km_model":       km_model,
        "label_names":    label_names,
        "lr_metrics":     lr_metrics,
        "knn_metrics":    knn_metrics,
        "km_metrics":     km_metrics,
        "km_scores":      km_scores,
        "best_k":         best_k,
        "test_t":         test_t,
    }


windowed_sample = windowed \
    .localCheckpoint()   # breaks the lineage — no Hadoop needed

print("DEBUG sample count:", windowed_sample.count())

results = run_model_training(windowed_sample)   # pass full dataset
