from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    regexp_extract, col, to_timestamp, when, lower, lit
)
import sys
import os

# explicitly point to project root regardless of where script is called from
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from config.schema import WEB_ACCESS_LOG_SCHEMA


from config.schema import WEB_ACCESS_LOG_SCHEMA

spark = SparkSession.builder \
    .appName("LogAnomalyDetectionForApplications") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
    .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ══════════════════════════════════════════════════════════════
# DATASET 1 — Apache Common Log Format (no referrer/user agent)
# Source: config/datasets/access_log.txt
# Example lines:
# 10.223.157.186 - - [15/Jul/2009:14:58:59 -0700] "GET / HTTP/1.1" 403 202
# 10.223.157.186 - - [15/Jul/2009:15:50:35 -0700] "GET /assets/js/lowpro.js HTTP/1.1" 200 10469
# Format: IP ident user [timestamp] "METHOD endpoint protocol" status bytes
# ══════════════════════════════════════════════════════════════

LOG_PATTERN_BASIC = (
    r'^(\S+)'            # group 1: IP address
    r'\s+\S+'            # ident (usually -)
    r'\s+\S+'            # user (usually -)
    r'\s+\[([^\]]+)\]'  # group 2: timestamp
    r'\s+"(\S+)'         # group 3: HTTP method
    r'\s+(\S+)'          # group 4: endpoint
    r'\s+\S+"'           # protocol (no group)
    r'\s+(\d{3})'        # group 5: status code
    r'\s+(\S+)'          # group 6: bytes
)

# ══════════════════════════════════════════════════════════════
# DATASET 2 — Apache Combined Log Format (has referrer + user agent)
# Source: config/datasets/web_server_access_log.txt
# Example lines:
# 54.36.149.41  - - [22/Jan/2019:03:56:14 +0330] "GET /filter/27 HTTP/1.1" 200 30577 "-" "AhrefsBot/6.1" "-"
# 31.56.96.51   - - [22/Jan/2019:03:56:16 +0330] "GET /image/60844 HTTP/1.1" 200 5667 "https://zanbil.ir" "Chrome/66.0" "-"
# Format: IP ident user [timestamp] "METHOD endpoint protocol" status bytes "referrer" "user_agent" "extra"
# ══════════════════════════════════════════════════════════════

LOG_PATTERN_COMBINED = (
    r'^(\S+)'            # group 1: IP address
    r'\s+\S+'            # ident (usually -)
    r'\s+\S+'            # user (usually -)
    r'\s+\[([^\]]+)\]'  # group 2: timestamp
    r'\s+"(\S+)'         # group 3: HTTP method
    r'\s+(\S+)'          # group 4: endpoint
    r'\s+\S+"'           # protocol (no group)
    r'\s+(\d{3})'        # group 5: status code
    r'\s+(\S+)'          # group 6: bytes
    r'\s+"([^"]*)"'      # group 7: referrer  (ignored)
    r'\s+"([^"]*)"'      # group 8: user agent
)

BOT_KEYWORDS = (
    "bot|crawler|spider|scraper|ahrefsbot|bingbot|googlebot"
    "|slurp|curl|wget|python-requests|java|go-http"
)

# ══════════════════════════════════════════════════════════════
# SHARED CLEANING — applied to both datasets after parsing
# ══════════════════════════════════════════════════════════════

def clean(df):
    return df \
        .withColumn(
            "timestamp",
            to_timestamp("timestamp_raw", "dd/MMM/yyyy:HH:mm:ss Z")
        ) \
        .withColumn(
            "bytes",
            when(col("bytes_raw") == "-", 0)
            .otherwise(col("bytes_raw").cast("int"))
        ) \
        .withColumn(
            "is_error",
            when(col("status_code") >= 400, 1).otherwise(0)
        ) \
        .withColumn(
            "status_class",
            when(col("status_code") < 200, "1xx")
            .when(col("status_code") < 300, "2xx")
            .when(col("status_code") < 400, "3xx")
            .when(col("status_code") < 500, "4xx")
            .otherwise("5xx")
        ) \
        .drop("timestamp_raw", "bytes_raw") \
        .filter(col("ip") != "")  # drop malformed lines

# ══════════════════════════════════════════════════════════════
# SCHEMA VALIDATION
# ══════════════════════════════════════════════════════════════

def validate_schema(df, name):
    expected_cols = set(f.name for f in WEB_ACCESS_LOG_SCHEMA.fields)
    actual_cols   = set(df.columns)
    missing  = expected_cols - actual_cols
    extra    = actual_cols - expected_cols
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}")
    if extra:
        print(f"[{name}] WARNING — unexpected extra columns: {extra}")
    print(f"[{name}] Schema validation passed.")

# ══════════════════════════════════════════════════════════════
# PARSE & CLEAN DATASET 1
# ══════════════════════════════════════════════════════════════

df_raw_1 = spark.read.text("C:/Users/VICTUS/Desktop/Engineering/Sem 8/Big Data/access_log.txt")

df_parsed_1 = df_raw_1.select(
    regexp_extract("value", LOG_PATTERN_BASIC, 1).alias("ip"),
    regexp_extract("value", LOG_PATTERN_BASIC, 2).alias("timestamp_raw"),
    regexp_extract("value", LOG_PATTERN_BASIC, 3).alias("method"),
    regexp_extract("value", LOG_PATTERN_BASIC, 4).alias("endpoint"),
    regexp_extract("value", LOG_PATTERN_BASIC, 5).cast("int").alias("status_code"),
    regexp_extract("value", LOG_PATTERN_BASIC, 6).alias("bytes_raw"),
    lit(None).cast("string").alias("user_agent"),  # not present in this format
    lit(0).alias("is_bot"),                         # can't detect without user agent
    lit("access_log").alias("source"),
)

df_clean_1 = clean(df_parsed_1)

# ══════════════════════════════════════════════════════════════
# PARSE & CLEAN DATASET 2
# ══════════════════════════════════════════════════════════════

df_raw_2 = spark.read.text("C:/Users/VICTUS/Desktop/Engineering/Sem 8/Big Data/access.log")

df_parsed_2 = df_raw_2.select(
    regexp_extract("value", LOG_PATTERN_COMBINED, 1).alias("ip"),
    regexp_extract("value", LOG_PATTERN_COMBINED, 2).alias("timestamp_raw"),
    regexp_extract("value", LOG_PATTERN_COMBINED, 3).alias("method"),
    regexp_extract("value", LOG_PATTERN_COMBINED, 4).alias("endpoint"),
    regexp_extract("value", LOG_PATTERN_COMBINED, 5).cast("int").alias("status_code"),
    regexp_extract("value", LOG_PATTERN_COMBINED, 6).alias("bytes_raw"),
    regexp_extract("value", LOG_PATTERN_COMBINED, 8).alias("user_agent"),
    lit("web_server_log").alias("source"),
).withColumn(
    "is_bot",
    when(lower(col("user_agent")).rlike(BOT_KEYWORDS), 1).otherwise(0)
)

df_clean_2 = clean(df_parsed_2)

# ══════════════════════════════════════════════════════════════
# MERGE & VALIDATE
# ══════════════════════════════════════════════════════════════

df_all = df_clean_1.unionByName(df_clean_2)

validate_schema(df_all, "df_all")

# ── Sanity check ───────────────────────────────────────────
total  = df_all.count()
bots   = df_all.filter(col("is_bot")   == 1).count()
errors = df_all.filter(col("is_error") == 1).count()
nulls  = df_all.filter(col("timestamp").isNull()).count()

print(f"Total rows    : {total}")
print(f"Bot traffic   : {bots}  ({100*bots/total:.1f}%)")
print(f"Error rows    : {errors}  ({100*errors/total:.1f}%)")
print(f"Null timestamp: {nulls}  ({100*nulls/total:.1f}%)  ← should be near 0")

df_all.printSchema()
df_all.show(10, truncate=False)

# check both sources are present
df_all.groupBy("source").count().show()

# show some rows specifically from dataset 2
df_all.filter(col("source") == "web_server_log").show(10, truncate=False)

# check both sources are present
df_all.groupBy("source").count().show()

# show some rows specifically from dataset 2
df_all.filter(col("source") == "web_server_log").show(10, truncate=False)

# ══════════════════════════════════════════════════════════════
# STEP 1 — SANITY CHECKS
# ══════════════════════════════════════════════════════════════

print("=== Status Code Distribution ===")
df_all.groupBy("status_code").count() \
    .orderBy("count", ascending=False).show(20)

print("=== Bot vs Human by Source ===")
df_all.groupBy("is_bot", "source").count().show()

print("=== Time Range ===")
from pyspark.sql.functions import min, max
df_all.select(
    min("timestamp").alias("earliest"),
    max("timestamp").alias("latest")
).show()
# ══════════════════════════════════════════════════════════════
# STEP 2 — ENRICH INDIVIDUAL ROWS
# ══════════════════════════════════════════════════════════════

from pyspark.sql.functions import (
    window, count, sum, avg, countDistinct,
    hour, dayofweek
)

df_enriched = df_all \
    .withColumn(
        "error_severity",
        when(col("status_code").isin(500, 502, 504), "fatal")
        .when(col("status_code").isin(499, 401, 403, 429), "medium")
        .when(col("status_code").isin(400, 404, 405, 416), "low")
        .otherwise("none")
    ) \
    .withColumn("hour_of_day", hour(col("timestamp"))) \
    .withColumn("day_of_week", dayofweek(col("timestamp"))) \
    .withColumn(
        "is_off_hours",
        when((col("hour_of_day") < 8) | (col("hour_of_day") > 22), 1).otherwise(0)
    )
df_enriched = df_enriched.repartition(200)
df_enriched.cache()
df_enriched.count()  # forces Spark to materialize the cache now

# ══════════════════════════════════════════════════════════════
# STEP 3 — AGGREGATE PER IP PER 1-MINUTE WINDOW
# ══════════════════════════════════════════════════════════════

windowed = df_enriched.groupBy(
    col("ip"),
    window(col("timestamp"), "1 minute")
).agg(
    count("*").alias("request_count"),
    sum("is_error").alias("error_count"),
    avg("bytes").alias("avg_bytes"),
    countDistinct("endpoint").alias("unique_endpoints"),
    sum("is_bot").alias("bot_requests"),
    sum("is_off_hours").alias("off_hours_requests"),
    sum(when(col("error_severity") == "fatal",  1).otherwise(0)).alias("fatal_count"),
    sum(when(col("error_severity") == "medium", 1).otherwise(0)).alias("medium_count"),
    sum(when(col("error_severity") == "low",    1).otherwise(0)).alias("low_count"),
).withColumn(
    "error_rate",
    col("error_count") / col("request_count")
)

# ══════════════════════════════════════════════════════════════
# STEP 4 — LABEL ANOMALY TYPE AND SEVERITY
# ══════════════════════════════════════════════════════════════

windowed = windowed \
    .withColumn(
        "anomaly_type",
        when(col("fatal_count") > 0,          "server_error")
        .when(col("request_count") > 200,     "traffic_spike")
        .when(col("error_rate") > 0.3,        "high_error_rate")
        .when(col("unique_endpoints") > 30,   "scanning")
        .when(col("bot_requests") > 0,        "bot_traffic")
        .otherwise("normal")
    ) \
    .withColumn(
        "severity_score",
        (
            (col("fatal_count")  / (col("request_count") + 1)) * 0.40 +
            (col("medium_count") / (col("request_count") + 1)) * 0.25 +
            col("error_rate") * 0.20 +
            (col("request_count") / 200).cast("double") * 0.15
        )
    ) \
    .withColumn(
        # cap score at 1.0
        "severity_score",
        when(col("severity_score") > 1.0, 1.0).otherwise(col("severity_score"))
    ) \
    .withColumn(
        "severity",
        when(col("severity_score") >= 0.7, "high")
        .when(col("severity_score") >= 0.3, "medium")
        .otherwise("low")
    )

# ══════════════════════════════════════════════════════════════
# STEP 5 — SANITY CHECK THE OUTPUT
# ══════════════════════════════════════════════════════════════

print("=== Anomaly Type Distribution ===")
windowed.groupBy("anomaly_type", "severity") \
    .count() \
    .orderBy("count", ascending=False) \
    .show()
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
 
OUTPUT_DIR = "C:/Users/VICTUS/Desktop/Engineering/Sem 8/Big Data/models"
 
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
 
    # Per-class F1
    print(f"\n  [{model_name}] Per-class F1:")
    n_classes = len(label_names)
    for i in range(n_classes):
        ev.setMetricName("f1").setMetricLabel(float(i))
        f1_i = ev.evaluate(preds)
        name = label_names[i] if i < len(label_names) else f"class_{i}"
        print(f"    {name:<20}: {f1_i:.4f}")
 
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
    .sample(fraction=0.05, seed=42) \
    .localCheckpoint()   # breaks the lineage — no Hadoop needed

print("DEBUG sample count:", windowed_sample.count())

results = run_model_training(windowed_sample)   # pass sample, not windowed