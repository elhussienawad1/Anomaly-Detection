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

df_raw_1 = spark.read.text("/home/abdallah/Anomaly-Detection/Data/access_log.txt")

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

df_raw_2 = spark.read.text("/home/abdallah/Anomaly-Detection/Data/access.log")

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
