from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    regexp_extract, col, to_timestamp, when
)

spark = SparkSession.builder \
    .appName("LogAnomalyDetection") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ── Load raw logs ──────────────────────────────────────────
df_raw = spark.read.text("config/datasets/access_log.txt")
df_raw.show(5, truncate=False)

# ── showed logs format ──────────────────────
# Example line:
# 192.168.1.1 - - [10/Oct/2024:13:55:36 -0700] "GET /index.html HTTP/1.1" 200 2326

# 10.223.157.186 - - [15/Jul/2009:14:58:59 -0700] "GET / HTTP/1.1" 403 202                     
# 10.223.157.186 - - [15/Jul/2009:14:58:59 -0700] "GET /favicon.ico HTTP/1.1" 404 209          
# 10.223.157.186 - - [15/Jul/2009:15:50:35 -0700] "GET / HTTP/1.1" 200 9157                    
# 10.223.157.186 - - [15/Jul/2009:15:50:35 -0700] "GET /assets/js/lowpro.js HTTP/1.1" 200 10469
# 10.223.157.186 - - [15/Jul/2009:15:50:35 -0700] "GET /assets/css/reset.css HTTP/1.1" 200 1014

LOG_PATTERN = (
    r'^(\S+)'                        # IP address
    r'\s+\S+'                        # ident (usually -)
    r'\s+\S+'                        # another indent
    r'\s+\[([^\]]+)\]'              # timestamp
    r'\s+"(\S+)'                     # HTTP method
    r'\s+(\S+)'                      # endpoint
    r'\s+\S+"'                       # protocol
    r'\s+(\d{3})'                    # status code
    r'\s+(\S+)'                      # bytes
)

# ── Parse into structured columns ─────────────────────────
df_parsed = df_raw.select(
    regexp_extract("value", LOG_PATTERN, 1).alias("ip"),
    regexp_extract("value", LOG_PATTERN, 2).alias("timestamp_raw"),
    regexp_extract("value", LOG_PATTERN, 3).alias("method"),
    regexp_extract("value", LOG_PATTERN, 4).alias("endpoint"),
    regexp_extract("value", LOG_PATTERN, 5).cast("int").alias("status_code"),
    regexp_extract("value", LOG_PATTERN, 6).alias("bytes_raw"),
)

# ── Clean & cast ───────────────────────────────────────────
df_clean = df_parsed \
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
        when(col("status_code") < 300, "2xx")
        .when(col("status_code") < 500, "4xx")
        .otherwise("5xx")
    ) \
    .drop("timestamp_raw", "bytes_raw") \
    .filter(col("ip") != "")   # drop malformed lines

df_clean.printSchema()
df_clean.show(10, truncate=False)