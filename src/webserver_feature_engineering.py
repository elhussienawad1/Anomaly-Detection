import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


from pyspark.sql.functions import (
    window, count, sum, avg, countDistinct,
    hour, dayofweek, col, when
)

def run_feature_engineering(df_all):
    """
    Takes the cleaned df_all from preprocessing.
    Returns (df_enriched, windowed) — enriched rows and windowed aggregates.
    """
    df_enriched = df_all \
        .withColumn(
            "error_severity",
            when(col("status_code").isin(500, 502, 503, 504), "fatal")  # added 503
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
    df_enriched.count()

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
        "error_rate", col("error_count") / col("request_count")
    ).withColumn(
        "anomaly_type",
        when(col("fatal_count") > 0,        "server_error")
        .when(col("request_count") > 200,   "traffic_spike")
        .when(col("error_rate") > 0.3,      "high_error_rate")
        .when(col("unique_endpoints") > 30, "scanning")
        .when(col("bot_requests") > 0,      "bot_traffic")
        .otherwise("normal")
    ).withColumn(
        "severity_score",
        when(
            (col("fatal_count") / (col("request_count") + 1)) * 0.40 +
            (col("medium_count") / (col("request_count") + 1)) * 0.25 +
            col("error_rate") * 0.20 +
            (col("request_count") / 200).cast("double") * 0.15 > 1.0,
            1.0
        ).otherwise(
            (col("fatal_count") / (col("request_count") + 1)) * 0.40 +
            (col("medium_count") / (col("request_count") + 1)) * 0.25 +
            col("error_rate") * 0.20 +
            (col("request_count") / 200).cast("double") * 0.15
        )
    ).withColumn(
        "severity",
        when(col("severity_score") >= 0.7, "high")
        .when(col("severity_score") >= 0.3, "medium")
        .otherwise("low")
    )

    print("=== Anomaly Type Distribution ===")
    windowed.groupBy("anomaly_type", "severity") \
        .count().orderBy("count", ascending=False).show()

    return df_enriched, windowed
