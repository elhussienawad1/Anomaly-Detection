import os
from pyspark.sql import SparkSession

def create_spark(app_name="LogAnomalyDetection"):
    return (
        SparkSession.builder
        .appName(app_name)
        .master("spark://192.168.1.90:7077")
        
        .config("spark.driver.memory", "4g")

        .config("spark.executor.memory", "3g")
        .config("spark.executor.cores", "3")
        .config("spark.executor.instances", "4")


        .config("spark.sql.shuffle.partitions", "48")
        .config("spark.default.parallelism", "24")

        .getOrCreate()
    )