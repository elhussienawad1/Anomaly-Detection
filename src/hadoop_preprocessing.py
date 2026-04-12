# src/ingestion/hadoop_logs.py
import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, col, when

def parse_hadoop_logs(spark, input_path):
    raw_logs = spark.read.text(input_path)

    log_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) (INFO|WARN|ERROR|FATAL) \[([^\]]+)\] ([^:]+): (.+)'
    
    parsed = raw_logs.select(
        regexp_extract('value', log_pattern, 1).alias('timestamp'),
        regexp_extract('value', log_pattern, 2).alias('log_level'),
        regexp_extract('value', log_pattern, 3).alias('thread'),
        regexp_extract('value', log_pattern, 4).alias('component'),
        regexp_extract('value', log_pattern, 5).alias('message'),
        col('value').alias('raw_log')
    )
    

    
    return parsed