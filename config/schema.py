# config/schemas.py
from pyspark.sql.types import StructType, StructField, StringType, TimestampType

UNIFIED_LOG_SCHEMA = StructType([
    StructField("timestamp", TimestampType(), False),
    StructField("log_level", StringType(), False),      # INFO, WARN, ERROR
    StructField("system", StringType(), False),          # hdfs, awsctd, openstack
    StructField("component", StringType(), True),        # e.g., "ContainerLauncher", "TaskAttemptImpl"
    StructField("message", StringType(), False),
    StructField("exception_type", StringType(), True),   # For anomaly detection
    StructField("stack_trace", StringType(), True),
    StructField("container_id", StringType(), True),     # Extract from message
    StructField("attempt_id", StringType(), True),
])