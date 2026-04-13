# config/schemas.py
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, IntegerType

# ══════════════════════════════════════════════════════════════
# SCHEMA 1 — Server/Application Logs (Hadoop, HDFS, OpenStack)
# Used by: colleague's dataset
# Example line:
# 2015-10-17 15:37:28,981 INFO ContainerLauncher - container launch failed
# ══════════════════════════════════════════════════════════════
SERVER_LOG_SCHEMA = StructType([
    StructField("timestamp",      TimestampType(), False),
    StructField("log_level",      StringType(),    False),  # INFO, WARN, ERROR
    StructField("system",         StringType(),    False),  # hdfs, yarn, openstack
    StructField("component",      StringType(),    True),   # ContainerLauncher, TaskAttemptImpl
    StructField("message",        StringType(),    False),
    StructField("exception_type", StringType(),    True),   # for anomaly detection
    StructField("stack_trace",    StringType(),    True),
    StructField("container_id",   StringType(),    True),   # extracted from message
    StructField("attempt_id",     StringType(),    True),
])

# ══════════════════════════════════════════════════════════════
# SCHEMA 2 — Web Access Logs (Apache Common / Combined Format)
# Used by: your datasets (access_log.txt + web_server_access_log.txt)
# Example line (Common):
# 10.223.157.186 - - [15/Jul/2009:15:50:35 -0700] "GET /assets/js/lowpro.js HTTP/1.1" 200 10469
# Example line (Combined):
# 54.36.149.41 - - [22/Jan/2019:03:56:14 +0330] "GET /filter/27 HTTP/1.1" 200 30577 "-" "AhrefsBot/6.1" "-"
# ══════════════════════════════════════════════════════════════
WEB_ACCESS_LOG_SCHEMA = StructType([
    StructField("timestamp",    TimestampType(), False),
    StructField("ip",           StringType(),    False),
    StructField("method",       StringType(),    False),  # GET, POST, PUT, DELETE
    StructField("endpoint",     StringType(),    False),
    StructField("status_code",  IntegerType(),   False),
    StructField("bytes",        IntegerType(),   True),
    StructField("user_agent",   StringType(),    True),   # null for Common format logs
    StructField("is_bot",       IntegerType(),   False),  # 0 or 1
    StructField("is_error",     IntegerType(),   False),  # 0 or 1
    StructField("status_class", StringType(),    False),  # 2xx, 4xx, 5xx
    StructField("source",       StringType(),    False),  # which log file it came from
])


