#!/bin/bash

set -e

# =========================
# INPUT
# =========================
MASTER_IP=$1

if [ -z "$MASTER_IP" ]; then
  echo "Usage: ./start_worker.sh <MASTER_IP>"
  exit 1
fi

SPARK_HOME=~/spark
SPARK_VERSION="3.5.8"
JAVA_REQUIRED="17"

echo "=== Spark Worker Bootstrap ==="
echo "Master: $MASTER_IP"
echo ""

# =========================
# CHECK JAVA VERSION
# =========================
echo "[1/6] Checking Java version..."
JAVA_VER=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}' | cut -d'.' -f1)

if [[ "$JAVA_VER" != "$JAVA_REQUIRED" ]]; then
  echo "ERROR: Java $JAVA_REQUIRED required, found $JAVA_VER"
  exit 1
fi

echo "Java OK ($JAVA_VER)"

# =========================
# CHECK SPARK VERSION
# =========================
echo "[2/6] Checking Spark version..."
if [ ! -d "$SPARK_HOME" ]; then
  echo "ERROR: Spark not found at $SPARK_HOME"
  exit 1
fi

SPARK_VER=$($SPARK_HOME/bin/spark-submit --version 2>&1 | grep "version" | head -n 1)

echo "Spark OK: $SPARK_VER"

# =========================
# SET ENV
# =========================
echo "[3/6] Setting environment..."

export SPARK_HOME=~/spark
export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH
export SPARK_LOCAL_IP=$(hostname -I | awk '{print $1}')
export PYSPARK_PYTHON=python3

# =========================
# TEST MASTER CONNECTIVITY
# =========================
echo "[4/6] Pinging master..."
ping -c 2 $MASTER_IP > /dev/null

if [ $? -ne 0 ]; then
  echo "ERROR: Cannot reach master $MASTER_IP"
  exit 1
fi

echo "Master reachable"

# =========================
# COPY DATASET (adjust path if needed)
# =========================
echo "[5/6] Ensuring dataset directory..."

sudo mkdir -p /data/logs

if [ -f "$HOME/Desktop/Anomaly-Detection/config/datasets/access.log" ]; then
  sudo cp $HOME/Desktop/Anomaly-Detection/config/datasets/access.log /data/logs/
fi

if [ -f "$HOME/Desktop/Anomaly-Detection/config/datasets/access_log.txt" ]; then
  sudo cp $HOME/Desktop/Anomaly-Detection/config/datasets/access_log.txt /data/logs/
fi

echo "Dataset ready at /data/logs"

# =========================
# SET WORKER RESOURCES
# =========================
echo "[6/6] Configuring worker resources..."

export SPARK_WORKER_CORES=4
export SPARK_WORKER_MEMORY=4g

# =========================
# START WORKER
# =========================
echo "Starting Spark worker..."

$SPARK_HOME/sbin/start-worker.sh spark://$MASTER_IP:7077

echo ""
echo "Worker started successfully"
echo "Check UI: http://$MASTER_IP:8080"