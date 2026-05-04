#!/bin/bash

set -e

# =========================
# INPUT
# =========================
if [ -z "$1" ]; then
  echo "Usage: ./master-init.sh <MASTER_IP>"
  exit 1
fi

MASTER_IP=$1

SPARK_HOME=/opt/spark

echo "=== Spark Master Bootstrap ==="
echo "Master IP: $MASTER_IP"
echo ""

# =========================
# CHECK SPARK
# =========================
echo "[1/5] Checking Spark installation..."

if [ ! -d "$SPARK_HOME" ]; then
  echo "ERROR: Spark not found at $SPARK_HOME"
  exit 1
fi

echo "Spark found at $SPARK_HOME"

# =========================
# ENV SETUP
# =========================
echo "[2/5] Setting environment..."

export SPARK_HOME=/opt/spark
export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH
export SPARK_LOCAL_IP=$MASTER_IP
export SPARK_MASTER_HOST=$MASTER_IP
export PYSPARK_PYTHON=python3

# persist for future sessions
echo "export SPARK_HOME=/opt/spark" >> ~/.bashrc
echo "export PATH=\$PATH:\$SPARK_HOME/bin:\$SPARK_HOME/sbin" >> ~/.bashrc
echo "export SPARK_LOCAL_IP=$MASTER_IP" >> ~/.bashrc
echo "export SPARK_MASTER_HOST=$MASTER_IP" >> ~/.bashrc

# =========================
# CLEAN OLD MASTER (IMPORTANT)
# =========================
echo "[3/5] Stopping old master (if any)..."
$SPARK_HOME/sbin/stop-master.sh || true

# =========================
# START MASTER
# =========================
echo "[4/5] Starting Spark master..."

$SPARK_HOME/sbin/start-master.sh

sleep 3

# =========================
# VERIFY
# =========================
echo "[5/5] Checking master process..."

jps | grep Master || {
  echo "ERROR: Master failed to start"
  exit 1
}

echo ""
echo "Master started successfully"
echo "===================================="
echo "UI: http://$MASTER_IP:8080"
echo "Cluster URL: spark://$MASTER_IP:7077"
echo "===================================="