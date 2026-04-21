# Anomaly-Detection

project structure

Anomaly-Detection/
    config/
        schema
    Data/
    archive.zip // da awel dataset fel proposal ghaleban msh hanehtag gherha
    src/
    main.ipynb

# create virtual environment 
# kolo on WSL 
python3 -m venv SparkEnv

source SparkEnv/bin/activate

pip install install pyspark 

# lazem teb2a menazel java w hatet el directory mazboot fel bashrc

unzip archive.zip 

# run ipynb momken men anaconda aw vs code lazem tehotto kernel SparkEnv lama yes2alak 3aleha

---------------------------------------------------------------------------
# Anomaly Detection in Large-Scale System Logs
**Team 19 — CMPS451 Spring 2026**
Abdallah ElMahdy | Noor Tantawy | George Ayman | Elhussien Awad

## Project Overview
Scalable anomaly detection platform built on Apache Spark that analyzes
14.8 million web server log entries to detect 5 types of anomalies
with up to 99.92% accuracy.

## Requirements
- Python 3.8+
- Java JDK 11 or 17
- Apache Spark 3.x or 4.x
- PySpark

Install Python dependencies:
 pip install -r requirements.txt

## Dataset Setup
Download datasets from Kaggle and place in config/datasets/:
- access_log.txt → https://www.kaggle.com/code/adepvenugopal/logs-dataset/input?select=access_log.txt
- web_server_access_log.txt → https://www.kaggle.com/datasets/eliasdabbas/web-server-access-logs

## Running the Project
From the project root:
python main.py

## Output
- output/visualizations/ — 20 PNG plots

## Project Structure
config/          — schemas and dataset paths
src/webserver_preprocessing/   — log parsing
src/webserver_feature_engineering/    — feature engineering
src/webserver_modeling/      — ML models
src/evaluawebserver_evaluation/  — metrics
src/webserver_visualization/ — plots
main.py          — entry point