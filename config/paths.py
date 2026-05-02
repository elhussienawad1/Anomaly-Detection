# config/paths.py
import os

# ── Each teammate sets their own paths here ────────────────
# Do NOT commit personal paths — add paths.py to .gitignore
# Copy paths_template.py and rename to paths.py locally

DATASET_1 = os.environ.get(
    "DATASET_1",
    "config/datasets/access_log.txt"          # default (Linux/Abdallah)
)

DATASET_2 = os.environ.get(
    "DATASET_2",
    "config/datasets/web_server_access_log.txt"
)

OUTPUT_DIR       = os.path.join(os.path.dirname(__file__), '..', 'output')
MODELS_DIR       = os.path.join(OUTPUT_DIR, 'models')
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, 'visualizations')

# Linux 
#run in terminal:
# export DATASET_1="/media/sf_project/Anomaly-Detection/config/datasets/access_log.txt"
# python main.py

# Windows  in PowerShell
# $env:DATASET_1="C:\Users\Noor Tantawy\Desktop\BigData\project\access_log.txt"
# $env:DATASET_2="C:\Users\Noor Tantawy\Desktop\BigData\project\access.log"
# python main.py
