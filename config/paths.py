# config/paths.py
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DATASET_1          = os.environ["DATASET_1"]
DATASET_2          = os.environ["DATASET_2"]
MODELS_DIR         = os.environ.get("MODELS_DIR",         os.path.join(os.path.dirname(__file__), '..', 'output', 'models'))
VISUALIZATIONS_DIR = os.environ.get("VISUALIZATIONS_DIR", os.path.join(os.path.dirname(__file__), '..', 'output', 'visualizations'))
# Linux 
#run in terminal:
# export DATASET_1="/media/sf_project/Anomaly-Detection/config/datasets/access_log.txt"
# python main.py

# Windows  in PowerShell
# $env:DATASET_1="C:\Users\Noor Tantawy\Desktop\BigData\project\access_log.txt"
# $env:DATASET_2="C:\Users\Noor Tantawy\Desktop\BigData\project\access.log"
# python main.py
