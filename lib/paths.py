import os
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "analysis")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)
