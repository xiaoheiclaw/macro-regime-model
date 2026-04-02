import os
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "analysis")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# TimesFM external tool paths
TIMESFM_PYTHON = "/Users/xiaohei/.openclaw/workspace-quant/timesfm-env/bin/python"
TIMESFM_SCRIPT = "/Users/xiaohei/.openclaw/workspace-quant/scripts/timesfm_predict.py"
