"""Shared test setup: put the project root on sys.path so tests can import
`lib.*` and `scripts.*` without a per-file sys.path.insert. (The project is
not pip-installed; this centralizes the path shim in one place.)"""
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
