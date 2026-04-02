import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_paths_exist():
    from lib.paths import DATA_DIR, ANALYSIS_DIR, SCRIPTS_DIR
    assert os.path.isdir(DATA_DIR)
    assert os.path.isdir(ANALYSIS_DIR)
    assert os.path.isdir(SCRIPTS_DIR)


def test_universe_weights_sum_to_one():
    from lib.universe import ASSETS, ASSET_NAMES, MARKET_WEIGHTS
    total = sum(MARKET_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"


def test_universe_consistency():
    from lib.universe import ASSETS, ASSET_NAMES, MARKET_WEIGHTS
    for asset in ASSETS:
        assert asset in ASSET_NAMES, f"{asset} missing from ASSET_NAMES"
        assert asset in MARKET_WEIGHTS, f"{asset} missing from MARKET_WEIGHTS"


def test_data_loader_returns_dataframe():
    from lib.data_loader import load_merged
    merged_path = os.path.join(
        str(Path(__file__).resolve().parent.parent), "data", "merged_data.csv"
    )
    if not os.path.exists(merged_path) or os.path.getsize(merged_path) < 100:
        import pytest
        pytest.skip("merged_data.csv not available or empty")

    df = load_merged(spx_fallback=False)
    assert len(df) > 0, "DataFrame is empty"
    assert "US10Y_yield" in df.columns
