import os
import pandas as pd
from lib.paths import DATA_DIR


def load_merged(spx_fallback=True) -> pd.DataFrame:
    """Load merged_data.csv with optional SPX/SPY fallback."""
    df = pd.read_csv(os.path.join(DATA_DIR, "merged_data.csv"), index_col=0, parse_dates=True)

    if spx_fallback and ("SPX" not in df.columns or df["SPX"].isna().all()):
        try:
            import yfinance as yf
            spx = yf.download("SPY", start="2020-01-01", progress=False, auto_adjust=True)["Close"]
            if isinstance(spx, pd.DataFrame):
                spx = spx.iloc[:, 0]
            spx.index = pd.to_datetime(spx.index)
            df["SPX"] = spx.reindex(df.index, method="ffill")
        except Exception:
            pass

    return df
