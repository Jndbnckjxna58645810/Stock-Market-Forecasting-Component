import pandas as pd

from src.utils.config_utils import load_run_config, load_model_config

def normalize_df(df, run):
    ticker = load_run_config(run)["ticker"]
    if isinstance(df.columns, pd.MultiIndex):
        if ticker: df = df.xs(ticker, axis=1, level=1)
        else: df.columns = df.columns.get_level_values(0)

    df.columns = [str(col).lower() for col in df.columns]

    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    return df

def handle_missing(df, method="drop"):
    if method == "drop": return df.dropna()
    elif method == "ffill": return df.ffill()
    elif method == "bfill": return df.bfill()
    else: return df

def handle_macro(df):
    df.index = pd.to_datetime(df.index)
    return df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')).ffill()

def merge_df(df_t, df_m): return handle_missing(df_t.join(handle_macro(df_m)), "ffill")

def get_max_lookback(run):
    features = load_model_config(load_run_config(run)["model_config"])["features"]
    max_lookback = 0

    for f in features:
        params = f.get("params", {})
        for v in params.values():
            if isinstance(v, int):
                max_lookback = max(max_lookback, v)

    return max_lookback