import pandas as pd
import math

from src.utils.logging_utils import get_logger

logger = get_logger("pipeline.preprocessing")

def normalize_df_by_parameters(df, ticker):
    if isinstance(df.columns, pd.MultiIndex):
        if ticker:
            df = df.xs(ticker, axis=1, level=1)

            logger.info(f"MultiIndex for {ticker} normalized for technical data")
        else: df.columns = df.columns.get_level_values(0)

    df.columns = [str(col).lower() for col in df.columns]

    logger.info(f"Columns in technical data lowercased")

    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    logger.info(f"Technical data index renamed to 'date'")

    return df

def handle_missing(df, method="drop"):
    logger.info(f"Missing data handled | method='{method}'")

    if method == "drop": return df.dropna()
    elif method == "ffill": return df.ffill()
    elif method == "bfill": return df.bfill()
    else: return df

def merge_and_align_datasets(df_t, df_m):
    if df_m.empty:
        logger.warning("Macro data empty | Returning technical data only")
        return df_t
    
    combined = df_t.join(df_m, how='outer')

    macro_cols = df_m.columns
    combined[macro_cols] = combined[macro_cols].ffill()

    final_df = combined.reindex(df_t.index)

    print(final_df)
    
    logger.info("Technical and macroeconomical data merged and aligned to interval")
    return final_df

def convert_bars_to_days(bars, interval):
    mapping = {"1h": 1/24, "4h": 4/24, "1d": 1, "1wk": 7, "1mo": 31}
    days_per_bar = mapping.get(interval, 1)
    return math.ceil(bars * days_per_bar * 1.4) + 31 * 2

def get_max_lookback_by_parameters(features):
    max_lookback = 0

    for f in features:
        params = f.get("params", {})
        for v in params.values():
            if isinstance(v, int):
                max_lookback = max(max_lookback, v)

    return max_lookback

def get_max_horizon_by_parameters(target):
    max_horizon = target["params"].get("horizon", 1)
    if target["params"].get("horizons", 0):
        max_horizon = max(target["params"].get("horizons", 1))
    return max_horizon