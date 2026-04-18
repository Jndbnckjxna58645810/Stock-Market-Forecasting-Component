import pandas as pd

def normalize_df(df, ticker=None):
    if isinstance(df.columns, pd.MultiIndex) and ticker is not None:
        df = df.xs(ticker, axis=1, level=1)

    df.columns = [col.lower() for col in df.columns]

    if df.index.name: df.index.name = df.index.name.lower()

    return df

def clean_df(df):
    df.dropna(inplace=True)
    return df

def handle_missing(df, method="drop"):
    if method == "drop": return df.dropna()
    elif method == "ffill": return df.ffill()
    elif method == "bfill": return df.bfill()
    else: return df