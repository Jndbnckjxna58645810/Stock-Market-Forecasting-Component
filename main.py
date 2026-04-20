from src.data.sources.technical import load_technical
from src.data.sources.macro import load_macro
from src.data.loader import save_processed_csv
from src.features.pipeline import apply_features
from src.preprocessing.preprocessing import merge_df, handle_missing, normalize_df

ticker, start_date, end_date, interval = "AAPL", "2015-01-01", "2020-01-01", "1d"

FEATURES = [
    ("sma", {"window": 20}), ("sma", {"window": 50}),
    ("ema", {"window": 20}), ("ema", {"window": 50}),
    ("momentum", {"window": 5}), ("volatility", {"window": 10}),
    ("rsi", {"window": 14}), ("macd", {}),
    ("lag", {"n": 1}), ("lag", {"n": 2}), ("return_lag", {}),
]

MACRO_FEATURES = {"interest_rate": "FEDFUNDS", "unemployment": "UNRATE", "inflation": "CPIAUCSL",}

df = load_technical(ticker, start_date, end_date, interval=interval, save=True, force_download=True)
macro = load_macro(MACRO_FEATURES, start_date, end_date, save=False)
df = normalize_df(df, ticker=ticker)
df = apply_features(df, FEATURES)
print(df)
df = merge_df(df, macro)
df = handle_missing(df, "ffill")
print(df)
save_processed_csv(df, ticker, start_date, end_date, interval=interval)
