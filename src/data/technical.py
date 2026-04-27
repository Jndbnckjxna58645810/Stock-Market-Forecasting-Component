import yfinance as yf
import pandas as pd

from src.config.config import RAW_DATA_DIR
from src.utils.csv_utils import load_csv, save_csv
from src.pipeline.preprocessing import normalize_df

def load_technical(ticker, start_date, end_date, interval="1d", offset=0, save=False, path=None, force_download=False):
    default_path = RAW_DATA_DIR / f"{ticker}_{start_date}_{end_date}_{interval}.csv"

    df = normalize_df(load_csv(default_path) if (default_path.exists() and not force_download) else yf.download(
            ticker, start=pd.to_datetime(start_date) - pd.DateOffset(days=offset), end=pd.to_datetime(end_date) + pd.DateOffset(days=1), interval=interval))

    if df.empty: raise ValueError(f"No data for {ticker}")

    if save: save_csv(df, default_path if path == None else path)
    return df