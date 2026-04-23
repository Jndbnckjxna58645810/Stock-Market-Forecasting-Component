import yfinance as yf

from src.config.config import RAW_DATA_DIR
from src.data.loader import load_csv, save_csv
from src.preprocessing.preprocessing import normalize_df

def load_technical(ticker, start_date, end_date, interval="1d", save=False, path=None, force_download=False):
    default_path = RAW_DATA_DIR / f"{ticker}_{start_date}_{end_date}_{interval}.csv"

    df = normalize_df(load_csv(default_path) if (default_path.exists() and not force_download) else yf.download(
            ticker, start=start_date, end=end_date, interval=interval))

    if df.empty: raise ValueError(f"No data for {ticker}")

    if save: save_csv(df, default_path if path == None else path)
    return df