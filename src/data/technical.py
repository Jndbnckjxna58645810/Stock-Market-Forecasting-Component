import yfinance as yf
import pandas as pd

from src.settings.config import RAW_DATA_DIR
from src.utils.io_utils import load_csv, save_csv
from src.utils.logging_utils import get_logger

from src.pipeline.preprocessing import normalize_df_by_parameters

logger = get_logger("data.technical")

def load_technical_by_parameters(ticker, start_date, end_date, interval="1d", save_technical=False, force_download=False, path=None):
    default_path = RAW_DATA_DIR / f"{ticker}_{start_date}_{end_date}_{interval}.csv"

    if default_path.exists() and not force_download:
        df = load_csv(default_path)

        logger.info(f"Data loaded from: {default_path}")
    else:
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval)

        logger.info(f"Data loaded from yfinance")

    df = normalize_df_by_parameters(df, ticker)

    if df.empty:
        logger.error(f"No data found for {ticker}")

        raise ValueError(f"No data found for {ticker}")

    if save_technical: save_csv(df, default_path if path == None else path)
    return df