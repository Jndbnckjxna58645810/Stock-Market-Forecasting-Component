import yfinance as yf

from src.utils.logging_utils import get_logger

from src.pipeline.preprocessing import normalize_df_by_parameters

logger = get_logger("data.technical")

def load_technical_by_parameters(ticker, start_date, end_date, interval="1d"):
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    df = normalize_df_by_parameters(df, ticker)

    logger.info(f"Data loaded from yfinance")

    if df.empty:
        logger.error(f"No data found for {ticker}")
        raise ValueError(f"No data found for {ticker}")
    
    return df