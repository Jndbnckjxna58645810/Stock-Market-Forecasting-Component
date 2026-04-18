from src.features import *
from src.data_loader import *

ticker, start_date, end_date, interval = "AAPL", "2010-01-01", "2020-01-01", "1wk"
df = load_yfinance(ticker, start_date, end_date, interval=interval, save=True)
df = apply_features(df)
save_processed_csv(df, ticker, start_date, end_date, interval=interval)
