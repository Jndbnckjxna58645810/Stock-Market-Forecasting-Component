import yfinance as yf
import pandas as pd

from src.settings.config import RAW_DATA_DIR
from src.utils.csv_utils import load_csv, save_csv
from src.utils.config_utils import ensure_run_config
from src.pipeline.preprocessing import normalize_df, get_max_lookback

from src.config.run_config import RunConfig

def load_technical(run: RunConfig, input_date=None):
    run = ensure_run_config(run)

    t = run.ticker
    s = pd.to_datetime(input_date) - pd.DateOffset(days=5) if input_date != None else run.start_date
    e = pd.to_datetime(input_date) + pd.DateOffset(days=1) if input_date != None else run.end_date
    i = run.interval

    default_path = RAW_DATA_DIR / f"{t}_{s}_{e}_{i}.csv"

    df = normalize_df(load_csv(default_path) if (default_path.exists()
                                                 and not run.data_config["technical"]["force_download"])
                                                 else yf.download(t,
                                                                  start=pd.to_datetime(s) - pd.DateOffset(days=get_max_lookback(run) * 2),
                                                                  end=pd.to_datetime(e) + pd.DateOffset(days=1), interval=i), run)

    if df.empty: raise ValueError(f"No data for {t}")

    if run.data_config["technical"]["save"] and input_date == None:
        path = run.data_config["technical"]["path"]
        save_csv(df, default_path if path == None else path)
    
    return df