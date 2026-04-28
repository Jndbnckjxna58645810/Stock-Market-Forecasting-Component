import math
import pandas as pd
from fredapi import Fred

from src.config.config import FRED_API_KEY, RAW_DATA_DIR
from src.pipeline.preprocessing import get_max_lookback
from src.utils.csv_utils import load_csv, save_csv
from src.utils.config_utils import load_model_config, load_run_config
from src.utils.vesrioning_utils import make_signature

def get_fred(): return Fred(api_key=FRED_API_KEY)

def load_macro(run, input_date=None):
    run_config = load_run_config(run)

    macro = load_model_config(run_config["model_config"])["macro_features"]
    s = pd.to_datetime(input_date) - pd.DateOffset(days=5) if input_date != None else run_config["start_date"]
    e = pd.to_datetime(input_date) + pd.DateOffset(days=1) if input_date != None else run_config["end_date"]

    default_path = RAW_DATA_DIR / f"macro_{s}_{e}_{make_signature(macro)}.csv"

    if default_path.exists() and not run_config["data"]["macro"]["force_download"]: df = load_csv(default_path)
    else:
        dfs, fred = [], get_fred()
        ms = pd.to_datetime(s) - pd.DateOffset(days=math.ceil(get_max_lookback(run) / 31) * 31)

        for key, value in macro.items():
            m = fred.get_series(value, ms, e).to_frame(name=key)
            if m.empty: raise ValueError(f"No data for {key} ({value})")
            dfs.append(m)

        df = pd.concat(dfs, axis=1)
        df.index.name = "date"

    if run_config["data"]["macro"]["save"] and input_date == None:
        path = run_config["data"]["macro"]["path"]
        save_csv(df, default_path if path == None else path)
    
    return df
