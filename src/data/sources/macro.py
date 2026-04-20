import hashlib
import pandas as pd
from fredapi import Fred

from src.config import FRED_API_KEY, RAW_DATA_DIR
from src.data.loader import load_csv, save_csv, get_next_version, get_versions

def get_fred(): return Fred(api_key=FRED_API_KEY)

def macro_signature(macro):
    signature = "_".join(sorted(macro.values()))
    return hashlib.md5(signature.encode()).hexdigest()[:8]

def load_macro(macro, start_date, end_date, save=False, path=None, force_download=False):
    default_path = RAW_DATA_DIR / f"macro_{start_date}_{end_date}_{macro_signature(macro)}.csv"

    if default_path.exists() and not force_download: df = load_csv(default_path)
    else:
        dfs, fred = [], get_fred()
        macro_start_date = pd.to_datetime(start_date) - pd.DateOffset(months=1)

        for key, value in macro.items():
            m = fred.get_series(value, macro_start_date, end_date).to_frame(name=key)
            if m.empty: raise ValueError(f"No data for {key} ({value})")
            dfs.append(m)

        df = pd.concat(dfs, axis=1)
        df.index.name = "date"

    if save: save_csv(df, default_path if path == None else path)
    return df