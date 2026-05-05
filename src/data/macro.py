import math
import pandas as pd
from fredapi import Fred

from src.settings.config import FRED_API_KEY, RAW_DATA_DIR

from src.utils.io_utils import load_csv, save_csv
from src.utils.config_utils import make_signature
from src.utils.logging_utils import get_logger

logger = get_logger("data.macro")

def get_fred(): return Fred(api_key=FRED_API_KEY)

def load_macro_by_parameters(macro_features, start_date, end_date, force_download=False, save_macro=False, path=None):
    default_path = RAW_DATA_DIR / f"macro_{start_date}_{end_date}_{make_signature(macro_features)}.csv"
    if default_path.exists() and not force_download:
        df = load_csv(default_path)

        logger.info(f"Data loaded from: {default_path}")
    else:
        dfs, fred = [], get_fred()
        
        for macro in macro_features:
            name, source = macro["name"], macro["source"]
            m = fred.get_series(source, start_date, end_date).to_frame(name=name)
            if m.empty:
                logger.error(f"No data found for {name} ({source})")

                raise ValueError(f"No data found for {name} ({source}) {start_date} to {end_date}")
            dfs.append(m)

            logger.info(f"Data loaded for {name} ({source})")

        df = pd.concat(dfs, axis=1)
        df.index.name = "date"

        logger.info(f"Data loaded from fredapi")
    
    if save_macro:
        save_csv(df, default_path if path == None else path)

        logger.info(f"Data saved to {default_path if path == None else path}")
    return df