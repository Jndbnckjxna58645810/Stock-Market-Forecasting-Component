import pandas as pd
from fredapi import Fred
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.settings.config import FRED_API_KEY

from src.utils.logging_utils import get_logger

logger = get_logger("data.macro")

def get_fred(): return Fred(api_key=FRED_API_KEY)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True)
def _fetch_single_series_with_retry(fred_client, source, start_date, end_date):
    """Isolated download step covered by the retry policy."""
    return fred_client.get_series(source, start_date, end_date)

def load_macro_by_parameters(macro_features, start_date, end_date):
    dfs, fred = [], get_fred()    
    for macro in macro_features:
        name, source = macro["name"], macro["source"]
        series = _fetch_single_series_with_retry(fred, source, start_date, end_date)
        m = series.to_frame(name=name)
        
        if m.empty:
            logger.error(f"No data found for {name} ({source})")
            raise ValueError(f"No data found for {name} ({source}) {start_date} to {end_date}")
        
        dfs.append(m)
        logger.info(f"Data loaded for {name} ({source})")

    df = pd.concat(dfs, axis=1)
    df.index.name = "date"

    logger.info(f"Data loaded from fredapi")
    return df