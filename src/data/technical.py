import yfinance as yf
import pandas as pd

from src.settings.config import RAW_DATA_DIR
from src.utils.csv_utils import load_csv, save_csv
from src.utils.config_utils import ensure
from src.pipeline.preprocessing import normalize_df_by_parameters, get_max_lookback_by_parameters

from src.config.run_config import RunConfig
from src.config.predict_config import PredictConfig
from src.config.model_metadata import ModelMetadata

def load_technical_by_parameters(ticker, start_date, end_date, interval="1d", save_technical=False, force_download=False, path=None):
    default_path = RAW_DATA_DIR / f"{ticker}_{start_date}_{end_date}_{interval}.csv"
    df = normalize_df_by_parameters(load_csv(default_path) if (default_path.exists() and not force_download)
                      else yf.download(ticker, start=start_date, end=end_date, interval=interval), ticker)
    if df.empty: raise ValueError(f"No data for {ticker}")

    if save_technical: save_csv(df, default_path if path == None else path)
    return df

def load_technical_dataset(run: RunConfig):
    run = ensure(run, RunConfig)
    return load_technical_by_parameters(
        run.ticker, run.start_date, run.end_date, run.interval,
        save_technical=run.data_config["technical"]["save"],
        force_download=run.data_config["technical"]["force_download"],
        path=run.data_config["technical"]["path"])

def load_technical_input(predict_config: PredictConfig):
    predict_config = ensure(predict_config, PredictConfig)
    model_metadata = ModelMetadata.from_name(predict_config.model_path)
    return load_technical_by_parameters(
        model_metadata.ticker,
        pd.to_datetime(predict_config.input_date) - pd.DateOffset(
            days=get_max_lookback_by_parameters(model_metadata.features) * 2 + 1),
        pd.to_datetime(predict_config.input_date) + pd.DateOffset(days=2),
        interval=model_metadata.interval,
        save_technical=False, force_download=True, path=None)