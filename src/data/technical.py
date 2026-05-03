import yfinance as yf
import pandas as pd

from src.settings.config import RAW_DATA_DIR
from src.utils.io_utils import load_csv, save_csv
from src.utils.config_utils import ensure
from src.utils.logging_utils import get_logger

from src.pipeline.preprocessing import normalize_df_by_parameters, get_max_lookback_by_parameters

from src.config.train_config import TrainConfig
from src.config.predict_config import PredictConfig
from src.config.model_metadata import ModelMetadata
from src.config.evaluate_config import EvaluateConfig

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

def load_technical_dataset(run: TrainConfig):
    run = ensure(run, TrainConfig)

    logger.info(f"Loading training data for {run.ticker}" +
                f" | Period: {run.start_date} to {run.end_date}" +
                f" | Interval: {run.interval}")

    return load_technical_by_parameters(
        run.ticker, run.start_date, run.end_date, run.interval,
        save_technical=run.data_config["technical"]["save"],
        force_download=run.data_config["technical"]["force_download"],
        path=run.data_config["technical"]["path"])

def load_technical_input(predict_config: PredictConfig):
    predict_config = ensure(predict_config, PredictConfig)
    model_metadata = ModelMetadata.from_name(predict_config.model_path)

    offset_start = get_max_lookback_by_parameters(model_metadata.features) * 2 + 1
    if model_metadata.hyperparameters.get("seq_len"):
        offset_start += 2 * model_metadata.hyperparameters.get("seq_len")

    logger.info(f"Loading input data for {model_metadata.ticker}" +
                f" | Period: {predict_config.start_date} to {predict_config.end_date}" +
                f" | Offset days: {offset_start}" +
                f" | Interval: {model_metadata.interval}")

    return load_technical_by_parameters(
        model_metadata.ticker,
        pd.to_datetime(predict_config.start_date) - pd.DateOffset(days=offset_start),
        pd.to_datetime(predict_config.end_date) + pd.DateOffset(days=1),
        interval=model_metadata.interval,
        save_technical=False, force_download=True, path=None)

def load_technical_evaluation_dataset(evaluate_config: EvaluateConfig, model_metadata: ModelMetadata):
    evaluate_config = ensure(evaluate_config, EvaluateConfig)

    offset_start = get_max_lookback_by_parameters(model_metadata.features) * 2 + 1

    logger.info(f"Loading evaluation data for {model_metadata.ticker}" +
                f" | Period: {evaluate_config.start_date} to {evaluate_config.end_date}" +
                f" | Offset days: {offset_start}" +
                f" | Interval: {model_metadata.interval}")

    return load_technical_by_parameters(
        evaluate_config.ticker,
        pd.to_datetime(evaluate_config.start_date) - pd.DateOffset(days=offset_start),
        pd.to_datetime(evaluate_config.end_date) + pd.DateOffset(days=1),
        evaluate_config.interval,
        save_technical=False, force_download=True, path=None)