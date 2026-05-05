import pandas as pd
import math

from src.data.technical import load_technical_by_parameters
from src.data.macro import load_macro_by_parameters

from src.utils.data_manager import save_processed_data, load_processed_data
from src.utils.config_utils import ensure
from src.utils.logging_utils import get_logger

from src.pipeline.apply_features import apply_features_by_parameters
from src.pipeline.preprocessing import handle_missing, get_max_lookback_by_parameters, get_max_horizon_by_parameters, merge_and_align_datasets, convert_bars_to_days

from src.config.train_config import TrainConfig
from src.config.predict_config import PredictConfig
from src.config.evaluate_config import EvaluateConfig
from src.config.model_metadata import ModelMetadata
from src.config.model_config import ModelConfig

logger = get_logger("pipeline.build_dataset")

def load_dataset(run: TrainConfig, model_config=None):
    run = ensure(run, TrainConfig)
    if model_config == None:
        if run.model_config_path == None: raise ValueError("Training requires model_config")
        model_config = ModelConfig.from_name(run.model_config_path)

    if not run.data_config["processed"]["force_download"]:
        try: return load_processed_data(run, model_config)
        except FileNotFoundError: return pd.DataFrame()
    return pd.DataFrame()

def build_dataset_by_parameters(ticker, start_date, end_date, interval,
                                features, macro_features, target,
                                hyperparameters, data_config=None):
    lookback_days = convert_bars_to_days((hyperparameters.get("seq_len", 0) * 2
                                          + get_max_lookback_by_parameters(features)), interval)
    lookforward_days = convert_bars_to_days(get_max_horizon_by_parameters(target), interval)
    
    effective_start = pd.to_datetime(start_date) - pd.DateOffset(days=lookback_days)
    effective_end = pd.to_datetime(end_date) + pd.DateOffset(days=lookforward_days)

    technical = load_technical_by_parameters(
        ticker, effective_start, effective_end, interval,
        save_technical=False, force_download=True, path=None)

    macro = load_macro_by_parameters(
        macro_features, effective_start, effective_end,
        save_macro=False, force_download=True, path=None)

    df = merge_and_align_datasets(technical, macro)
    df = apply_features_by_parameters(df, features)
    df = handle_missing(df, method="ffill")

    print(df)

    df = df.dropna()

    print(df)

    logger.info(f"Processed dataset built for {ticker}" +
                f" | Period: {start_date} to {end_date}" +
                f" | Offset period: {effective_start} to {effective_end}" +
                f" | Interval: {interval}")
    
    last_date = pd.to_datetime(df.index[-1]).date()
    requested_start = pd.to_datetime(start_date).date()

    if df.empty or (last_date < requested_start):
        logger.error(f"Insufficient data for {ticker} after offsets.")
        raise ValueError(f"Insufficient data for {ticker} after offsets.")
         
    return df

def build_training_dataset(run: TrainConfig, model_config=None):
    run = ensure(run, TrainConfig)
    if model_config == None:
        if run.model_config_path == None: raise ValueError("Training requires model_config")
        model_config = ModelConfig.from_name(run.model_config_path)

    loaded = load_dataset(run, model_config)
    if not loaded.empty:
        logger.info(f"Saved processed dataset loaded for {run.ticker}" +
                    f" | Period: {run.start_date} to {run.end_date}" +
                    f" | Interval: {run.interval}" +
                    f" | Features from configuration file: {run.model_config_path}")

        return loaded
    
    df = build_dataset_by_parameters(
        run.ticker,
        run.start_date, run.end_date,
        run.interval,
        model_config.features, model_config.macro_features,
        model_config.target,
        model_config.hyperparameters,
        data_config=run.data_config)

    save_processed_data(df, run, model_config)
    return df

def build_input(predict_config: PredictConfig):
    predict_config = ensure(predict_config, PredictConfig)
    model_metadata = ModelMetadata.from_name(predict_config.model_path)

    return build_dataset_by_parameters(
        model_metadata.ticker,
        predict_config.start_date, predict_config.end_date,
        model_metadata.interval,
        model_metadata.features, model_metadata.macro_features,
        model_metadata.target,
        model_metadata.hyperparameters)

def build_evaluation_dataset(evaluate_config: EvaluateConfig, model_metadata: ModelMetadata):
    evaluate_config = ensure(evaluate_config, EvaluateConfig)
    model_metadata = ensure(model_metadata, ModelMetadata)

    return build_dataset_by_parameters(
        evaluate_config.ticker,
        evaluate_config.start_date, evaluate_config.end_date,
        evaluate_config.interval,
        model_metadata.features, model_metadata.macro_features,
        model_metadata.target,
        model_metadata.hyperparameters)
