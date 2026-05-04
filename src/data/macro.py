import math
import pandas as pd
from fredapi import Fred

from src.settings.config import FRED_API_KEY, RAW_DATA_DIR
from src.pipeline.preprocessing import get_max_lookback_by_parameters

from src.utils.io_utils import load_csv, save_csv
from src.utils.config_utils import ensure, make_signature
from src.utils.logging_utils import get_logger

from src.config.train_config import TrainConfig
from src.config.model_config import ModelConfig
from src.config.predict_config import PredictConfig
from src.config.model_metadata import ModelMetadata
from src.config.evaluate_config import EvaluateConfig

logger = get_logger("data.macro")

def get_fred(): return Fred(api_key=FRED_API_KEY)

def load_macro_by_parameters(macro_features, features, start_date, end_date, force_download=False, save_macro=False, path=None):
    default_path = RAW_DATA_DIR / f"macro_{start_date}_{end_date}_{make_signature(macro_features)}.csv"
    if default_path.exists() and not force_download:
        df = load_csv(default_path)

        logger.info(f"Data loaded from: {default_path}")
    else:
        dfs, fred = [], get_fred()
        ms = pd.to_datetime(start_date) - pd.DateOffset(days=math.ceil(get_max_lookback_by_parameters(features) / 31) * 31)
        
        for macro in macro_features:
            name, source = macro["name"], macro["source"]
            m = fred.get_series(source, ms, end_date).to_frame(name=name)
            if m.empty:
                logger.error(f"No data found for {name} ({source})")

                raise ValueError(f"No data found for {name} ({source})")
            dfs.append(m)

            logger.info(f"Data loaded for {name} ({source})")

        df = pd.concat(dfs, axis=1)
        df.index.name = "date"

        logger.info(f"Data loaded from fredapi")
    
    if save_macro:
        save_csv(df, default_path if path == None else path)

        logger.info(f"Data saved to {default_path if path == None else path}")
    return df

def load_macro_dataset(run: TrainConfig, model_config=None):
    run = ensure(run, TrainConfig)
    if model_config == None:
        model_config = ModelConfig.from_name(run.model_config_path)

    logger.info(f"Loading training data for {run.ticker}" +
                f" | Period: {run.start_date} to {run.end_date}")
    
    return load_macro_by_parameters(
        model_config.macro_features, model_config.features,
        run.start_date, run.end_date,
        save_macro=run.data_config["macro"]["save"],
        force_download=run.data_config["macro"]["force_download"],
        path=run.data_config["macro"]["path"])

def load_macro_input(predict_config: PredictConfig):
    predict_config = ensure(predict_config, PredictConfig)
    model_metadata = ModelMetadata.from_name(predict_config.model_path)

    offset_start = get_max_lookback_by_parameters(model_metadata.features) * 2 + 1
    if model_metadata.hyperparameters.get("seq_len"):
        offset_start += 2 * model_metadata.hyperparameters.get("seq_len")

    logger.info(f"Loading input data for {model_metadata.ticker}" +
                f" | Period: {predict_config.start_date} to {predict_config.end_date}" +
                f" | Offset days: {offset_start}")
    
    return load_macro_by_parameters(
        model_metadata.macro_features, model_metadata.features,
        pd.to_datetime(predict_config.start_date) - pd.DateOffset(days=offset_start),
        pd.to_datetime(predict_config.end_date) + pd.DateOffset(days=1+len(model_metadata.target_cols)*2),
        save_macro=False, force_download=True, path=None)

def load_macro_evaluation_dataset(evaluate_config: EvaluateConfig, model_metadata: ModelMetadata):
    evaluate_config = ensure(evaluate_config, EvaluateConfig)
    model_metadata = ensure(model_metadata, ModelMetadata)

    offset_start = get_max_lookback_by_parameters(model_metadata.features) * 2 + 1
    if model_metadata.hyperparameters.get("seq_len"):
        offset_start += 2 * model_metadata.hyperparameters.get("seq_len")

    logger.info(f"Loading evaluation data for {model_metadata.ticker}" +
                f" | Period: {evaluate_config.start_date} to {evaluate_config.end_date}" +
                f" | Offset days: {offset_start}")
    
    return load_macro_by_parameters(
        model_metadata.macro_features,
        model_metadata.features,
        pd.to_datetime(evaluate_config.start_date) - pd.DateOffset(days=offset_start),
        pd.to_datetime(evaluate_config.end_date) + pd.DateOffset(days=1),
        save_macro=False, force_download=True, path=None)
