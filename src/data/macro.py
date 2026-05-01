import math
import pandas as pd
from fredapi import Fred

from src.settings.config import FRED_API_KEY, RAW_DATA_DIR
from src.pipeline.preprocessing import get_max_lookback_by_parameters
from src.utils.csv_utils import load_csv, save_csv
from src.utils.config_utils import ensure
from src.utils.vesrioning_utils import make_signature

from src.config.train_config import TrainConfig
from src.config.model_config import ModelConfig
from src.config.predict_config import PredictConfig
from src.config.model_metadata import ModelMetadata
from src.config.evaluate_config import EvaluateConfig

def get_fred(): return Fred(api_key=FRED_API_KEY)

def load_macro_by_parameters(macro_features, features, start_date, end_date, force_download=False, save_macro=False, path=None):
    default_path = RAW_DATA_DIR / f"macro_{start_date}_{end_date}_{make_signature(macro_features)}.csv"
    if default_path.exists() and not force_download: df = load_csv(default_path)
    else:
        dfs, fred = [], get_fred()
        ms = pd.to_datetime(start_date) - pd.DateOffset(days=math.ceil(get_max_lookback_by_parameters(features) / 31) * 31)
        
        for macro in macro_features:
            name, source = macro["name"], macro["source"]
            m = fred.get_series(source, ms, end_date).to_frame(name=name)
            if m.empty: raise ValueError(f"No data for {name} ({source})")
            dfs.append(m)

        df = pd.concat(dfs, axis=1)
        df.index.name = "date"
    
    if save_macro: save_csv(df, default_path if path == None else path)
    return df

def load_macro_dataset(run: TrainConfig, model_config=None):
    run = ensure(run, TrainConfig)
    if model_config == None:
        model_config = ModelConfig.from_name(run.model_config_path)
    
    return load_macro_by_parameters(
        model_config.macro_features, model_config.features,
        run.start_date, run.end_date,
        save_macro=run.data_config["macro"]["save"],
        force_download=run.data_config["macro"]["force_download"],
        path=run.data_config["macro"]["path"])

def load_macro_input(predict_config: PredictConfig):
    predict_config = ensure(predict_config, PredictConfig)
    model_metadata = ModelMetadata.from_name(predict_config.model_path)

    return load_macro_by_parameters(
        model_metadata.macro_features, model_metadata.features,
        pd.to_datetime(predict_config.start_date) - pd.DateOffset(days=5),
        pd.to_datetime(predict_config.end_date) + pd.DateOffset(days=1),
        save_macro=False, force_download=True, path=None)

def load_macro_evaluation_dataset(evaluate_config: EvaluateConfig, model_metadata: ModelMetadata):
    evaluate_config = ensure(evaluate_config, EvaluateConfig)
    model_metadata = ensure(model_metadata, ModelMetadata)
    
    return load_macro_by_parameters(
        model_metadata.macro_features,
        model_metadata.features,
        evaluate_config.start_date, evaluate_config.end_date,
        save_macro=False, force_download=True, path=None)
