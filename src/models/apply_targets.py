from src.features.targets import *

from src.utils.config_utils import ensure

from src.config.train_config import TrainConfig
from src.config.model_config import ModelConfig
from src.config.model_metadata import ModelMetadata

TARGET_FUNCTIONS = {
    "return": target_return, "direction": target_direction,
    "price": target_price, "multi_return": target_multi_return
}

def apply_target_by_parameters(df, target_config):
    name = target_config["name"]
    params = target_config.get("params", {})

    func = TARGET_FUNCTIONS[name]
    result = func(df, **params)

    target_cols = []

    if isinstance(result, pd.DataFrame):
        for col in result.columns:
            df[col] = result[col]
            target_cols.append(col)
    else:
        col_name = f"target_{name}"
        df[col_name] = result
        target_cols.append(col_name)

    return df, target_cols

def apply_target_to_dataset(df, run: TrainConfig, model_config=None):
    run = ensure(run, TrainConfig)
    if model_config == None:
        model_config = ModelConfig.from_name(run.model_config_path)
    df, target_cols = apply_target_by_parameters(df, model_config.target)
    return df, target_cols

def apply_target_to_evaluation_dataset(df, model_metadata: ModelMetadata):
    model_metadata = ensure(model_metadata, ModelMetadata)
    df, target_cols = apply_target_by_parameters(df, model_metadata.target)
    return df, target_cols
