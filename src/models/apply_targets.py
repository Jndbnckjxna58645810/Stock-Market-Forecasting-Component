from src.features.targets import *

from src.utils.config_utils import ensure

from src.config.run_config import RunConfig
from src.config.model_config import ModelConfig

TARGET_FUNCTIONS = {
    "return": target_return, "direction": target_direction,
    "price": target_price, "multi_return": target_multi_return
}

def apply_target(df, run : RunConfig):
    run = ensure(run, RunConfig)
    target_config = ModelConfig.from_name(run.model_config_path).target

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