import pandas as pd

from src.features.targets import *

from src.utils.logging_utils import get_logger

from src.config.common import TargetConfig

logger = get_logger("pipeline.apply_targets")

TARGET_FUNCTIONS = {
    "return": target_return, "direction": target_direction,
    "price": target_price, "multi_return": target_multi_return
}

def apply_targets_by_parameters(df, targets):
    all_target_cols = []

    for i, target_config in enumerate(targets):
        name = target_config.name
        params = target_config.params
        
        func = TARGET_FUNCTIONS[name]
        result = func(df, **params)

        param_suffix = f"_{i}"
        if "horizon" in params:
            param_suffix += f"_h{params['horizon']}"
        if "smoothing" in params and params["smoothing"] > 1:
            param_suffix += f"_s{params['smoothing']}"
        if "log" in params and params["log"]:
            param_suffix += "_log"
        
        threshold = params.get("threshold", 0)
        if threshold > 0.0001:
            t_str = str(threshold).replace(".", "")
            param_suffix += f"_t{t_str}"

        if isinstance(result, pd.DataFrame):
            for col in result.columns:
                unique_col = f"{col}{param_suffix}"
                df[unique_col] = result[col]
                all_target_cols.append(unique_col)
                logger.info(f"Target applied: {unique_col} (from {name})")
        else:
            unique_col = f"target_{name}{param_suffix}"
            df[unique_col] = result
            all_target_cols.append(unique_col)
            logger.info(f"Target applied: {unique_col}")

    return df, all_target_cols