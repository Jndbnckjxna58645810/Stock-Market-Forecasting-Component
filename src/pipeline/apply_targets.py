from src.features.targets import *

from src.utils.logging_utils import get_logger

logger = get_logger("pipeline.apply_targets")

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

            logger.info(f"Target applied: {col} from name '{name}' with parameters: {params}")
    else:
        col_name = f"target_{name}"
        df[col_name] = result
        target_cols.append(col_name)

        logger.info(f"Target applied: {col_name} from name '{name}' with parameters: {params}")

    return df, target_cols