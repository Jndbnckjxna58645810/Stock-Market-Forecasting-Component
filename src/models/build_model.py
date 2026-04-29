from src.config.run_config import RunConfig

from src.utils.config_utils import load_model_config, load_run_config

def build_model(run : RunConfig):
    model_config = load_model_config(run.model_config_path)

    name = model_config.model["name"]
    params = model_config.model["params"]

    if name == "xgb":
        import xgboost as xgb
        return xgb.XGBRegressor(**params)

    elif name == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**params)

    elif name == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(**params)

    else: raise ValueError("Unknown model")