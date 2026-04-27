def build_model(model_config):
    name = model_config["model"]["name"]
    params = model_config["model"]["params"]

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