import numpy as np
import pandas as pd
import datetime as dt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from src.pipeline.build_dataset import build_dataset

from src.models.apply_targets import apply_target

from src.utils.model_utils import save_model
from src.utils.config_utils import ensure

from src.config.run_config import RunConfig
from src.config.model_config import ModelConfig

def train_xgb(run: RunConfig, model_config: ModelConfig):
    run = ensure(run, RunConfig)
    df = build_dataset(run)

    df, target_cols = apply_target(df, run)
    df = df.dropna()

    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    df = df.drop(columns=to_drop)

    X = df.drop(columns=target_cols)
    y = df[target_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    import xgboost as xgb
    model = xgb.XGBRegressor(**model_config.model["params"])

    model.fit(X_train, y_train)

    importances = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    k = int(len(importances) * 0.8)
    selected = importances.iloc[:k].index.tolist()

    model.fit(X_train[selected], y_train)

    preds = model.predict(X_test[selected])

    print("R2:", r2_score(y_test, preds))
    print("MSE:", mean_squared_error(y_test, preds))

    metadata = {
        "ticker": run.ticker,
        "interval": run.interval,

        "features": model_config.features,
        "selected_features": selected,
        "macro_features": model_config.macro_features,

        "target": model_config.target,

        "model": model_config.model,

        "metrics": {
            "r2": r2_score(y_test, preds),
            "mse": mean_squared_error(y_test, preds)
        },

        "n_rows": len(df),
        "created_at": dt.datetime.now().strftime("%Y%m%d_%H%M%S"),

        "model_config_path": run.model_config_path,
    }

    return save_model(model, metadata, run, model_config)