import numpy as np
import pandas as pd
import datetime as dt

from sklearn.metrics import r2_score, mean_squared_error

from src.pipeline.build_dataset import build_dataset

from src.models.apply_targets import apply_target_to_dataset

from src.utils.model_utils import save_model
from src.utils.config_utils import ensure

from src.config.train_config import TrainConfig
from src.config.model_config import ModelConfig

def train_xgb(run: TrainConfig, model_config: ModelConfig):
    run = ensure(run, TrainConfig)
    df = build_dataset(run)

    df, target_cols = apply_target_to_dataset(df, run)
    df = df.dropna()

    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    df = df.drop(columns=to_drop)

    X = df.drop(columns=target_cols)
    y = df[target_cols]

    train_end = run.split["train_end"]
    val_end = run.split["val_end"]

    X_train = X.loc[:train_end].iloc[:-1]
    y_train = y.loc[:train_end].iloc[:-1]

    X_val = X.loc[train_end:val_end].iloc[:-1]
    y_val = y.loc[train_end:val_end].iloc[:-1]

    X_test = X.loc[val_end:]
    y_test = y.loc[val_end:]

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
        "start_date": run.start_date,
        "end_date": run.end_date,
        "interval": run.interval,

        "split": run.split,

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

    return save_model({"model": model, "scaler": None},
                      metadata, run, model_config)