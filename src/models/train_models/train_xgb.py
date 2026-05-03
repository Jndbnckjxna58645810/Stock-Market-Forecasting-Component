import numpy as np
import pandas as pd
import datetime as dt

from src.pipeline.prepare_training_data import prepare_training_data

from src.models.shared.metrics import compute_multi_target_metrics

from src.utils.config_utils import ensure
from src.utils.logging_utils import get_logger

from src.config.train_config import TrainConfig
from src.config.model_config import ModelConfig

logger = get_logger("models.train_models.train_xgb")

def train_xgb(run: TrainConfig, model_config=None):
    run = ensure(run, TrainConfig)
    if model_config == None:
        if run.model_config_path == None: raise ValueError("Training requires model_config")
        model_config = ModelConfig.from_name(run.model_config_path)
    
    logger.info(f"XGB Training initiated for {run.ticker}" +
                f" | Period: {run.start_date} to {run.end_date}" +
                f" | Interval: {run.interval}" + (
                    f" | Parameters from configuration file: {run.model_config_path}"
                    if run.model_config_path else ""))

    data = prepare_training_data(run, model_config)

    df, target_cols = data["df"], data["target_cols"] #careful here
    X_train, y_train = data["X_train"], data["y_train"].values
    X_val, y_val = data["X_val"], data["y_val"].values
    X_test, y_test = data["X_test"], data["y_test"].values
    
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]

    X_train = X_train.drop(columns=to_drop)
    X_val = X_val.drop(columns=to_drop, errors="ignore")
    X_test = X_test.drop(columns=to_drop, errors="ignore")

    logger.info(f"Columns dropped based on correlation: {', '.join(to_drop)}")

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

    raw_preds = model.predict(X_test[selected])
    preds_df = pd.DataFrame(raw_preds, index=X_test.index, columns=target_cols)
    
    y_test_df = data["y_test"]

    common_idx = y_test_df.index.intersection(preds_df.index)
    y_true = y_test_df.loc[common_idx]
    y_pred = preds_df.loc[common_idx]

    logger.info(f"XGB Model training completed")

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
        "target_cols": target_cols,

        "hyperparameters": model_config.hyperparameters,

        "model": model_config.model,

        "metrics": compute_multi_target_metrics(y_true, y_pred),

        "feature_importances": importances.to_dict(),

        "n_rows": len(df),
        "created_at": dt.datetime.now().strftime("%Y%m%d_%H%M%S"),

        "model_config_path": run.model_config_path,
    }

    from src.models.registry import save_model
    return save_model({"model": model, "x_scaler": None, "y_scaler": None},
                      metadata, run, model_config)