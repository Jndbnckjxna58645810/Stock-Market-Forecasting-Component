import numpy as np
import datetime as dt

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

from src.pipeline.prepare_training_data import prepare_training_data
from src.pipeline.create_sequences import create_sequences

from src.models.shared.metrics import compute_metrics

from src.utils.config_utils import ensure
from src.utils.logging_utils import get_logger

from src.config.train_config import TrainConfig
from src.config.model_config import ModelConfig

logger = get_logger("models.train_models.train_lstm")

def train_lstm(run: TrainConfig, model_config=None):
    run = ensure(run, TrainConfig)
    if model_config is None:
        if run.model_config_path is None: raise ValueError("Training requires model_config")
        model_config = ModelConfig.from_name(run.model_config_path)

    hp = model_config.hyperparameters
    seq_len = hp.get("seq_len", 20)
    units = hp.get("units", 64)
    epochs = hp.get("epochs", 10)
    batch_size = hp.get("batch_size", 32)
    dropout = hp.get("dropout", 0.2)
    learning_rate = hp.get("learning_rate", 0.001)

    logger.info(f"LSTM Training initiated for {run.ticker}" +
                f" | Period: {run.start_date} to {run.end_date}" +
                f" | Interval: {run.interval}" + (
                    f" | Parameters from configuration file: {run.model_config_path}"
                    if run.model_config_path else "") +
                    f" | Hyperparameters used: seq_len={seq_len}, " +
                    f"units={units}, epochs={epochs}, " +
                    f"batch_size={batch_size}, dropout={dropout}, " +
                    f"learning_rate={learning_rate}")

    data = prepare_training_data(run, model_config)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train)
    X_val_scaled = x_scaler.transform(X_val)
    X_test_scaled = x_scaler.transform(X_test)

    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)
    y_test_scaled = y_scaler.transform(y_test)

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_len)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, seq_len)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, seq_len)

    num_targets = y_train.shape[1]

    model = Sequential([
        LSTM(units, input_shape=(seq_len, X_train_seq.shape[2]), return_sequences=True),
        Dropout(dropout),
        LSTM(units // 2),
        Dropout(dropout),
        Dense(num_targets)
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=1
    )

    preds_scaled = model.predict(X_test_seq)
    
    preds = y_scaler.inverse_transform(preds_scaled)
    y_test_final = y_scaler.inverse_transform(y_test_seq)

    logger.info(f"LSTM Model training completed")

    metadata = {
        "ticker": run.ticker,
        "start_date": run.start_date,
        "end_date": run.end_date,
        "interval": run.interval,

        "split": run.split,

        "features": model_config.features,
        "selected_features": data["df"].drop(
            columns=data["target_cols"]).columns.tolist(),
        "macro_features": model_config.macro_features,

        "target": model_config.target,
        "target_cols": data["target_cols"],

        "hyperparameters": model_config.hyperparameters,

        "model": model_config.model,

        "metrics": compute_metrics(y_test_final, preds),
        
        "feature_importances": None,

        "n_rows": len(data["df"]),
        "created_at": dt.datetime.now().strftime("%Y%m%d_%H%M%S"),

        "model_config_path": run.model_config_path,
    }

    from src.models.registry import save_model
    return save_model({"model": model, "x_scaler": x_scaler, "y_scaler": y_scaler},
                      metadata, run, model_config)