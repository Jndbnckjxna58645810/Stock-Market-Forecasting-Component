import numpy as np

from src.utils.config_utils import ensure
from src.utils.logging_utils import get_logger

from src.pipeline.build_dataset import build_input
from src.pipeline.create_sequences import create_sequences

from src.config.predict_config import PredictConfig
from src.config.model_metadata import ModelMetadata

logger = get_logger("models.predict_models.predict_sequence")

def predict_sequence_by_parameters(model_bundle, X, seq_len):
    model = model_bundle.get("model")
    x_scaler = model_bundle.get("x_scaler")
    y_scaler = model_bundle.get("y_scaler")

    if x_scaler:
        X_scaled = x_scaler.transform(X)

        logger.info("Scaling performed on X")
    else: X_scaled = X.values

    X_seq, _ = create_sequences(X_scaled, np.zeros(len(X_scaled)), seq_len)

    preds_scaled = model.predict(X_seq)

    if y_scaler:
        preds_final = y_scaler.inverse_transform(preds_scaled)

        logger.info("Descaling performed on y")
    else:
        preds_final = preds_scaled

    dates_final = X.index[seq_len:].strftime('%Y-%m-%d').tolist()
    
    logger.info("Prediction completed")
    return {"dates": dates_final, "preds": preds_final.tolist()}

def predict_sequence(predict_config):
    predict_config = ensure(predict_config, PredictConfig)
    metadata = ModelMetadata.from_name(predict_config.model_path)
    
    logger.info(f"Predicting using {predict_config.model_path}" +
                f" | Period: {predict_config.start_date} to {predict_config.end_date}")

    from src.models.registry import load_model
    return predict_sequence_by_parameters(
        load_model(predict_config.model_path),
        build_input(predict_config)[metadata.selected_features],
        metadata.hyperparameters.get("seq_len", 20))