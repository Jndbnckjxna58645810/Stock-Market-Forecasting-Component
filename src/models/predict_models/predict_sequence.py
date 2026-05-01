import numpy as np

from src.utils.config_utils import ensure
from src.pipeline.build_dataset import build_input

from src.models.load_model import load_model

from src.pipeline.create_sequences import create_sequences

from src.config.predict_config import PredictConfig
from src.config.model_metadata import ModelMetadata

def predict_sequence_by_parameters(model_bundle, X, seq_len):
    model = model_bundle.get("model")
    x_scaler = model_bundle.get("x_scaler")
    y_scaler = model_bundle.get("y_scaler")

    if x_scaler: X = x_scaler.transform(X)

    X_seq, _ = create_sequences(X, np.zeros(len(X)), seq_len)

    preds_scaled = model.predict(X_seq)

    if y_scaler:
        return y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    
    return preds_scaled.ravel()

def predict_sequence(predict_config):
    predict_config = ensure(predict_config, PredictConfig)
    metadata = ModelMetadata.from_name(predict_config.model_path)
    
    return predict_sequence_by_parameters(
        load_model(predict_config.model_path),
        build_input(predict_config)[metadata.selected_features],
        metadata.hyperparameters.get("seq_len", 20))