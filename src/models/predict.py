from src.utils.config_utils import ensure
from src.utils.model_utils import load_model

from src.pipeline.build_dataset import build_input

from src.config.predict_config import PredictConfig
from src.config.model_metadata import ModelMetadata

def predict(predict_config):
    predict_config = ensure(predict_config, PredictConfig)

    bundle = load_model(predict_config.model_path)
    model, scaler = bundle.get("model"), bundle.get("scaler")

    X = build_input(predict_config)[ModelMetadata.from_name(
            predict_config.model_path).selected_features]
    if scaler != None: X = scaler.transform(X)

    return model.predict(X)