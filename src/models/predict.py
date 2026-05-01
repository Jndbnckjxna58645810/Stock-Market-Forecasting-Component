from src.utils.config_utils import ensure

from src.models.predict_models.predict_tabular import predict_tabular
from src.models.predict_models.predict_sequence import predict_sequence

from src.config.model_metadata import ModelMetadata
from src.config.predict_config import PredictConfig

def predict(predict_config: PredictConfig):
    predict_config = ensure(predict_config, PredictConfig)
    model_metadata = ModelMetadata.from_name(predict_config.model_path)

    model_name = model_metadata.model["name"]
    if model_name == "xgb" or model_name == "rf": return predict_tabular(predict_config)
    elif model_name == "lstm": return predict_sequence(predict_config)
    else: raise ValueError("Unknown model")