from src.config.model_metadata import ModelMetadata

from src.models.load_models.load_model_tabular import load_model_tabular
from src.models.load_models.load_model_sequence import load_model_sequence

def load_model(name):
    model_metadata = ModelMetadata.from_name(name)
    model_name = model_metadata.model["name"]
    if model_name == "xgb" or model_name == "rf": return load_model_tabular(name)
    elif model_name == "lstm": return load_model_sequence(name)
    else: raise ValueError("Unknown model")
