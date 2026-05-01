from src.utils.config_utils import ensure

from src.config.model_metadata import ModelMetadata
from src.config.evaluate_config import EvaluateConfig

from src.models.shared.validate import validate_models_compatible

from src.models.evaluate_models.evaluate_model_sequence import evaluate_model_sequence
from src.models.evaluate_models.evaluate_model_tabular import evaluate_model_tabular

def evaluate(evaluate_config: EvaluateConfig):
    evaluate_config = ensure(evaluate_config, EvaluateConfig)
    validate_models_compatible(evaluate_config)

    results = {}
    for m in evaluate_config.models:
        metadata = ModelMetadata.from_name(m)
        model_name = metadata.model["name"]
        if model_name == "xgb" or model_name == "rf":
            results[m] = evaluate_model_tabular(evaluate_config, m)
        elif model_name == "lstm":
            results[m] = evaluate_model_sequence(evaluate_config, m)
        else: raise ValueError("Unknown model")

    return results