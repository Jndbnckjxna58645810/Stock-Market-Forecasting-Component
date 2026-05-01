import pandas as pd

from src.config.evaluate_config import EvaluateConfig
from src.config.model_metadata import ModelMetadata

from src.utils.config_utils import ensure

def validate_models_compatible(evaluate_config: EvaluateConfig):
    evaluate_config = ensure(evaluate_config, EvaluateConfig)
    models_metadata = []
    for model in evaluate_config.models:
        models_metadata.append(ModelMetadata.from_name(model))
    
    if not models_metadata: raise ValueError("No models have been selected")

    for m in models_metadata:
        m = ensure(m, ModelMetadata)
        if m.ticker != evaluate_config.ticker: raise ValueError(f"Invalid ticker: {m.ticker}")
        if m.interval != evaluate_config.interval: raise ValueError(f"Invalid interval: {m.interval}")

        if m.target != models_metadata[0].target: raise ValueError(f"Invalid target: {m.target}")

        train_start, train_end = pd.to_datetime(m.start_date), pd.to_datetime(m.split["train_end"])
        if (train_start <= pd.to_datetime(evaluate_config.end_date)) and (
            train_end >= pd.to_datetime(evaluate_config.start_date)):
            raise ValueError(f"Overlap with training data: {train_start} - {train_end}")