import pandas as pd

from src.config.evaluate_config import EvaluateConfig
from src.config.model_metadata import ModelMetadata

from src.utils.config_utils import ensure
from src.utils.logging_utils import get_logger

logger = get_logger("models.shared.validate")

def validate_models_compatible(evaluate_config: EvaluateConfig):
    evaluate_config = ensure(evaluate_config, EvaluateConfig)

    models_metadata = []
    for model in evaluate_config.models:
        m = ModelMetadata.from_name(model)
        models_metadata.append(m)

        message = (f" | Expected: ticker={evaluate_config.ticker}, " +
                   f"interval={evaluate_config.interval} | Targets must be in the same order" +
                   f" | Evaluation period: {evaluate_config.start_date} to {evaluate_config.end_date}" +
                   f" | Actual: ticker={m.ticker}, interval={m.interval}" +
                   f" | Training period: {m.start_date} to {m.end_date}")

        if m.ticker != evaluate_config.ticker:
            logger.error(f"Ticker mismatch for {model}{message}")
            raise ValueError(f"Ticker mismatch for {model}{message}")
        
        if m.interval != evaluate_config.interval:
            logger.error(f"Interval mismatch for {model}{message}")
            raise ValueError(f"Interval mismatch for {model}{message}")

        if m.targets != evaluate_config.targets:
            logger.error(f"Target mismatch for {model}{message}")
            raise ValueError(f"Target mismatch for {model}{message}")

        train_start, train_end = pd.to_datetime(m.start_date), pd.to_datetime(m.split.train_end)
        if (train_start <= pd.to_datetime(evaluate_config.end_date)) and (
            train_end >= pd.to_datetime(evaluate_config.start_date)):
            logger.error(f"Overlap with training data for {model}{message}")
            raise ValueError(f"Overlap with training data for {model}{message}")
        
    if not models_metadata:
        message = (f"No models selected for evaluation" +
                   f" | ticker={evaluate_config.ticker}" +
                   f" | interval={evaluate_config.interval}" +
                   f" | Period: {evaluate_config.start_date} to {evaluate_config.end_date}")
        logger.error(message)
        
        raise ValueError(message)
    
    logger.info("Validation completed")