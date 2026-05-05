from src.utils.config_utils import ensure

from src.pipeline.build_dataset import build_dataset_by_parameters
from src.pipeline.apply_targets import apply_target_by_parameters

from src.config.evaluate_config import EvaluateConfig
from src.config.model_metadata import ModelMetadata

def prepare_evaluation_data(evaluate_config: EvaluateConfig, model_metadata: ModelMetadata):
    evaluate_config = ensure(evaluate_config, EvaluateConfig)
    model_metadata = ensure(model_metadata, ModelMetadata)

    df = build_dataset_by_parameters(
        evaluate_config.ticker,
        evaluate_config.start_date, evaluate_config.end_date,
        evaluate_config.interval,
        model_metadata.features, model_metadata.macro_features,
        model_metadata.target,
        model_metadata.hyperparameters)
    
    df, target_cols = apply_target_by_parameters(df, evaluate_config.target)

    return df, target_cols