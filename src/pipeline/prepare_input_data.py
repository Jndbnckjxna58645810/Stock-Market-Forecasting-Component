from src.utils.config_utils import ensure

from src.pipeline.build_dataset import build_dataset_by_parameters
from src.config.predict_config import PredictConfig
from src.config.model_metadata import ModelMetadata

def prepare_input_data(predict_config: PredictConfig):
    predict_config = ensure(predict_config, PredictConfig)
    model_metadata = ModelMetadata.from_name(predict_config.model_path)

    return build_dataset_by_parameters(
        model_metadata.ticker,
        predict_config.start_date, predict_config.end_date,
        model_metadata.interval,
        model_metadata.features, model_metadata.macro_features,
        model_metadata.targets,
        model_metadata.model.hyperparameters)