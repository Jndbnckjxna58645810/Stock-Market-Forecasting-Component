import pandas as pd

from src.models.load_models.load_model_tabular import load_model_tabular
from src.models.load_models.load_model_sequence import load_model_sequence

from src.models.predict_models.predict_tabular import predict_tabular
from src.models.predict_models.predict_sequence import predict_sequence

from src.models.evaluate_models.evaluate_model_sequence import evaluate_model_sequence
from src.models.evaluate_models.evaluate_model_tabular import evaluate_model_tabular

from src.models.save_models.save_model_tabular import save_model_tabular
from src.models.save_models.save_model_sequence import save_model_sequence

from src.models.train_models.train_lstm import train_lstm
from src.models.train_models.train_rf import train_rf
from src.models.train_models.train_xgb import train_xgb

from src.utils.config_utils import ensure
from src.utils.logging_utils import get_logger

from src.config.model_metadata import ModelMetadata
from src.config.evaluate_config import EvaluateConfig
from src.config.predict_config import PredictConfig
from src.config.train_config import TrainConfig
from src.config.model_config import ModelConfig

from src.models.shared.validate import validate_models_compatible

logger = get_logger("models.registry")

STRATEGIES = {
    "xgb": {"evaluator": evaluate_model_tabular, "predictor": predict_tabular,
            "loader": load_model_tabular, "saver": save_model_tabular, "trainer": train_xgb},
    "rf": {"evaluator": evaluate_model_tabular, "predictor": predict_tabular,
           "loader": load_model_tabular, "saver": save_model_tabular, "trainer": train_rf},
    "lstm": {"evaluator": evaluate_model_sequence, "predictor": predict_sequence,
            "loader": load_model_sequence, "saver": save_model_sequence, "trainer": train_lstm}}

def evaluate_models(evaluate_config: EvaluateConfig):
    evaluate_config = ensure(evaluate_config, EvaluateConfig)
    validate_models_compatible(evaluate_config)

    logger.info(f"Evaluation started | Models: {', '.join(evaluate_config.models)}")

    results = {}
    for m_name in evaluate_config.models:
        metadata = ModelMetadata.from_name(m_name)
        m_type = metadata.model["name"]

        logger.info(f"Evaluation started | Model: {m_name} | Type: {m_type}")

        if not STRATEGIES.get(m_type):
            logger.error(f"Evaluation failed | Unknown model type: {m_type}")

            raise ValueError(f"Evaluation failed | Unknown model type: {m_type}")

        strategy = STRATEGIES[m_type]["evaluator"]
        results[m_name] = strategy(evaluate_config, m_name)
    
    logger.info(f"Evaluation finished | Models: {', '.join(evaluate_config.models)}")

    return results

def predict(predict_config: PredictConfig):
    predict_config = ensure(predict_config, PredictConfig)
    model_metadata = ModelMetadata.from_name(predict_config.model_path)

    logger.info(f"Prediction started | Model: {predict_config.model_path} | Type: {model_metadata.model['name']}")

    if not STRATEGIES.get(model_metadata.model['name']):
            logger.error(f"Prediction failed | Unknown model type: {model_metadata.model['name']}")

            raise ValueError(f"Prediction failed | Unknown model type: {model_metadata.model['name']}")

    strategy = STRATEGIES[model_metadata.model["name"]]["predictor"]

    result = strategy(predict_config)

    return pd.DataFrame(result["preds"], index=result["dates"], columns=model_metadata.target_cols)

def load_model(name):
    model_metadata = ModelMetadata.from_name(name)

    logger.info(f"Loading model | Model: {name} | Type: {model_metadata.model['name']}")

    if not STRATEGIES.get(model_metadata.model['name']):
            logger.error(f"Model loading failed | Unknown model type: {model_metadata.model['name']}")

            raise ValueError(f"Model loading failed | Unknown model type: {model_metadata.model['name']}")

    strategy = STRATEGIES[model_metadata.model["name"]]["loader"]
    return strategy(name)

def save_model(model, metadata, run: TrainConfig, model_config=None):
    run = ensure(run, TrainConfig)
    if model_config == None:
        model_config = ModelConfig.from_name(run.model_config_path)
    model_config = ensure(model_config, ModelConfig)

    logger.info(f"Saving model" + (
        f" | Model configuration: {run.model_config_path}" if run.model_config_path else "") +
        f" | Type: {model_config.model['name']}")
    
    if not STRATEGIES.get(model_config.model['name']):
            logger.error(f"Model saving failed | Unknown model type: {model_config.model['name']}")

            raise ValueError(f"Model saving failed | Unknown model type: {model_config.model['name']}")

    strategy = STRATEGIES[model_config.model["name"]]["saver"]
    return strategy(model, metadata, run, model_config)

def train(run: TrainConfig, model_config=None):
    run = ensure(run, TrainConfig)
    if model_config == None:
        if run.model_config_path == None: raise ValueError("Training requires model_config")
        model_config = ModelConfig.from_name(run.model_config_path)
    model_config = ensure(model_config, ModelConfig)

    logger.info(f"Training model" + (
        f" | Model configuration: {run.model_config_path}" if run.model_config_path else "") +
        f" | Type: {model_config.model['name']}")

    if not STRATEGIES.get(model_config.model['name']):
            logger.error(f"Training failed | Unknown model type: {model_config.model['name']}")

            raise ValueError(f"Training failed | Unknown model type: {model_config.model['name']}")

    strategy = STRATEGIES[model_config.model["name"]]["trainer"]
    return strategy(run, model_config)