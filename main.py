from src.models.train import train
from src.models.predict import predict
from src.utils.model_utils import load_model
from src.pipeline.build_dataset import build_input
from src.config.train_config import TrainConfig
from src.config.model_config import ModelConfig
from src.config.evaluate_config import EvaluateConfig
from src.models.evaluate_models.evaluate import evaluate_models

#run = TrainConfig.from_name("xgb_AAPL_other.json")
#train(run, ModelConfig.from_name(run.model_config_path))

#print(build_input("xgb_AAPL_1d_20260429_224558_02-01-2025.json"))
#print(predict("xgb_AAPL_1d_20260429_224558_02-01-2025.json"))

print(evaluate_models("evaluation_example.json"))