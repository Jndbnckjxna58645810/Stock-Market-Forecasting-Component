from src.models.train import train
from src.models.predict import predict
from src.utils.model_utils import load_model
from src.pipeline.build_dataset import build_input
from src.config.run_config import RunConfig
from src.config.model_config import ModelConfig

run = RunConfig.from_name("xgb_AAPL_2005-01-01_2025-01-01_1d.json")
#train(run, ModelConfig.from_name(run.model_config_path))

print(predict("xgb_AAPL_1d_20260429_224558_02-01-2025.json"))