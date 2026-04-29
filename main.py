from src.models.train import train
from src.utils.config_utils import load_run_config, load_model_config

run = load_run_config("xgb_AAPL_2005-01-01_2025-01-01_1d.json")
train(run, load_model_config(run.model_config_path))
