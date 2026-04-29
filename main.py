from src.models.train import train
from src.utils.config_utils import load_run_config

train(load_run_config("xgb_AAPL_2005-01-01_2025-01-01_1d.json"))
