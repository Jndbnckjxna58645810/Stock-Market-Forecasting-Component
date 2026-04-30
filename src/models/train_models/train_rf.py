import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.pipeline.build_dataset import build_dataset

from src.utils.model_utils import save_model
from src.utils.config_utils import ensure

from src.config.train_config import TrainConfig
from src.config.model_config import ModelConfig

def train_rf(run: TrainConfig, model_config: ModelConfig):
    run = ensure(run, TrainConfig)
    df = build_dataset(run)

    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(**model_config["params"])
    return None