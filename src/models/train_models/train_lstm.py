import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.pipeline.build_dataset import build_dataset

from src.models.apply_targets import apply_target

from src.utils.model_utils import save_model
from src.utils.config_utils import ensure

from src.config.run_config import RunConfig
from src.config.model_config import ModelConfig

def train_lstm(run: RunConfig, model_cofig: ModelConfig):
    run = ensure(run, RunConfig)
    df = build_dataset(run)
    return None