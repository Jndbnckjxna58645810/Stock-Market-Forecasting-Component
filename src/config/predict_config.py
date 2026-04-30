from src.config.base_config import BaseConfig
from src.settings.config import PREDICTION_CONFIG_DIR

class PredictConfig(BaseConfig):
    CLASS_DIR = PREDICTION_CONFIG_DIR

    @property
    def input_date(self): return self._data["input_date"]

    @property
    def model_path(self): return self._data["model_path"]