from src.config.base_config import BaseConfig
from src.settings.config import PREDICTION_CONFIG_DIR

class PredictConfig(BaseConfig):
    CLASS_DIR = PREDICTION_CONFIG_DIR

    @property
    def start_date(self): return self._data["start_date"]

    @property
    def end_date(self): return self._data["end_date"]

    @property
    def model_path(self): return self._data["model_path"]