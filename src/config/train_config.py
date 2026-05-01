from src.config.base_config import BaseConfig
from src.settings.config import TRAIN_CONFIG_DIR

class TrainConfig(BaseConfig):
    CLASS_DIR = TRAIN_CONFIG_DIR

    @property
    def ticker(self): return self._data["ticker"]

    @property
    def start_date(self): return self._data["start_date"]

    @property
    def end_date(self): return self._data["end_date"]

    @property
    def interval(self): return self._data["interval"]

    @property
    def split(self): return self._data.get("split", {})

    @property
    def data_config(self): return self._data.get("data", {})

    @property
    def model_config_path(self): return self._data["model_config"]