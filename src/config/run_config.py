from src.config.base_config import BaseConfig
from src.settings.config import RUN_CONFIG_DIR

class RunConfig(BaseConfig):
    CLASS_DIR = RUN_CONFIG_DIR
    
    @property
    def ticker(self): return self._data["ticker"]

    @property
    def start_date(self): return self._data["start_date"]

    @property
    def end_date(self): return self._data["end_date"]

    @property
    def interval(self): return self._data["interval"]

    @property
    def data_config(self): return self._data.get("data", {})

    @property
    def model_config_path(self): return self._data["model_config"]