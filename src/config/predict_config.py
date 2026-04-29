from src.config.base_config import BaseConfig

class RPredictConfig(BaseConfig):
    @property
    def input_date(self): return self._data["input_date"]

    @property
    def model_path(self): return self._data["model_path"]