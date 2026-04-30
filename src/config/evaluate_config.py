from src.config.base_config import BaseConfig

from src.settings.config import EVALUATE_CONFIG_DIR

class EvaluateConfig(BaseConfig):
    CLASS_DIR  = EVALUATE_CONFIG_DIR

    @property
    def models(self): return self._data["models"]

    @property
    def ticker(self): return self._data["ticker"]

    @property
    def start_date(self): return self._data["start_date"]

    @property
    def end_date(self): return self._data["end_date"]

    @property
    def interval(self): return self._data["interval"]
