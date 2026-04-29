from src.config.base_config import BaseConfig

class ModelConfig(BaseConfig):
    @property
    def features(self): return self._data["features"]

    @property
    def target(self): return self._data["target"]

    @property
    def macro_features(self): return self._data["macro_features"]

    @property
    def model(self): return self._data["model"]