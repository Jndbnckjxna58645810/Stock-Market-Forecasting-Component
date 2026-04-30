from src.config.base_config import BaseConfig

from src.settings.config import MODELS_CONFIG_DIR

class ModelConfig(BaseConfig):
    CLASS_DIR  = MODELS_CONFIG_DIR

    @property
    def features(self): return self._data["features"]

    @property
    def target(self): return self._data["target"]

    @property
    def macro_features(self): return self._data["macro_features"]

    @property
    def model(self): return self._data["model"]
