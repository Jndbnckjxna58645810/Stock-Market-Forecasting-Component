from src.config.base_config import BaseConfig

class ModelMetadata(BaseConfig):
    @property
    def ticker(self): return self._data["ticker"]

    @property
    def interval(self): return self._data["interval"]

    @property
    def features(self): return self._data["features"]

    @property
    def selected_features(self): return self._data["selected_features"]

    @property
    def macro_features(self): return self._data["macro_features"]

    @property
    def target(self): return self._data["target"]

    @property
    def model(self): return self._data["model"]

    @property
    def metrics(self): return self._data["metrics"]

    @property
    def model_path_config(self): return self._data["model_config_path"]