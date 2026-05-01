from src.config.base_config import BaseConfig

from src.settings.config import MODELS_DIR

class ModelMetadata(BaseConfig):
    CLASS_DIR = MODELS_DIR

    @classmethod
    def from_name(cls, model_path):
        path = MODELS_DIR / model_path / "metadata.json"
        return cls.from_file(path)

    @property
    def ticker(self): return self._data["ticker"]

    @property
    def start_date(self): return self._data["start_date"]

    @property
    def end_date(self): return self._data["end_date"]

    @property
    def interval(self): return self._data["interval"]

    @property
    def split(self): return self._data["split"]

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
    def feature_importances(self): return self._data["feature_importances"]

    @property
    def model_path_config(self): return self._data["model_config_path"]