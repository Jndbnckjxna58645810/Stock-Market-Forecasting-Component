import json

from src.settings.config import BASE_DIR
from src.utils.path_utils import resolve_path

class BaseConfig:
    def __init__(self, data: dict): self._data = data

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as f: data = json.load(f)
        return cls(data)

    def get(self, key, default=None): return self._data.get(key, default)

    def to_dict(self): return self._data

    CLASS_DIR = BASE_DIR

    @classmethod
    def from_name(cls, name):
        if cls.CLASS_DIR is None:
            raise ValueError(f"{cls.__name__} has no CLASS_DIR defined")

        path = resolve_path(name, cls.CLASS_DIR)
        return cls.from_file(path)