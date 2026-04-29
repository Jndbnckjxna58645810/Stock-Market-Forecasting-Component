from pathlib import Path
import json

class BaseConfig:
    def __init__(self, data: dict): self._data = data

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as f: data = json.load(f)
        return cls(data)

    def get(self, key, default=None): return self._data.get(key, default)

    def to_dict(self): return self._data