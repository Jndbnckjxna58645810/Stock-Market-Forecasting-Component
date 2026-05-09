import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path

class FileResolvable:
    CLASS_DIR: Path = None

    @classmethod
    def from_name(cls, name: str):
        if cls.CLASS_DIR is None:
            raise ValueError(f"{cls.__name__} has no CLASS_DIR defined")
        
        filename = name if name.endswith(".json") else f"{name}.json"
        path = cls.CLASS_DIR / filename
        
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self): return asdict(self)