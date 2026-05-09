from dataclasses import dataclass, field
from typing import Dict, Any

from src.settings.config import PREDICTION_CONFIG_DIR

from src.config.file_resolvable import FileResolvable

@dataclass
class PredictConfig(FileResolvable):
    CLASS_DIR = PREDICTION_CONFIG_DIR
    
    start_date: str
    end_date: str
    model_path: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
            model_path=data.get("model_path")
        )