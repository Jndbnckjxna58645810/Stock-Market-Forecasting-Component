from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

from src.config.file_resolvable import FileResolvable
from src.config.common import SplitConfig, DataStrategy

from src.settings.config import TRAIN_CONFIG_DIR

@dataclass
class TrainConfig(FileResolvable):
    CLASS_DIR = TRAIN_CONFIG_DIR
    
    ticker: str
    start_date: str
    end_date: str
    interval: str
    model_config_path: str
    split: SplitConfig = field(default_factory=SplitConfig)
    data: DataStrategy = field(default_factory=DataStrategy)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            ticker=data.get("ticker"),
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
            interval=data.get("interval"),
            model_config_path=data.get("model_config_path"),
            split=SplitConfig(**data.get("split", {})),
            data=DataStrategy(**data.get("data", {}))
        )