from dataclasses import dataclass, field
from typing import List, Dict, Any

from src.settings.config import EVALUATE_CONFIG_DIR

from src.config.file_resolvable import FileResolvable
from src.config.common import TargetConfig

@dataclass
class EvaluateConfig(FileResolvable):
    CLASS_DIR = EVALUATE_CONFIG_DIR
    
    models: List[str]
    ticker: str
    start_date: str
    end_date: str
    interval: str
    targets: List[TargetConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        raw_targets = data.get("targets")
        if raw_targets is None:
            single_target = data.get("target")
            raw_targets = [single_target] if single_target else []

        return cls(
            models=data.get("models", []),
            ticker=data.get("ticker"),
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
            interval=data.get("interval"),
            targets=[
                TargetConfig(name=t["name"], params=t.get("params", {})) 
                for t in raw_targets if t
            ]
        )