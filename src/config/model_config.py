from src.config.file_resolvable import FileResolvable
from src.config.common import TargetConfig, ModelSettings

from src.settings.config import MODELS_CONFIG_DIR

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class ModelConfig(FileResolvable):
    CLASS_DIR = MODELS_CONFIG_DIR
    features: List[Dict[str, Any]] = field(default_factory=list)
    targets: List[TargetConfig] = field(default_factory=list)
    model: ModelSettings = field(default_factory=ModelSettings)
    macro_features: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            features=[f for f in data.get("features", []) if isinstance(f, dict)],
            targets=[TargetConfig.from_dict(t) for t in data.get("targets", [])],
            model=ModelSettings(
                name=data.get("model", {}).get("name", "xgb"),
                params=data.get("model", {}).get("params", {}),
                hyperparameters=data.get("model", {}).get("hyperparameters", {})
            ),
            macro_features=data.get("macro_features", [])
        )