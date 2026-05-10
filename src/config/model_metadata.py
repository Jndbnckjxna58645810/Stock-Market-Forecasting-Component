from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.settings.config import MODELS_DIR

from src.config.file_resolvable import FileResolvable
from src.config.common import TargetConfig, SplitConfig, ModelSettings

@dataclass
class ModelMetadata(FileResolvable):
    CLASS_DIR = MODELS_DIR

    ticker: str
    interval: str
    created_at: str
    n_rows: int

    start_date: str
    end_date: str
    split: SplitConfig
    features: List[Dict]
    selected_features: List[str]
    macro_features: List[Dict]

    model: ModelSettings
    model_config_path: str

    targets: List[TargetConfig]
    target_cols: List[str]
    metrics: Dict[str, float]
    feature_importances: Optional[Dict[str, float]] = None

    @classmethod
    def from_name(cls, model_folder_name: str):
        path = MODELS_DIR / model_folder_name / "metadata.json"
        with open(path, "r") as f:
            import json
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            ticker=data.get("ticker"),
            interval=data.get("interval"),
            created_at=data.get("created_at"),
            n_rows=data.get("n_rows", 0),
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
            split=SplitConfig(**data.get("split", {})),
            features=data.get("features", []),
            selected_features=data.get("selected_features", []),
            macro_features=data.get("macro_features", []),
            model=ModelSettings(
                name=data.get("model", {}).get("name", "xgb"),
                params=data.get("model", {}).get("params", {}),
                hyperparameters=data.get("model", {}).get("hyperparameters", {})
            ),
            model_config_path=data.get("model_config_path"),
            targets=[TargetConfig.from_dict(t) for t in data.get("targets", [])],
            target_cols=data.get("target_cols", []),
            metrics=data.get("metrics", {}),
            feature_importances=data.get("feature_importances")
        )