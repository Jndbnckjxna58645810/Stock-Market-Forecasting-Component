from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

@dataclass
class TargetConfig:
    name: str = "return"
    params: Dict[str, Any] = field(default_factory=lambda: {"horizon": 1, "smoothing": 1})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            name=data.get("name", "return"),
            params=data.get("params", {})
        )

@dataclass
class ModelSettings:
    name: str = "xgb"
    params: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            name=data.get("name", "xgb"),
            params=data.get("params", {}),
            hyperparameters=data.get("hyperparameters", {})
        )

@dataclass
class SplitConfig:
    type: str = "date"
    train_end: str = ""
    val_end: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            type=data.get("type", "date"),
            train_end=data.get("train_end", ""),
            val_end=data.get("val_end", "")
        )

@dataclass
class DataStrategy:
    technical: Dict[str, bool] = field(default_factory=dict)
    macro: Dict[str, bool] = field(default_factory=dict)
    processed: Dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            technical=data.get("technical", {}),
            macro=data.get("macro", {}),
            processed=data.get("processed", {})
        )