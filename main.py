from src.config.train_config import TrainConfig
from src.config.model_config import ModelConfig
from src.config.evaluate_config import EvaluateConfig
from src.config.predict_config import PredictConfig
from src.config.common import *
from src.models.registry import evaluate_models, train, predict
from src.data.technical import load_technical_by_parameters
from src.data.macro import load_macro_by_parameters
from src.pipeline.apply_features import apply_features_by_parameters
from src.pipeline.apply_targets import apply_targets_by_parameters
from src.pipeline.preprocessing import merge_and_align_datasets
from src.config.common import TargetConfig

model_config_for_later = ModelConfig.from_dict({
  "features": [
    {"name": "sma", "params": {"window": 20}},
    {"name": "sma", "params": {"window": 50}},
    {"name": "ema", "params": {"window": 20}},
    {"name": "ema", "params": {"window": 50}},
    {"name": "momentum", "params": {"window": 5}},
    {"name": "volatility", "params": {"window": 10}},
    {"name": "volatility_ratio", "params": {}},
    {"name": "rsi", "params": {"window": 14}},
    {"name": "macd", "params": {}, "col_name": ["macd_line", "macd_signal"]},
    {"name": "macd_hist", "params": {}},
    {"name": "log_return", "params": {}},
    {"name": "range", "params": {}},
    {"name": "hl_position", "params": {}},
    {"name": "volume_change", "params": {}},
    {"name": "volume_sma", "params": {"window": 10}},
    {"name": "volume_ratio", "params": {}},
    {"name": "dist_sma", "params": {"window": 20}},
    {"name": "dist_sma", "params": {"window": 50}},
    {"name": "rolling_max", "params": {"window": 10}},
    {"name": "rolling_min", "params": {"window": 10}},
    {"name": "breakout_up", "params": {}},
    {"name": "breakout_down", "params": {}},
    {"name": "day_of_week", "params": {}},
    {"name": "month", "params": {}},
    {"name": "zscore_close", "params": {"window": 50}}
  ],

  "targets": [
      {"name": "multi_return", "params": {"horizons": [1,2,3,5]}},
      {"name": "multi_return", "params": {"horizons": [10]}},
      {"name": "return", "params": {"horizon": 1, "smoothing": 1, "log": False}}
  ],

  "macro_features": [
    {"name": "interest_rate", "source": "FEDFUNDS"},
    {"name": "inflation", "source": "CPIAUCSL"},
    {"name": "unemployment", "source": "UNRATE"}
  ],

  "hyperparameters": {},

  "model": {
    "name": "xgb",
    "params": {
      "n_estimators": 100,
      "max_depth": 3,
      "learning_rate": 0.001
    }
  }
})

# print(model_config_for_later.features)
# print(model_config_for_later.model.name)

model_config_new = ModelConfig(
    features=[
        {"name": "sma", "params": {"window": 20}},
        {"name": "sma", "params": {"window": 50}},
        {"name": "ema", "params": {"window": 20}},
        {"name": "ema", "params": {"window": 50}}],
    targets=[
        TargetConfig(name="multi_return", params={"horizons": [1,2,3,5]}),
        TargetConfig(name="multi_return", params={"horizons": [4]}),
        TargetConfig(name="return", params={"horizon": 1, "smoothing": 1, "log": False})
      ],
    model=ModelSettings(
        name="xgb",
        params={"n_estimators": 100,"max_depth": 3, "learning_rate": 0.001},
        hyperparameters={}),
    macro_features=[
        {"name": "interest_rate", "source": "FEDFUNDS"},
        {"name": "inflation", "source": "CPIAUCSL"},
        {"name": "unemployment", "source": "UNRATE"}])

# print(model_config_new.features)
# print(model_config_for_later.model.name)

train_config = TrainConfig.from_dict({
  "mode": "train",
  "ticker": "AAPL",
  "start_date": "2000-01-01",
  "end_date": "2025-01-01",
  "interval": "1d",

  "split": {
    "type": "date",
    "train_end": "2020-01-01",
    "val_end": "2022-01-01"
  },

  "data": {
  "technical": {
      "save": True,
      "force_download": False
    },
    "macro": {
      "save": True,
      "force_download": False
    },
    "processed": {
      "save": True,
      "force_download": False
    }
  },

  "model_config_path": "lstm_multi_return.json"
})

# technical_raw = load_technical_by_parameters(
#     train_config.ticker, train_config.start_date, train_config.end_date, train_config.interval)

# print(technical_raw)

# macro_raw = load_macro_by_parameters(
#     model_config_new.macro_features, train_config.start_date, train_config.end_date)

# print(macro_raw)

# df = merge_and_align_datasets(technical_raw, macro_raw)
# df = apply_features_by_parameters(df, model_config_new.features)

# print(df)

# df = apply_targets_by_parameters(df, model_config_new.targets)

# print(df)

#train(train_config, model_config_for_later)

#train("ex_LSTM_GOOGL_2010-01-01_2025-01-01_1d.json")

name = train(train_config, model_config_for_later)

print(predict(PredictConfig("2026-01-01", "2026-02-01", name)))

# print(evaluate_models(EvaluateConfig(["rf_AAPL_1d_20260509_173212"], "AAPL", "2025-01-01", "2026-01-01", "1d", [
#     TargetConfig(name="multi_return", params={"horizons": [1,2,3,5]}),
#     TargetConfig(name="multi_return", params={"horizons": [4]}),
#     TargetConfig(name="return", params={"horizon": 1, "smoothing": 1, "log": False})
# ])))
