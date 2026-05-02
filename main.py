from src.config.train_config import TrainConfig
from src.config.model_config import ModelConfig
from src.config.evaluate_config import EvaluateConfig
from src.config.predict_config import PredictConfig
from src.models.registry import evaluate_models, train, predict
from src.pipeline.build_dataset import build_dataset

model_config_for_later = ModelConfig({
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

  "target": {
      "name": "return",
      "params": {
        "horizon": 1,
        "smoothing": 1,
        "log": False
      }
    },

  "macro_features": [
    {"name": "interest_rate", "source": "FEDFUNDS"},
    {"name": "inflation", "source": "CPIAUCSL"},
    {"name": "unemployment", "source": "UNRATE"}
  ],

  "hyperparameters": {
    "seq_len": 20,
    "epochs": 10,
    "units": 64,
    "batch_size": 32,
    "dropout": 0.2,
    "learning_rate": 0.001
  },

  "model": {
    "name": "lstm",
  }
})

train_config = TrainConfig({
  "ticker": "AAPL",
  "start_date": "2000-01-01",
  "end_date": "2020-01-01",
  "interval": "1d",

  "split": {
    "type": "date",
    "train_end": "2012-01-01",
    "val_end": "2016-01-01"
  },

  "data": {
  "technical": {
      "save": True,
      "path": None,
      "force_download": False
    },
    "macro": {
      "save": True,
      "path": None,
      "force_download": False
    },
    "processed": {
      "save": True,
      "path": None,
      "force_download": True
    }
  },

  "model_config": None
})

print(build_dataset("rf_AAPL_other.json"))

print(evaluate_models(EvaluateConfig({
    "models": ["lstm_AAPL_1d_20260502_110631",
               "rf_AAPL_1d_20260502_110622",
               "xgb_AAPL_1d_20260502_110622",
               train(train_config, model_config_for_later)],
    "ticker": "AAPL",
    "start_date": "2025-01-01",
    "end_date": "2026-01-01",
    "interval": "1d"
})))

print(predict(PredictConfig({
  "model_path": "rf_AAPL_1d_20260502_110622",
  "start_date": "2025-01-01",
  "end_date": "2026-01-01"
})))