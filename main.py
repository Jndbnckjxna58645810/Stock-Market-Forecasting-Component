from src.models.train import train
from src.pipeline.build_dataset import build_dataset

train("xgb_GOOGL_2010-01-01_2025-01-01_1d.json")
train("xgb_AAPL_2005-01-01_2025-01-01_1d.json")
 