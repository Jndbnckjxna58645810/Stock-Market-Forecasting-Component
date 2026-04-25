import datetime as dt

from src.config.config import PROCESSED_DATA_DIR
from src.utils.data_utils import build_input
from src.data.loader import save_csv

save_csv(build_input("xgb_AAPL_2015-01-01_2020-01-01_1d.json", dt.date(2026, 4, 23)), PROCESSED_DATA_DIR / "example_input.csv")
