import pandas as pd

from src.utils.io_utils import list_contents
from src.utils.config_utils import ensure

from src.config.model_metadata import ModelMetadata

from src.settings.config import *

def get_all_model_metadata():
    model_names = list_contents(MODELS_DIR)
    model_metadata = []

    for m in model_names:
        model_metadata.append(ModelMetadata.from_name(m))

    return model_metadata

def get_filter_options():
    metadata_list = get_all_model_metadata()

    target_sets = {} # Key: String description, Value: The actual List[TargetConfig]
    tickers = set()
    intervals = set()

    for m in metadata_list:
        tickers.add(m.ticker)
        intervals.add(m.interval)
        
        # Create a unique string for the ENTIRE list of targets
        # e.g., "1: Return(h=1), 2: Multi(h=[1,5])"
        t_str = " | ".join([get_target_string(t) for t in m.targets])
        
        if t_str not in target_sets:
            target_sets[t_str] = m.targets

    return sorted(list(tickers)), sorted(list(intervals)), target_sets

def get_target_string(target):
    t_name = target.name
    if not target.params:
        return t_name
    t_params = ", ".join([f"{k}={v}" for k, v in target.params.items()])
    return f"{t_name}({t_params})"

def get_valid_models(ticker, interval, start_date, end_date, targets):
    valid_models = []
    model_names = list_contents(MODELS_DIR)
    for model in model_names:
        m = ModelMetadata.from_name(model)

        if m.ticker != ticker: continue
        
        if m.interval != interval: continue

        if m.targets != targets: continue

        train_start, train_end = pd.to_datetime(m.start_date), pd.to_datetime(m.split.train_end)
        if (train_start <= pd.to_datetime(end_date)) and (
            train_end >= pd.to_datetime(start_date)): continue
        
        valid_models.append(model)
        
    return valid_models