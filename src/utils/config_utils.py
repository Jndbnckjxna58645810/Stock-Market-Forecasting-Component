import json

from src.config.config import *
from src.utils.path_utils import resolve_path

def load_config(path, directory):
    with open(resolve_path(path, directory), "r") as f: return json.load(f)

def load_run_config(path): return load_config(path, RUN_CONFIG_DIR)

def load_model_config(path): return load_config(path, MODELS_CONFIG_DIR)

def get_config_parameters(run):
    run_config = load_run_config(run)
    return run_config["ticker"], run_config["start_date"], run_config["end_date"], run_config["interval"], load_model_config(run_config["model_config"])