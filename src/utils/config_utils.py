import json

from src.settings.config import *
from src.utils.path_utils import resolve_path

from src.config.run_config import RunConfig
from src.config.model_config import ModelConfig

def load_config(path, directory):
    with open(resolve_path(path, directory), "r") as f: return json.load(f)

def load_run_config(path): return RunConfig.from_file(resolve_path(path, RUN_CONFIG_DIR))

def load_model_config(path): return ModelConfig.from_file(resolve_path(path, MODELS_CONFIG_DIR))